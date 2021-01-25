from dolfin import(SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
import mshr
import matplotlib.pyplot as plt
import numpy as np

from Membrane import Membrane

class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    
class LeftDirichlet(SubDomain):
    def __init__(self, h):
        self.h = h
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        x_left = near(x[0], 0.)
        y_top = x[1] > self.h - DOLFIN_EPS
        y_bot = x[1] < 0 +DOLFIN_EPS
        cond = x_left or (y_top or y_bot)
        return on_boundary and cond

class LeftRobin(SubDomain):
    def __init__(self, b, o, L):
        self.b = b
        self.o = o
        self.L = L
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self.b+self.o*self.L)

class RightDirichlet(SubDomain):
    def __init__(self, h, L):
        self.h = h
        self.L = L
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        x_right = near(x[0], self.L)
        y_top = x[1] > self.h - DOLFIN_EPS
        y_bot = x[1] < 0 + DOLFIN_EPS
        cond = x_right or (y_top or y_bot)        
        return on_boundary and cond

class RightRobin(SubDomain):
    def __init__(self, b):
        self.b = b
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self.b)

class Overlap(SubDomain):
    def __init__(self, b, o, L):
        self.b = b
        self.o = o
        self.L = L
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return between(x[0], [self.b, self.b+self.o*self.L])


class Rectangular(Membrane):
    def __init__(self, mesh_resolution,
                    polynomial_degree, adjust, h, L, o):

        self.h = h
        self.L = L
        self.o = o
        self.b = L*(1-o)/2
        b = self.b

        domain = mshr.Rectangle(Point(0., 0.), Point(L, h))
        domain1 = mshr.Rectangle(Point(0., 0.,), Point(b+L*o, h))
        domain2 = mshr.Rectangle(Point(b, 0.), Point(L, h))

        self.insides = Overlap(b, o, L)
        self.outsides = OnBoundary()
        self.left_dirichlet = LeftDirichlet(h)
        self.left_robin = LeftRobin(b, o, L)
        self.right_dirichlet = RightDirichlet(h, L)
        self.right_robin = RightRobin(b)
        boundaries = list([self.insides, self.outsides, self.left_dirichlet,
                                self.left_robin, self.right_dirichlet, self.right_robin])

        Membrane.__init__(self, domain, domain1, domain2, mesh_resolution,
                            boundaries, polynomial_degree, adjust)

class LShapedLeftDirichlet(SubDomain):
    def __init__(self, h1):
        self.h1 = h1
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
            y_top = near(x[1], self.h1)
            x_left = near(x[0], 0.)
            y_bot = near(x[1], 0.)
            cond = (y_top or x_left) or (y_bot)
            return on_boundary and cond

class LShapedLeftRobin(SubDomain):
    def __init__(self, L1, h1, b):
        self.L1 = L1
        self.h1 = h1
        self.b = b
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        x_right = between(x[0], [self.L1, self.L1+self.b])
        y_right = between(x[1], [0, self.h1])
        cond = x_right and y_right
        return on_boundary and cond       

class LShapedRightDirichlet(SubDomain):
    def __init__(self, L1, L2, h1, h2):
        self.L1 = L1
        self.L2 = L2
        self.h1 = h1
        self.h2 = h2
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        y_top = near(x[1], self.h1)
        y_bot = near(x[1], -(self.h2-self.h1))
        y_clamp = y_top or y_bot

        x_right = near(x[0], self.L1+self.L2)
        x_left = near(x[0], self.L1)
        y_left = x[1] < 0 +DOLFIN_EPS
        x_clamp = x_right or (x_left and y_left)

        return on_boundary and (y_clamp or x_clamp)    

class LShapedRightRobin(SubDomain):
    def __init__(self, L1, h1, b):
        self.L1 = L1
        self.h1 = h1
        self.b = b
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        if near(x[1], self.h1):
            x_top = between(x[0], [self.L1, self.L1+self.b])
            return on_boundary and x_top
        elif near(x[0], self.L1):
            y_left = between(x[1], [0., self.h1])
            return on_boundary and y_left           

class LShapedOverlap(SubDomain):
    def __init__(self, L1, h1, b):
        self.L1 = L1
        self.h1 = h1
        self.b = b
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        if between(x[1], [0, self.h1]):
            xx = (self.b/self.h1)*x[1] + self.L1
            x_cond = between(x[0], [self.L1, xx])
            return x_cond
        else:
            return False        

class TShapedLeftDirichlet(SubDomain):
    def __init__(self, L):
        self.L = L
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], [0, self.L])

class TShapedLeftRobin(SubDomain):
    def __init__(self, L, c):
        self.L = L
        self.c = c
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], [self.L, self.L +self.c])

class TShapedRightDirichlet(SubDomain):
    def __init__(self, L, h):
        self.L = L
        self.h = h
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        y_top = near(x[1], self.h+self.h)
        y_bot = near(x[1], -self.h)
        y_cond = y_top or y_bot

        x_right = near(x[0], self.L+self.L)
        x_left = near(x[0], self.L) and not between(x[1], [0., self.h])
        x_cond = x_right or x_left

        cond = y_cond or x_cond

        return on_boundary and cond

class TShapedRightRobin(SubDomain):
    def __init__(self, L, h):
        self.L = L
        self.h = h
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        x_cond = near(x[0], self.L)
        y_cond = between(x[1], [0., self.h])
        cond = x_cond and y_cond
        return on_boundary and cond

class TShapedOverlap(SubDomain):
    def __init__(self, L, h, c):
        self.L = L
        self.h = h
        self.c = c
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        x_cond = between(x[0], [self.L, self.L +self.c])
        y_cond = between(x[1], [0., self.h])
        return x_cond and y_cond

class FinShapedLeftDirichlet(SubDomain):
    def __init__(self, frontier_x):
        self.fx = frontier_x
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
            x_cond = x[0] >= self.fx -DOLFIN_EPS
            return on_boundary and not x_cond

class FinShapedLeftRobin(SubDomain):
    def __init__(self, frontier_x):
        self.fx = frontier_x
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return on_boundary and x[0] >= self.fx - DOLFIN_EPS   

class FinShapedRightDirichlet(SubDomain):
    def __init__(self, frontier_x):
        self.fx = frontier_x
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        x_cond = x[0] <= self.fx +DOLFIN_EPS
        return on_boundary and not x_cond       

class FinShapedRightRobin(SubDomain):
    def __init__(self, frontier_x):
        self.fx = frontier_x
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        return on_boundary and x[0] <= self.fx +DOLFIN_EPS   

class FinOverlap(SubDomain):
    def __init__(self, L, h, a, b, foot, frontier_x):
        self.a = a
        self.b = b
        self.f = foot
        self.L = L
        self.fx = frontier_x
        self.h = h
        SubDomain.__init__(self)
    
    def inside(self, x, on_boundary):
        cond1 = x[0] > -self.a
        cond2 = x[0] < -self.a -self.f +self.L
        if (cond1 and cond2):
            if between(x[0], [-self.a, -self.fx]):
                yy = np.sqrt((self.b**2)*(1 - ((x[1]/self.a)**2)))
                return between(x[1], [-yy, +yy])
            
            else:
                return between(x[1], [-self.h/2, +self.h/2])
        else:
            return False     

class LShaped(Membrane):
    def __init__(self, mesh_resolution, polynomial_degree, adjust, L1, L2, h1, h2, b):

        self.L1 = L1
        self.L2 = L2
        self.h1 = h1
        self.h2 = h2
        self.b = b

        p1 = Point(0., 0.)
        p2 = Point(L1, h1)
        p3 = Point(L1+b, h1)
        p4 = Point(L1, 0.)

        domain1 = mshr.Rectangle(p1, p2) +mshr.Polygon([p4, p3, p2])
        
        p5 = Point(L1, -(h2-h1))
        p6 = Point(L1+L2, h1)

        domain2 = mshr.Rectangle(p5,p6)
        domain = domain1+domain2
        
        self.insides = LShapedOverlap(self.L1, self.h1, self.b)
        self.outsides = OnBoundary()
        self.left_dirichlet = LShapedLeftDirichlet(self.h1)
        self.left_robin = LShapedLeftRobin(self.L1, self.h1, self.b)
        self.right_dirichlet = LShapedRightDirichlet(self.L1, self.L2, self.h1, self.h2)
        self.right_robin = LShapedRightRobin(self.L1, self.h1, self.b)
        boundaries = list([self.insides, self.outsides, self.left_dirichlet,
                                self.left_robin, self.right_dirichlet, self.right_robin])

        Membrane.__init__(self, domain, domain1, domain2, mesh_resolution, boundaries,
                            polynomial_degree, adjust)

class TShaped(Membrane):
    def __init__(self, mesh_resolution, polynomial_degree, adjust, L, h, c):
        self.h = h
        self.L = L
        self.c = c

        P1 = Point(0., 0.)
        P2 = Point(L+c, h)
        domain1 = mshr.Rectangle(P1,P2)
        
        P3 = Point(L, -h)
        P4 = Point(L+L, h+h)
        domain2 = mshr.Rectangle(P3, P4)
        domain = domain1+domain2

        self.insides = TShapedOverlap(self.L, self.h, self.c)
        self.outsides = OnBoundary()
        self.left_dirichlet = TShapedLeftDirichlet(self.L)
        self.left_robin = TShapedLeftRobin(self.L, self.c)
        self.right_dirichlet = TShapedRightDirichlet(self.L, self.h)
        self.right_robin = TShapedRightRobin(self.L, self.h)
        boundaries = list([self.insides, self.outsides, self.left_dirichlet,
                                self.left_robin, self.right_dirichlet, self.right_robin])

        Membrane.__init__(self, domain, domain1, domain2, mesh_resolution, boundaries,
                            polynomial_degree, adjust)

class FinShaped(Membrane):
    def __init__(self, mesh_resolution, polynomial_degree, adjust, a, b, L, h, foot):
        self.center = Point(0.,0.)
        self.a = a
        self.b = b
        self.L = L
        if (L < foot +a -a*np.sqrt((1- (h**2)/(4*(b**2))))):
            print('ERROR: Selected L is too small.')
        elif (L > foot +a +a*np.sqrt((1- (h**2)/(4*(b**2))))):
            print('ERROR: Select L is too big.')
        self.h = h
        self.f = foot
        self.fx = -a*np.sqrt((1- (h**2)/(4*(b**2))))
        self.fy = h/2

        P1 = Point(-a -foot, -h/2)
        P2 = Point(-a -foot +L, h/2)
        domain1 = mshr.Rectangle(P1,P2)
        center =  Point(0., 0.)
        domain2 = mshr.Ellipse(center, a, b)
        domain = domain1 +domain2

        self.insides = FinOverlap(self.L, self.h, self.a, self.b,
                                   self.f, self.fx)
        self.outsides = OnBoundary()
        self.left_dirichlet = FinShapedLeftDirichlet(self.fx)
        self.left_robin = FinShapedLeftRobin(self.fx)
        self.right_dirichlet = FinShapedRightDirichlet(self.fx)
        self.right_robin = FinShapedRightRobin(self.fx)
        
        boundaries = list([self.insides, self.outsides, self.left_dirichlet,
                            self.left_robin, self.right_dirichlet, self.right_robin])       

        Membrane.__init__(self, domain, domain1, domain2, mesh_resolution,
                            boundaries, polynomial_degree, adjust)