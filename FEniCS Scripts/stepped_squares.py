from dolfin import(SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
import mshr
import matplotlib.pyplot as plt
import numpy as np

from Membrane import (Membrane, standard_solver, membrane_iterator,
                         generate_relatory)


h = 1.
L = 1.
c = 0.3
r = h/4 

P1 = Point(0., h)
P2 = Point(L+c, h+h)
left_rectangle = mshr.Rectangle(P1, P2)

C1 = Point(L+c, h+h-r)
C2 = Point(L+c, h+r)

minus_circle = mshr.Circle(C1, r)
plus_circle = mshr.Circle(C2, r)

domain1 = (left_rectangle +plus_circle) -minus_circle
domain2 = mshr.Rectangle(Point(L, 0.), Point(L+L, h+h+h))
domain = domain1+domain2

membrane = Membrane(domain, domain1, domain2,
                    mesh_resolution=12.7, polynomial_degree=1)

# Define SubDomains

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class LeftDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], [0, L])

class LeftRobin(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > L - DOLFIN_EPS)

class RightDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        y_top = near(x[1], h+h+h)
        y_bot = near(x[1], 0.)
        y_cond = y_top or y_bot

        x_right = near(x[0], L+L)
        x_left = not between(x[0], [h, h+h])
        x_cond = x_right or x_left

        cond = y_cond or x_cond

        return on_boundary and cond

class RightRobin(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], [h, h+h])

class Overlap(SubDomain):
    def inside(self, x, on_boundary):
        if between(x[1], [h, h+h]):
            if between(x[1], [h, h+0.5*h]):
                if between(x[0], [L, L+c]):
                    return True
                else:
                    xx = (x[0] -C2.x())**2
                    yy = (x[1] -C2.y())**2
                    rr = r**2
                    return (xx + yy) <= rr
            else:
                if between(x[0], [L, L+c-r]):
                    return True
                else:
                    xx = (x[0] -C1.x())**2
                    yy = (x[1] -C1.y())**2
                    rr = r**2
                    circ_cond = (xx +yy >= rr)
                    return circ_cond and (x[0] < L+c +DOLFIN_EPS)

insides = Overlap()
outsides = OnBoundary()
left_outsides = LeftDirichlet()
left_robin = LeftRobin()
right_outsides = RightDirichlet()
right_robin = RightRobin()

# Solution

freqs, vecs = membrane.initial_solution(outsides, left_outsides,
                             right_outsides, mode_number=0)
u = Function(membrane.V)
u.vector()[:] = vecs[0]
r = freqs[0]
L2, H1, SH1, u1, u2, r1, r2 = membrane_iterator(insides=insides, 
                                outsides=outsides,
                                left_outsides=left_outsides,
                                left_robin=left_robin,
                                right_outsides=right_outsides,
                                right_robin=right_robin,
                                num_of_iterations=15,
                                membrane=membrane, mode_num=0)

generate_relatory('stepped/', membrane, L2, H1, SH1, u,
                     u1, u2, r, r1, r2)