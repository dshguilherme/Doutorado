<<<<<<< Updated upstream
from dolfin import(FunctionSpace, TrialFunction, TestFunction, inner,
                   grad, dx, PETScMatrix, assemble, DirichletBC,
                   SLEPcEigenSolver, Function, Point, File, SubDomain,
                   near, Measure, MeshFunction, between, refine)
import mshr

=======
import numpy as np

from dolfin import(FunctionSpace, TrialFunction, TestFunction, inner,
                   grad, dx, PETScMatrix, assemble, DirichletBC,
                   SLEPcEigenSolver, Function, Point, File, SubDomain,
                   near, Measure, MeshFunction, between, refine,
                   DOLFIN_EPS, FacetNormal, sqrt, lhs)
import mshr


>>>>>>> Stashed changes
class RectangularMembrane:
    def __init__(self, point1, point2, mesh_resolution, polynomial_order):
       self.p1 = point1
       self.p2 = point2
       self.p = polynomial_order
       self.resolution = mesh_resolution
       self.domain = mshr.Rectangle(point1,point2)
       self.mesh = mshr.generate_mesh(self.domain, mesh_resolution)

    def assemble_matrices(self):
        M = self.assemble_mass()
        K, V = self.assemble_stiffness()
        return K, M, V

    def assemble_mass(self):
        V = FunctionSpace(self.mesh, "Lagrange", self.p)
        u = TrialFunction(V)
        v = TestFunction(V)
        m = u*v*dx
        M = PETScMatrix()
        assemble(m, tensor=M)
        return M
    
    def assemble_stiffness(self):
        V = FunctionSpace(self.mesh, "Lagrange", self.p)
        u = TrialFunction(V)
        v = TestFunction(V)
        k = inner(grad(u), grad(v))*dx
        K = PETScMatrix()
        assemble(k, tensor=K)
        return K, V

    def assemble_lui_stiffness(self, g, f, robin_boundary):
        V = FunctionSpace(self.mesh, "Lagrange", self.p)
        u = TrialFunction(V)
        v = TestFunction(V)
        robin = MeshFunction('size_t', self.mesh, 
                             self.mesh.topology().dim()-1)
        robin_boundary.mark(robin, 1)
        ds = Measure('ds', subdomain_data=robin)
        k = inner(grad(u), grad(v))*dx -(1-g)*(u.dx(0))*v*ds(1) +f*u*v*ds(1)
        K = PETScMatrix()
        assemble(k, tensor=K)
        return K, V

    def insides(self):
        point1 = self.p1
        point2 = self.p2
        class Inside(SubDomain):
            def inside(self, x, on_boundary):
                return between(x[0], [point1.x(), point2.x()])
        subdomain = Inside()
        return subdomain

    def left(self):
        class Left(SubDomain):
            point1 = self.p1
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], self.point1.x())
        subdomain = Left()
        return subdomain

    def right(self):
        class Right(SubDomain):
            point2 = self.p2
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], self.point2.x())
        subdomain = Right()
        return subdomain    

    def top(self):
        class Top(SubDomain):
            point2 = self.p2
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], self.point2.y())
        subdomain = Top()
        return subdomain    
    
    def bottom(self):
        class Bottom(SubDomain):
            point1 = self.p1
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], self.point1.y())
        subdomain = Bottom()
        return subdomain

    def boundary(self):
        class OnBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary
        subdomain = OnBoundary()
        return subdomain

    def refine_mesh(self, subdomain):
        markers = MeshFunction("bool", self.mesh, self.mesh.topology().dim())
        subdomain.mark(markers, True)
        self.mesh = refine(self.mesh, markers)
    
    def refine_boundary(self, boundary_subdomain):
        markers = MeshFunction("bool", self.mesh, self.mesh.topology().dim()-1)
        boundary_subdomain.mark(markers, True)
        self.mesh = refine(self.mesh, markers)

def standard_solver(K,M):
    solver = SLEPcEigenSolver(K,M)
    solver.parameters['spectrum'] = 'smallest magnitude'
    solver.parameters['tolerance'] = 1e-4
    solver.parameters['problem_type'] = 'pos_gen_non_hermitian'
<<<<<<< Updated upstream
    return solver
=======
    return solver

class StandardFin:
    def __init__(self, center, a, b, foot, L, h, mesh_resolution, polynomial_order):
        self.center = center
        self.a = a
        self.b = b
        self.foot = foot
        self.L = L
        self.h = h
        self.p = polynomial_order
        self.resolution = mesh_resolution
        
        foot_x = center.x() -a -foot
        foot_y = center.y() -0.5*h
        head_x = foot_x + L
        head_y = foot_y + h
        self.rec = mshr.Rectangle(Point(foot_x, foot_y), Point(head_x, head_y))
        self.elli = mshr.Ellipse(center, a, b)
        
        self.domain = self.rec + self.elli
        self.left = self.rec - self.elli
        self.right = self.elli - self.rec
        self.overlap = self.domain - (self.left + self.right)

        self.mesh = mshr.generate_mesh(self.domain, mesh_resolution)
        self.rectangle_mesh = mshr.generate_mesh(self.rec, mesh_resolution)
        self.ellipse_mesh = mshr.generate_mesh(self.elli, mesh_resolution)
        self.overlap_mesh = mshr.generate_mesh(self.overlap, mesh_resolution)

        self.frontier_x = -a * np.sqrt(1 - (h / (2*b) )**2 )
        self.frontier_y = h/2


    def assemble_matrices(self, mesh):
        M = self.assemble_mass(mesh)
        K, V = self.assemble_stiffness(mesh)
        return K, M, V

    def assemble_mass(self, mesh):
        V = FunctionSpace(mesh, "Lagrange", self.p)
        u = TrialFunction(V)
        v = TestFunction(V)
        m = u*v*dx
        M = PETScMatrix()
        assemble(m, tensor=M)
        return M
    
    def assemble_stiffness(self, mesh):
        V = FunctionSpace(mesh, "Lagrange", self.p)
        u = TrialFunction(V)
        v = TestFunction(V)
        k = inner(grad(u), grad(v))*dx
        K = PETScMatrix()
        assemble(k, tensor=K)
        return K, V

    def assemble_lui_stiffness(self, g, f, mesh, robin_boundary):
        V = FunctionSpace(mesh, "Lagrange", self.p)
        u = TrialFunction(V)
        v = TestFunction(V)
        n = FacetNormal(mesh)
        robin = MeshFunction('size_t', mesh, 
                             mesh.topology().dim()-1)
        robin_boundary.mark(robin, 1)
        ds = Measure('ds',subdomain_data=robin)
        a = inner(grad(u),grad(v))*dx
        b = (1-g)*(inner(grad(u),n))*v*ds(1)
        c = f*u*v*ds(1)
        k = lhs(a -b +c)
        K = PETScMatrix()
        assemble(k, tensor=K)
        return K, V        

    def overlap_region(self):
        fx = self.frontier_x
        a = self.a
        b = self.b
        class Overlap(SubDomain):
            def inside(self, x, on_boundary):
                x_cond = between(x[0], [-a, fx])
                yy = x[1]**2
                tmp = 1 - (x[0]/a)**2
                dd = b*tmp*b
                y_cond = (yy <= dd)
                return x_cond and y_cond
        subdomain = Overlap()
        return subdomain
    
    def rec_outside(self):
        fx = self.frontier_x
        class Outside(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] <= fx +DOLFIN_EPS
        subdomain = Outside()
        return subdomain
    
    def rec_robin(self):
        fx = self.frontier_x
        class RecRobin(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] >= fx - DOLFIN_EPS
        subdomain = RecRobin()
        return subdomain
           
    
    def elli_outside(self):
        fx = self.frontier_x
        class Outside(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] >= fx - DOLFIN_EPS
        subdomain = Outside()
        return subdomain
    
    def elli_robin(self):
        return self.rec_outside()
    
    def boundary(self):
        class OnBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary
        subdomain = OnBoundary()
        return subdomain
    
    def clamp(self):
        foot_x = self.center.x() -self.a -self.foot
        class Clamp(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], foot_x)
        subdomain = Clamp()
        return subdomain

    def refine_meshes(self, subdomain):
        rec_marker = MeshFunction("bool", self.rectangle_mesh,
                                   self.rectangle_mesh.topology().dim())
        elli_marker = MeshFunction("bool", self.ellipse_mesh,
                                    self.ellipse_mesh.topology().dim())
        subdomain.mark(rec_marker, True)
        subdomain.mark(elli_marker, True)
        self.rectangle_mesh = refine(self.rectangle_mesh, rec_marker)
        self.ellipse_mesh = refine(self.ellipse_mesh, elli_marker)

>>>>>>> Stashed changes
