from dolfin import(FunctionSpace, TrialFunction, TestFunction, inner,
                   grad, dx, PETScMatrix, assemble, DirichletBC,
                   SLEPcEigenSolver, Function, Point, File, SubDomain,
                   near, Measure, MeshFunction, between, refine)
import mshr

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
    return solver