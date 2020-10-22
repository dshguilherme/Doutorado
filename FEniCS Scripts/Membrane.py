from dolfin import(FunctionSpace, TrialFunction, TestFunction, inner,
                   grad, dx, PETScMatrix, assemble, DirichletBC,
                   SLEPcEigenSolver, Function, Point, File, SubDomain,
                   near, Measure, MeshFunction, between, refine,
                   DOLFIN_EPS, FacetNormal, sqrt, lhs,
                   PETScDMCollection)
import mshr

def standard_solver(K,M):
    solver = SLEPcEigenSolver(K,M)
    solver.parameters['spectrum'] = 'smallest magnitude'
    solver.parameters['tolerance'] = 1e-4
    solver.parameters['problem_type'] = 'pos_gen_non_hermitian'
    return solver

class Membrane:
    def __init__(self, domain, domain1, domain2, mesh_resolution,
                 polynomial_order):
        self.domain = domain
        self.domain1 = domain1
        self.domain2 = domain2
        self.left = domain1 - domain2
        self.right = domain2 - domain1
        self.overlap_domain = domain - (self.left +self.right)

        self.mesh = mshr.generate_mesh(domain, mesh_resolution)
        self.mesh1 = mshr.generate_mesh(domain1, mesh_resolution)
        self.mesh2 = mshr.generate_mesh(domain2, mesh_resolution)
        self.overlap_mesh = mshr.generate_mesh(self.overlap_domain,
                                               mesh_resolution)
        
        self.p = polynomial_order
        self.V = FunctionSpace(self.mesh, "Lagrange", self.p)
        self.V1 = FunctionSpace(self.mesh1, "Lagrange", self.p)
        self.V2 = FunctionSpace(self.mesh2, "Lagrange", self.p)
        self.VO = FunctionSpace(self.overlap_mesh, "Lagrange", self.p)
        
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

    def refine_meshes(self, subdomain):
        sub1_marker = MeshFunction("bool", self.mesh1,
                                     self.mesh1.topology().dim())
        sub2_marker = MeshFunction("bool", self.mesh2,
                                     self.mesh2.topology().dim())
        subdomain.mark(sub1_marker, True)
        subdomain.mark(sub2_marker, True)
        self.mesh1 = refine(self.mesh1, sub1_marker)
        self.mesh2 = refine(self.mesh2, sub2_marker)
    
    def overlap_error_norms(self, u1, u2, B1, B2):
        V12 =  self.VO

        u12 = Function(V12)
        u21 = Function(V12)

        u12.vector()[:] = B1*u1.vector()
        u21.vector()[:] = B2*u2.vector()

        error = ((u12-u21)**2)*dx
        L2 = assemble(error)

        semi_error = inner(grad(u12-u21), grad(u12-u21))*dx
        H1 = L2 + assemble(semi_error)
        return L2, H1
    
    def big_error_norms(self, u, overlap_subdomain):
        V = FunctionSpace(self.mesh, "Lagrange", self.p)
        
        uu = Function(V)
        uu.vector()[:] = u.vector()

        overlap_marker = MeshFunction('size_t', self.mesh,
                                        self.mesh.topology().dim())
        overlap_subdomain.mark(overlap_marker, 1)

        dxo = Measure('dx', subdomain_data = overlap_marker)
        
        error = (uu**2)*dxo
        semi_error = inner(grad(uu),grad(uu))*dxo
        
        L2 = assemble(error)
        H1 = L2 + assemble(semi_error)

        return L2, H1

