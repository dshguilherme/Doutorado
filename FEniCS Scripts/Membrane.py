from dolfin import(FunctionSpace, TrialFunction, TestFunction, inner,
                   grad, dx, PETScMatrix, assemble, DirichletBC,
                   SLEPcEigenSolver, Function, Point, File, SubDomain,
                   near, Measure, MeshFunction, between, refine,
                   DOLFIN_EPS, FacetNormal, sqrt, lhs,
                   PETScDMCollection, project, Constant, ds)
import mshr
import numpy as np

def standard_solver(K,M):
    solver = SLEPcEigenSolver(K,M)
    solver.parameters['spectrum'] = 'smallest magnitude'
    solver.parameters['tolerance'] = 1e-4
    solver.parameters['problem_type'] = 'pos_gen_non_hermitian'
    return solver

class Membrane:
    def __init__(self, domain, domain1, domain2, mesh_resolution,
                 polynomial_degree):
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
        
        self.p = polynomial_degree
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
    
    def overlap_error_norms(self, u1, u2):
        V12 =  self.VO
        B1 = self.B1
        B2 = self.B2

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
    def build_transfer_matrices(self):
        V1 = self.V1
        V2 = self.V2
        VO = self.VO
        self.B1 = PETScDMCollection.create_transfer_matrix(V1,VO)
        self.B2 = PETScDMCollection.create_transfer_matrix(V2,VO)
        self.BO1 = PETScDMCollection.create_transfer_matrix(VO,V1)
        self.BO2 = PETScDMCollection.create_transfer_matrix(VO,V2)

    def schwarz_algorithm(self, max_iterations, initial_u1, initial_u2,
                          left_robin, left_dirichlet, right_robin,
                          right_dirichlet, M1, M2):
        self.build_transfer_matrices()
        B1 = self.B1
        BO1 = self.BO1
        B2 = self.B2
        BO2 = self.BO2

        V1 = self.V1
        V2 = self.V2
        VO = self.VO
        
        rx1 = initial_u1.vector()
        rx2 = initial_u2.vector()

        iter = 0
        L2_error = np.zeros((max_iterations+1,1))
        H1_error = np.zeros((max_iterations+1,1))
        L2, H1 = self.overlap_error_norms(initial_u1, initial_u2)

        L2_error[iter] = L2
        H1_error[iter] = H1
        while(iter < max_iterations):
            #Step 1: do f1 = du/dn2, g1 = u2
            uu2 = Function(V2)
            if (np.absolute(rx2.max()) < np.absolute(rx2.min())):
                uu2.vector()[:] = -rx2
            else:
                uu2.vector()[:] = rx2

            uu21 = Function(VO)
            uu21.vector()[:] = B2*uu2.vector()

            g1 = Function(V1)
            g1.vector()[:] = BO1*uu21.vector()

            du2 = Function(V2)
            n2 = FacetNormal(self.mesh2)
            du2 = assemble(inner(grad(uu2),n2)*ds)
            du2 = project(du2, V2)
            du2O = Function(VO)
            du2O.vector()[:] = B2*du2.vector()

            f1 = Function(V1)
            f1.vector()[:] = BO1*du2O.vector()

            #Step 2: Solve for u1
            boundary = left_robin
            K1, _ = self.assemble_lui_stiffness(g=g1, f=f1, mesh=self.mesh1,
                                                robin_boundary=boundary)
            bc1 = DirichletBC(V1, Constant(0.), left_dirichlet)
            bc1.apply(K1)
            solver1 = standard_solver(K1,M1)
            solver1.solve(0)
            r1, _, rx1, _ = solver1.get_eigenpair(0)

            #Step 3: do f1 = du1/dn1, g2 = u1
            uu1 = Function(V1)
            if (np.absolute(rx1.max()) < np.absolute(rx1.min())):
                uu1.vector()[:] = -rx1
            else:
                uu1.vector()[:] = rx1

            uu12 = Function(VO)
            uu12.vector()[:] = B1*uu1.vector()

            g2 = Function(V2)
            g2.vector()[:] = BO2*uu12.vector()

            du1 = Function(V1)
            n1 = FacetNormal(self.mesh1)
            du1 = assemble(inner(grad(uu1), n1)*ds)
            du1 = project(du1, V1)

            du12 = Function(VO)
            du12.vector()[:] = B1*du1.vector()

            f2 = Function(V2)
            f2.vector()[:] = BO2*du12.vector()

            #Step 4: solve for u2
            boundary = right_robin
            K2, _ = self.assemble_lui_stiffness(g=g2, f=f2, mesh=self.mesh2,
                                                robin_boundary=boundary)
            
            solver2 = standard_solver(K2, M2)
            solver2.solve(0)
            r2, _, rx2, _ = solver2.get_eigenpair(0)

            #Step 5: Calculate the L2 norm for convergence
            uu21.vector()[:] = B2*uu2.vector()
            uu12.vector()[:] = B1*uu1.vector()
            error = ((uu12-uu21)**2)*dx
            L2 = assemble(error)

            semi_error = inner(grad(uu12 - uu21), grad(uu12 - uu21))*dx
            H1 = L2 + assemble(semi_error)
            iter += 1
            L2_error[iter] = L2
            H1_error[iter] = H1
        return L2_error, H1_error, uu1, uu2

def membrane_iterator(insides, outsides, left_outsides, left_robin,
                        right_outsides, right_robin, num_of_iterations,
                        membrane):

    # Locally Refine the Meshes on the Overlap Region
    # for i in range(number_of_refinements):
    #     membrane.refine_meshes(insides)

    # Matrices
    K, M, V = membrane.assemble_matrices(membrane.mesh)
    bc = DirichletBC(V, Constant(0.), outsides)
    bc.apply(K)

    K1, M1, V1 = membrane.assemble_matrices(membrane.mesh1)
    bc1 = DirichletBC(V1, Constant(0.), left_outsides)
    bc1.apply(K1)

    K2, M2, V2 = membrane.assemble_matrices(membrane.mesh2)
    bc2 = DirichletBC(V2, Constant(0.), right_outsides)
    bc2.apply(K2)

    # Initial Solutions
    solver = standard_solver(K,M)
    solver.solve(0)
    r, _, rx, _ = solver.get_eigenpair(0)
    print("Target Frequency:", sqrt(r))

    u = Function(V)
    if (np.absolute(rx.max()) < np.absolute(rx.min())):
            u.vector()[:] = -rx
    else:
            u.vector()[:] = rx

    solver1 = standard_solver(K1,M1)
    solver1.solve(0)
    r1, _, rx1, _ =  solver1.get_eigenpair(0)
    print("Initial Left Domain Frequency:", sqrt(r1))

    u1 = Function(V1)
    if (np.absolute(rx1.max()) < np.absolute(rx1.min())):
        u1.vector()[:] = -rx1
    else:
        u1.vector()[:] = rx1

    solver2 = standard_solver(K2,M2)
    solver2.solve(0)
    r2, _, rx2, _ =  solver2.get_eigenpair(0)
    print("Initial Right Domain Frequency:", sqrt(r2))
    u2 = Function(V2)
    if (np.absolute(rx2.max()) < np.absolute(rx2.min())):
        u2.vector()[:] = -rx2
    else:
        u2.vector()[:] = rx2

    # Transfer Matrices and Overlap Solution
    membrane.build_transfer_matrices()

    # L2, H1 = membrane.overlap_error_norms(u1, u2)
    LL, HH = membrane.big_error_norms(u, insides)



    # Schwarz Algorithm
    L2, H1, u1, u2 = membrane.schwarz_algorithm(num_of_iterations, u1, u2, 
                                                left_robin, left_outsides,
                                                right_robin, right_outsides, M1, M2)

    print('Initial L2 Relative Error:', np.sqrt(L2[0]/LL))
    print('Initial H1 Relative Error:', np.sqrt(H1[0]/HH))
    for i in range(1,len(L2)):
        print('Iteration', i, 'L2 error:', np.sqrt(L2[i]/LL))
        print('Iteration', i, 'H1 error:', np.sqrt(H1[i]/HH))


