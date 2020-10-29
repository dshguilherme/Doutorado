import os

from dolfin import(FunctionSpace, TrialFunction, TestFunction, inner,
                   grad, dx, PETScMatrix, assemble, DirichletBC,
                   SLEPcEigenSolver, Function, Point, File, SubDomain,
                   near, Measure, MeshFunction, between, refine,
                   DOLFIN_EPS, FacetNormal, sqrt, lhs,
                   PETScDMCollection, project, Constant, ds, plot, File)
import mshr
import numpy as np
import matplotlib.pyplot as plt

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
        self.VO = FunctionSpace(self.overlap_mesh, "Lagrange", self.p)

        self.K, self.M, self.V = self.assemble_matrices(self.mesh)
        self.K1, self.M1, self.V1 = self.assemble_matrices(self.mesh1)
        self.K2, self.M2, self.V2 = self.assemble_matrices(self.mesh2)

    def initial_solution(self, main_boundary, left_boundary,
                         right_boundary, mode_number):
        KK = self.K
        bc = DirichletBC(self.V, Constant(0.), main_boundary)
        bc.apply(KK)
        solver = standard_solver(KK, self.M)
        solver.solve(mode_number)

        KK1 = self.K1
        bc1 = DirichletBC(self.V1, Constant(0.), left_boundary)
        bc1.apply(KK1)
        solver1 = standard_solver(KK1, self.M1)
        solver1.solve(mode_number)
    
        KK2 = self.K2
        bc2 = DirichletBC(self.V2, Constant(0.), right_boundary)
        bc2.apply(KK2)
        solver2 = standard_solver(KK2, self.M2)
        solver2.solve(mode_number)

        r, _, rx, _ = solver.get_eigenpair(mode_number)
        r1, _, rx1, _ = solver1.get_eigenpair(mode_number)
        r2, _, rx2, _ = solver2.get_eigenpair(mode_number)

        freqs = [r, r1, r2]
        vecs = [rx, rx1, rx2]
        return freqs, vecs


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
        SH = H1-L2

        return L2, H1, SH
    def build_transfer_matrices(self):
        V1 = self.V1
        V2 = self.V2
        VO = self.VO
        self.B1 = PETScDMCollection.create_transfer_matrix(V1,VO)
        self.B2 = PETScDMCollection.create_transfer_matrix(V2,VO)
        self.BO1 = PETScDMCollection.create_transfer_matrix(VO,V1)
        self.BO2 = PETScDMCollection.create_transfer_matrix(VO,V2)

    def schwarz_algorithm(self, max_iterations, main_boundary, left_robin,
                          left_dirichlet, right_robin,
                          right_dirichlet, mode_number):
        self.build_transfer_matrices()
        B1 = self.B1
        BO1 = self.BO1
        B2 = self.B2
        BO2 = self.BO2

        V1 = self.V1
        V2 = self.V2
        VO = self.VO
        freqs, vecs = self.initial_solution(main_boundary, left_dirichlet,
                            right_dirichlet, mode_number)

        r1 = freqs[1]
        r2 = freqs[2]

        M1 = self.M1
        M2 = self.M2

        rx1 = vecs[1]
        if abs(rx1.max()) < abs(rx1.min()):
            rx1 = -rx1
        rx2 = vecs[2]
        if abs(rx2.max()) < abs(rx2.min()):
            rx2 = -rx2

        iter = 0
        L2_error = np.zeros((max_iterations+1,1))
        H1_error = np.zeros((max_iterations+1,1))
        SH_error = np.zeros((max_iterations+1,1))
        r_left = np.zeros((max_iterations+1, 1))
        r_right = np.zeros((max_iterations+1, 1))

        u1 = Function(self.V1)
        u1.vector()[:] = rx1
        
        u2 = Function(self.V2)
        u2.vector()[:] = rx2

        L2, H1 = self.overlap_error_norms(u1, u2)

        L2_error[iter] = L2
        H1_error[iter] = H1
        SH_error[iter] = H1-L2
        r_left[iter] = r1
        r_right[iter] = r2
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
            bc2 = DirichletBC(V2, Constant(0.), right_dirichlet)
            bc2.apply(K2)
            
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
            r_left[iter] = r1
            r_right[iter] = r2
            L2_error[iter] = L2
            H1_error[iter] = H1
            SH_error[iter] = H1-L2
        return L2_error, H1_error, SH_error, uu1, uu2, r_left, r_right

def membrane_iterator(insides, outsides, left_outsides, left_robin,
                        right_outsides, right_robin, num_of_iterations,
                        membrane, mode_num):

    # Locally Refine the Meshes on the Overlap Region
    # for i in range(number_of_refinements):
    #     membrane.refine_meshes(insides)

    # Initial Solutions
    freqs, vecs = membrane.initial_solution(outsides,
                             left_outsides, right_outsides, mode_num)
    r = freqs[0]
    r1 = freqs[1]
    r2 = freqs[2]
    print("Target Frequency:", sqrt(r))
    print("Initial Left Domain Frequency:", sqrt(r1))
    print("Initial Right Domain Frequency:", sqrt(r2))

    rx = vecs[0]
    rx1 = vecs[1]
    rx2 = vecs[2]

    # Transfer Matrices and Overlap Solution
    membrane.build_transfer_matrices()

    u = Function(membrane.V)
    u.vector()[:] = rx
    LL, HH, SS = membrane.big_error_norms(u, insides)



    # Schwarz Algorithm
    L2, H1, SH, u1, u2, r1, r2 = membrane.schwarz_algorithm(num_of_iterations, insides,
                                                left_robin, left_outsides,
                                                right_robin, right_outsides, 0)

    print('Initial L2 Relative Error:', np.sqrt(L2[0]/LL))
    print('Initial H1 Relative Error:', np.sqrt(H1[0]/HH))
    print('Initial H1 Relative Semi-Norm Error:', np.sqrt(SH[0]/SS))
    for i in range(1,len(L2)):
        print('Iteration', i, 'L2 error:', np.sqrt(L2[i]/LL))
        print('Iteration', i, 'H1 error:', np.sqrt(H1[i]/HH))
        print('Iteration', i, 'Semi H1 error:', np.sqrt(SH[i]/SS))
    return np.sqrt(L2/LL), np.sqrt(H1/HH), np.sqrt(SH/SS), u1, u2, r1, r2

def generate_relatory(filepath, membrane, L2, H1, SH1, u, u1, u2, r, r1, r2):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    file = File(filepath+'paraview.pvd')
    file << u
    file << u1
    file << u2

    fig, ax = plt.subplots()
    ax.plot(L2, label='Relative L2 Error')
    ax.plot(H1, label='Relative H1 Error')
    ax.legend(loc='upper right')
    ax.set_ylabel('Relative Error', fontsize=18)
    ax.set_xlabel('Iteration Steps', fontsize=18)
    ax.grid(b=True, ls='-.')
    fig.tight_layout()

    plt.savefig(filepath +'errornorms.png')
    plt.close()

    fig, ax = plt.subplots()
    c = plot(u1)
    fig.colorbar(c)
    ax.set_title('Mode of Vibration on Left Domain')
    plt.savefig(filepath +'u1.png')
    plt.close()

    fig, ax = plt.subplots()
    c = plot(u2)
    fig.colorbar(c)
    ax.set_title('Mode of Vibration on Right Domain')
    plt.savefig(filepath +'u2.png')
    plt.close()

    fig, ax = plt.subplots()
    b = plot(u1)
    c = plot(u2)
    fig.colorbar(c)
    ax.set_title('Juxtaposition of Left and Right Domains')
    plt.savefig(filepath +'u1+u2.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(np.sqrt(abs((r1))), label='Left Domain Frequency')
    ax.plot((np.sqrt(abs(r2))), label='Right Domain Frequency')
    ax.axhline(y=np.sqrt(r), label='Target Frequency', color='r')
    ax.legend(loc='upper right')
    ax.set_ylabel('Natural Frequency $\omega _n$')
    ax.set_xlabel('Iteration Steps')
    plt.savefig(filepath +'eigenvalues.png')
    plt.close()

    fig, ax = plt.subplots()
    d = plot(u)
    fig.colorbar(d)
    ax.set_title('Reference Mode of Vibration')
    plt.savefig(filepath+'target solution')
    plt.close()

    B1 = membrane.B1
    B2 = membrane.B2
    
    V12 = membrane.VO
    u12 = Function(V12)
    u21 = Function(V12)

    u12.vector()[:] = B1*u1.vector()
    u21.vector()[:] = B2*u2.vector()
    
    fig, ax = plt.subplots()
    e = plot((u12-u21))
    fig.colorbar(e)
    ax.set_title('Absolute Difference in the Overlap Region')
    plt.savefig(filepath+'OverlapDiff.png')
    plt.close()

    fig, ax = plt.subplots()
    plot(membrane.mesh)
    ax.set_title('Main Domain Mesh')
    plt.savefig(filepath+'mainmesh.png')
    plt.close()

    fig, ax = plt.subplots()
    plot(membrane.mesh1)
    ax.set_title('Left Domain Mesh')
    plt.savefig(filepath+'leftmesh.png')
    plt.close()

    fig, ax = plt.subplots()
    plot(membrane.mesh2)
    ax.set_title('Right Domain Mesh')
    plt.savefig(filepath+'rightmesh.png')
    plt.close()

    fig, ax = plt.subplots()
    plot(membrane.overlap_mesh)
    ax.set_title('Transfer/Overlap Domain Mesh')
    plt.savefig(filepath+'overlapmesh.png')