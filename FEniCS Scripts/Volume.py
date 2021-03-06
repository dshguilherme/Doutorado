from dolfin import(FunctionSpace, TrialFunction, TestFunction, inner,
                   grad, dx, PETScMatrix, assemble, SLEPcEigenSolver,
                   Function, Point, File, SubDomain,
                   near, Measure, MeshFunction, between, refine,
                   DOLFIN_EPS, FacetNormal, sqrt, lhs,
                   PETScDMCollection, project, Constant, ds, plot, File)

import mshr
import numpy as np
import matplotlib.pyplot as plt

from Membrane import Membrane, standard_solver

class Volume(Membrane):

    def __init__(self, domain, domain1, domain2, mesh_resolution, boundaries,
                 polynomial_degree, adjust):
        self.domain = domain
        self.domain1 = domain1
        self.domain2 = domain2
        self.left = domain1 - domain2
        self.right = domain2 - domain1
        self.overlap_domain = domain - (self.left +self.right)
        self.mesh_resolution = mesh_resolution
        self.mesh = mshr.generate_mesh(domain, mesh_resolution)
        self.mesh1 = mshr.generate_mesh(domain1, mesh_resolution/2)
        self.mesh2 = mshr.generate_mesh(domain2, mesh_resolution/2)
        self.overlap_mesh = mshr.generate_mesh(self.overlap_domain,
                                               mesh_resolution)

        self.insides = boundaries[0]
        self.left_robin = boundaries[1]
        self.right_robin = boundaries[2]

        self.adjustment = adjust
        self.p = polynomial_degree
        self.VO = FunctionSpace(self.overlap_mesh, "Lagrange", self.p)

        self.K, self.M, self.V = self.assemble_matrices(self.mesh)
        self.K1, self.M1, self.V1 = self.assemble_matrices(self.mesh1)
        self.K2, self.M2, self.V2 = self.assemble_matrices(self.mesh2)    
    
    def initial_solution(self, mode_number):
        KK = self.K
        solver = standard_solver(KK, self.M)
        solver.solve(int(mode_number))

        KK1 = self.K1
        solver1 = standard_solver(KK1, self.M1)
        solver1.solve(int(mode_number))

        KK2 = self.K2
        solver2 = standard_solver(KK2, self.M2)
        solver2.solve(int(mode_number))
        r, _, rx, _ = solver.get_eigenpair(mode_number)
        r1, _, rx1, _ = solver1.get_eigenpair(mode_number)
        r2, _, rx2, _ = solver2.get_eigenpair(mode_number)

        freqs = [r, r1, r2]
        vecs = [rx, rx1, rx2]

        return freqs, vecs
    
    def calculate_volumes(self):
        dx = Measure('dx', domain = self.mesh1)
        a = Constant(1.0)*dx
        V1 = assemble(a)
        
        dx = Measure('dx', domain = self.mesh2)
        a = Constant(1.0)*dx
        V2 = assemble(a)
        
        dx = Measure('dx', domain = self.overlap_mesh)
        a = Constant(1.0)*dx
        VO = assemble(a)
        
        dx = Measure('dx', domain = self.mesh)
        a = Constant(1.0)*dx
        V = assemble(a)

        return V, V1, V2, VO

    def schwarz_algorithm(self, max_iterations, mode_number):
        
        left_robin = self.left_robin
        right_robin = self.right_robin

        self.build_transfer_matrices()
        B1 = self.B1
        BO1 = self.BO1
        B2 = self.B2
        BO2 = self.BO2

        V1 = self.V1
        V2 = self.V2
        VO = self.VO
        freqs, vecs = self.initial_solution(mode_number)

        r1 = freqs[1]
        r2 = freqs[2]

        M1 = self.M1
        M2 = self.M2

        rx1 = vecs[1]
        rx2 = vecs[2]

        if abs(rx1.max()) < abs(rx1.min()):
            rx1 = self.adjustment*rx1/(rx1.min())
        else:
            rx1 = self.adjustment*rx1/(rx1.max())

        if abs(rx2.max()) < abs(rx2.min()):
            rx2 = self.adjustment*rx2/(rx2.min())
        else:
            rx2 = self.adjustment*rx2/(rx2.max())

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
                rx2 = self.adjustment*rx2/(rx2.min())
                uu2.vector()[:] = rx2
            else:
                rx2 = self.adjustment*rx2/(rx2.max())
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
            solver1 = standard_solver(K1,M1)
            solver1.solve(10)
            r1, _, rx1, _ = solver1.get_eigenpair(mode_number)

            #Step 3: do f1 = du1/dn1, g2 = u1
            uu1 = Function(V1)
            if (np.absolute(rx1.max()) < np.absolute(rx1.min())):
                uu1.vector()[:] = self.adjustment*rx1/(rx1.min())
            else:
                uu1.vector()[:] = self.adjustment*rx1/(rx1.max())

            uu12 = Function(VO)
            uu12.vector()[:] = B1*uu1.vector()

            g2 = Function(V2)
            g2.vector()[:] = BO2*uu12.vector()

            du1 = Function(V1)
            n1 = FacetNormal(self.mesh1)
            du1 = assemble(inner(grad(uu1),n1)*ds)
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
            solver2.solve(10)
            r2, _, rx2, _ = solver2.get_eigenpair(mode_number)
            
            uu2 = Function(V2)
            if (np.absolute(rx2.max()) < np.absolute(rx2.min())):
                rx2 = self.adjustment*rx2/(rx2.min())
                uu2.vector()[:] = rx2
            else:
                rx2 = self.adjustment*rx2/(rx2.max())
                uu2.vector()[:] = rx2
            #Step 5: Calculate the L2 norm for convergence
            uu21.vector()[:] = B2*uu2.vector()
            uu12.vector()[:] = B1*uu1.vector()
            error = ((uu12-uu21)**2)*dx
            L2 = assemble(error)

            semi_error = inner(grad(uu12)-grad(uu21), grad(uu12)-grad(uu21))*dx
            H1 = L2 + assemble(semi_error)
            iter += 1
            r_left[iter] = r1
            r_right[iter] = r2
            L2_error[iter] = L2
            H1_error[iter] = H1
            SH_error[iter] = H1-L2
        return L2_error, H1_error, SH_error, uu1, uu2, r_left, r_right
       
    def report_results(self, num_of_iterations, mode_num, supress_results=False):
        freqs, vecs = self.initial_solution(mode_number=mode_num)

        A, A1, A2, AO = self.calculate_volumes()

        r = freqs[0]
        r1 = freqs[1]
        r2 = freqs[2]
        print("Target Frequency:", sqrt(r))
        print("Initial Left Domain Frequency:", sqrt(r1))
        print("Initial Right Domain Frequency:", sqrt(r2))

        rx = vecs[0]
        rx1 = vecs[1]
        rx2 = vecs[2]
        
        self.build_transfer_matrices()

        u = Function(self.V)
        u.vector()[:] = self.adjustment*rx
        LL, HH, SS = self.big_error_norms(u, self.insides)

        L2, H1, _, u1, u2, r1, r2 = self.schwarz_algorithm(num_of_iterations, mode_num)

        print('Initial L2 Error:', np.sqrt(L2[0]/LL), 'Initial H1 Error:', np.sqrt(H1[0]/HH))
        
        for i in range(1,len(L2)):
            print('Iteration', i, 'L2 Error:', np.sqrt(L2[i]/LL), 'H1 Error:', np.sqrt(H1[i]/HH))

        if (supress_results == False):
            file = File('paraview.pvd')
            file << u
            file << u1
            file << u2

            uu1 = Function(self.V1)
            uu1.vector()[:] = rx1
            fig, ax = plt.subplots()
            c = plot(uu1)
            fig.colorbar(c)
            plt.savefig('initial_left.png')
            plt.cla()
            plt.clf()
            plt.close()

            uu2 = Function(self.V2)
            uu2.vector()[:] = rx2
            fig, ax = plt.subplots()
            c = plot(uu2)
            fig.colorbar(c)
            plt.savefig('initial_right.png')
            plt.cla()
            plt.clf()
            plt.close()
        
            fig, ax = plt.subplots()
            ax.plot(L2/(LL*AO), label='L2 Error Norm (relative)')
            ax.plot(H1/(HH*AO), label='H1 Error Norm (relative)')
            plt.grid(b=True, ls='-.')
            ax.legend(loc='upper right')
            ax.set_ylabel('Relative Error Norms per Volume',fontsize=18)
            ax.set_xlabel('Iteration Steps', fontsize=18)
            plt.savefig('error_norms.png')
            plt.cla()
            plt.clf()
            plt.close()

            fig, ax = plt.subplots()
            c= plot(u1)
            fig.colorbar(c)
            ax.set_title('Mode of Vibration on Left Domain')
            plt.savefig('final_left.png')
            plt.cla()
            plt.clf()
            plt.close()

            fig, ax = plt.subplots()
            c= plot(u2)
            fig.colorbar(c)
            ax.set_title('Mode of Vibration on Right Domain')
            plt.savefig('final_right.png')
            plt.cla()
            plt.clf()
            plt.close()

            fig, ax = plt.subplots()
            b = plot(u1)
            c = plot(u2)
            fig.colorbar(c)
            ax.set_title('Juxtaposition of Left and Right Domains')
            plt.savefig('composition.png')
            plt.cla()
            plt.clf()
            plt.close()
        
            fig, ax = plt.subplots()
            ax.plot(np.sqrt(abs((r1))), label='Left Domain Frequency')
            ax.plot((np.sqrt(abs(r2))), label='Right Domain Frequency')
            plt.grid(b=True)
            ax.axhline(y=np.sqrt(r), label='Target Frequency', color='r')
            ax.legend(loc='lower right')
            ax.set_ylabel(r'Natural Frequency $\omega _n$')
            ax.set_xlabel('Iteration Steps')
            plt.savefig('eigenvalues.png')
            plt.close()

            fig, ax = plt.subplots()
            d = plot(u)
            fig.colorbar(d)
            ax.set_title('Reference Mode of Vibration')
            plt.savefig('target.png')
            plt.close()

        return L2/(LL*AO), H1/(H1*AO), L2, H1