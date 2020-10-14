from dolfin import (SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
from StandardMembrane import RectangularMembrane, standard_solver
import mshr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Geometry Definition
def schwarz_transfer_membrane(L, h, o, mesh_resolution,
                               number_of_refinements, max_iterations,
                               polynomial_order):
    b = L*(1-o)/2 # Non Overlapping Side of the Rectangle
    P1 = Point(0., 0.)
    P2 = Point(L, h)
    P3 = Point(b+L*o, h)
    P4 = Point(b, 0.)

    domain = RectangularMembrane(P1,P2, mesh_resolution, polynomial_order)
    domain1 = RectangularMembrane(P1,P3, mesh_resolution, polynomial_order)
    domain2 = RectangularMembrane(P4,P2, mesh_resolution, polynomial_order)
    domain12 = RectangularMembrane(P4,P3, mesh_resolution, polynomial_order)

    malha = getattr(domain, 'mesh')
    overlap_marker = MeshFunction('size_t', malha, malha.topology().dim())
    domain12.insides().mark(overlap_marker, 1)

    # Locally Refine the Meshes on the Overlap Region
    for i in range(number_of_refinements):
        domain1.refine_mesh(domain12.insides())
        domain2.refine_mesh(domain12.insides())
    # Matrices
    K, M, V = domain.assemble_matrices()
    bc = DirichletBC(V, Constant(0.), domain.boundary())
    bc.apply(K)

    K1, M1, V1 = domain1.assemble_matrices()
    boundary_one = [domain1.left(), domain2.bottom(), domain2.top()]
    for subdomain in boundary_one:
        bc1 = DirichletBC(V1, Constant(0.), subdomain)
        bc1.apply(K1)

    K2, M2, V2 = domain2.assemble_matrices()
    boundary_two = [domain2.bottom(), domain2.right(), domain2.top()]
    for subdomain in boundary_two:
        bc2 = DirichletBC(V2, Constant(0.), subdomain)
        bc2.apply(K2)

    # Initial Solutions
    solver = standard_solver(K,M)
    solver.solve(0)
    r, c, rx, cx =  solver.get_eigenpair(0)
    print("Target Frequency:", sqrt(r))
    u = Function(V)
    if (np.absolute(rx.max()) < np.absolute(rx.min())):
        u.vector()[:] = -rx
    else:
        u.vector()[:] = rx

    solver1 = standard_solver(K1,M1)
    solver1.solve(0)
    r1, c1, rx1, cx1 =  solver1.get_eigenpair(0)
    print("Initial Omega 1 Frequency:", sqrt(r1))
    u1 = Function(V1)
    if (np.absolute(rx1.max()) < np.absolute(rx1.min())):
        u1.vector()[:] = -rx1
    else:
        u1.vector()[:] = rx1

    solver2 = standard_solver(K2,M2)
    solver2.solve(0)
    r2, c2, rx2, cx2 =  solver2.get_eigenpair(0)
    print("Initial Omega2 Frequency:", sqrt(r2))
    u2 = Function(V2)
    if (np.absolute(rx2.max()) < np.absolute(rx2.min())):
        u2.vector()[:] = -rx2
    else:
        u2.vector()[:] = rx2

    # Transfer Matrices and Overlap solution
    overlapping_mesh = getattr(domain12, 'mesh')
    overlapping_mesh = refine(overlapping_mesh)

    V12 = FunctionSpace(overlapping_mesh, "Lagrange", 1)

    B1 = PETScDMCollection.create_transfer_matrix(V1,V12)
    BB1 = PETScDMCollection.create_transfer_matrix(V12,V1)
    B2 = PETScDMCollection.create_transfer_matrix(V2, V12)
    BB2 = PETScDMCollection.create_transfer_matrix(V12,V2)

    B = PETScDMCollection.create_transfer_matrix(V12,V)

    u12 = Function(V12)
    u12.vector()[:] = B1*u1.vector()

    u21 = Function(V12)
    u21.vector()[:] = B2*u2.vector()

    error = ((u12-u21)**2)*dx
    L2 = sqrt(abs(assemble(error)))

    semi_error = inner(grad(u12-u21),grad(u12-u21))*dx
    H1 = L2 +sqrt(abs(assemble(semi_error)))

    uu = Function(V)
    uu.vector()[:] = u.vector()

    dxo = Measure('dx', subdomain_data=overlap_marker)
    dividendo = (uu**2)*dxo
    semi_dividendo = (inner(grad(uu),grad(uu)))*dxo

    LL = sqrt(abs(assemble(dividendo)))
    HH = LL + sqrt(abs(assemble(semi_dividendo)))
    print('Initial Relative L2 error:', L2/LL)
    print('Initial Relative H1 error:', H1/HH)

    # Schwarz Algorithm
    iter = 0
    L2_error = np.zeros((max_iterations+1,1))
    L2_error[iter] = L2/LL

    H1_error = np.zeros((max_iterations+1,1))
    H1_error[iter] = H1/HH
    while (iter < max_iterations):
        # First Step: do f1 = du2/dn2, g1 = u2
        # Step 1.5: Project u2 -> u21 -> u1
        uu2 = Function(V2)
        if (np.absolute(rx2.max()) < np.absolute(rx2.min())):
            uu2.vector()[:] = -rx2
        else:
            uu2.vector()[:] = rx2

        uu21 = Function(V12)
        uu21.vector()[:] = B2*uu2.vector()

        g1 = Function(V1)
        g1.vector()[:] = BB1*uu21.vector()

        plt.figure()
        c = plot(g1)
        plt.colorbar(c)
        plt.axis([b, b+L*o, 0, 1])
        plt.title('g1 iteration step ' +str(iter))
        plt.savefig('g1 step' +str(iter)+'.png')
        plt.close()

        # Step 1.75: Project du2/dn2 -> u21 -> du1/dn1
        du2 = Function(V2)
        du2 = project(-uu2.dx(0), V2)

        du21 = Function(V12)
        du21.vector()[:] = B2*du2.vector()

        f1 = Function(V1)
        f1.vector()[:] = BB1*du21.vector()

        plt.figure()
        c = plot(f1)
        plt.colorbar(c)
        plt.axis([b, b+L*o, 0, 1])
        plt.title('f1 iteration step ' +str(iter))
        plt.savefig('f1 step' +str(iter)+'.png')
        plt.close()


        # Step 2 solve for u1
        boundary = domain1.right()
        K1, _ = domain1.assemble_lui_stiffness(g1, f1, boundary)
        for subdomain in boundary_one:
            bc1 = DirichletBC(V1, Constant(0.), subdomain)
            bc1.apply(K1)

        solver1 = standard_solver(K1,M1)
        solver1.solve(0)
        r1, c1, rx1, cx1 = solver1.get_eigenpair(0)
    #  print("Omega 1 Frequency after step:", sqrt(r1))

        # Step 3 do f2 = du1/dn1, g2 = u1
        uu1 = Function(V1)
        if (np.absolute(rx1.max()) < np.absolute(rx1.min())):
            uu1.vector()[:] = -rx1
        else:
            uu1.vector()[:] = rx1

        uu12 = Function(V12)
        uu12.vector()[:] = B1*uu1.vector()

        g2 = Function(V2)
        g2.vector()[:] = BB2*uu12.vector()

        plt.figure()
        c = plot(g2)
        plt.colorbar(c)
        plt.axis([b, b+L*o, 0, 1])
        plt.title('g2 iteration step ' +str(iter))
        plt.savefig('g2 step' +str(iter)+'.png')
        plt.close()


        du1 = Function(V1)
        du1 = project(-uu1.dx(0), V1)

        du12 = Function(V12)
        du12.vector()[:] = B1*du1.vector()

        f2 = Function(V2)
        f2.vector()[:] = BB2*du12.vector()

        plt.figure()
        c = plot(f2)
        plt.colorbar(c)
        plt.axis([b, b+L*o, 0, 1])
        plt.title('f2 iteration step ' +str(iter))
        plt.savefig('f2 step' +str(iter)+'.png')
        plt.close()


        # Step 4 solve for u2
        boundary = domain2.left()
        K2, _ = domain2.assemble_lui_stiffness(g2, f2, boundary)
        for subdomain in boundary_two:
            bc2 = DirichletBC(V2, Constant(0.), subdomain)
            bc2.apply(K2)
        solver2 = standard_solver(K2,M2)
        solver2.solve(0)
        r2, c2, rx2, cx2 = solver2.get_eigenpair(0)
    # print("Omega 2 Frequency after step:", sqrt(r2))

        # Step 5 Calculate the L2 norm for convergence
        error = ((uu12-uu21)**2)*dx
        L2 = sqrt(abs(assemble(error)))
        print('Iteration ' +str(iter) +':')
        print('Relative L2 error is:', L2/LL)
        
        semi_error = inner(grad(uu12-uu21),grad(uu12-uu21))*dx
        H1 = L2 +sqrt(abs(assemble(semi_error)))
        print('Relative H1 error is:', H1/HH)
        iter += 1
        L2_error[iter] = L2/LL
        H1_error[iter] = H1/HH
        prev = H1/HH
    return L2_error, H1_error 

L2_error, H1_error = schwarz_transfer_membrane(L=1.5, h=1, o=0.1,
                                                mesh_resolution=10, 
                                                number_of_refinements=1,
                                                max_iterations=10,
                                                polynomial_order=3)

fig, ax = plt.subplots()
plt.title('Error Norms',fontsize='x-large')
ax.plot(L2_error, label='Relative L2 Error Norm')
ax.plot(H1_error, 'r', label='Relative H1 Error Norm')
plt.ylabel('Relative Error', fontsize='x-large')
plt.xlabel('Iteration', fontsize='x-large')
legend = ax.legend(loc='upper center', shadow=True, fontsize='large')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.show()
plt.savefig('L2H1_error.png')
plt.close()
