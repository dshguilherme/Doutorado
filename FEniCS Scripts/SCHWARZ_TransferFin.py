from dolfin import (SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate, FacetNormal)
from StandardMembrane import StandardFin, standard_solver
import mshr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Geometry Definition
number_of_refinements = 1
max_iterations = 10

my_fin = StandardFin(center=Point(0., 0.), a=131.32/2, b=85.09/2, foot=3.3,
                      L=20, h=50.80, mesh_resolution=5, polynomial_order=2)

retangulo = my_fin.rec
elipse = my_fin.elli

malha = getattr(my_fin, 'mesh')
overlap_marker = MeshFunction('size_t', malha, malha.topology().dim())
my_fin.overlap_region().mark(overlap_marker, 1)

# Locally Refine the Meshes on the Overlap Region
for i in range(number_of_refinements):
    my_fin.refine_meshes(my_fin.overlap_region())

# Matrices
K, M, V = my_fin.assemble_matrices(my_fin.mesh)
bc = DirichletBC(V, Constant(0.), my_fin.clamp())
bc.apply(K)

K1, M1, V1 = my_fin.assemble_matrices(my_fin.rectangle_mesh)
bc = DirichletBC(V1, Constant(0.), my_fin.clamp())
bc.apply(K1)

K2, M2, V2 = my_fin.assemble_matrices(my_fin.ellipse_mesh)


# Initial Solutions
solver = standard_solver(K,M)
solver.solve(0)
r, c, rx, cx = solver.get_eigenpair(0)
print("Target Frequency:", sqrt(r))
u = Function(V)
if (np.absolute(rx.max()) < np.absolute(rx.min())):
    u.vector()[:] = -rx
else:
    u.vector()[:] = rx
plot(u)
plt.savefig('solution.png')

solver1 = standard_solver(K1,M1)
solver1.solve(0)
r1, c1, rx1, cx1 = solver1.get_eigenpair(0)
print("Initial Rectangle Frequency:", sqrt(r1))

u1 = Function(V1)
if (np.absolute(rx1.max()) < np.absolute(rx1.min())):
    u1.vector()[:] = -rx1
else:
    u1.vector()[:] = rx1

solver2 = standard_solver(K2,M2)
solver2.solve(0)
r2, c2, rx2, cx2 = solver2.get_eigenpair(0)
print("Initial Ellipse Frequency:", sqrt(r2))
u2 = Function(V2)
if (np.absolute(rx2.max()) < np.absolute(rx2.min())):
    u2.vector()[:] = -rx2
else:
    u2.vector()[:] = rx2

# Transfer Matrices and Overlap Solution
overlapping_mesh = my_fin.overlap_mesh
overlapping_mesh = refine(overlapping_mesh)

V12 = FunctionSpace(overlapping_mesh, "Lagrange",1)
B1 = PETScDMCollection.create_transfer_matrix(V1, V12)
BB1 = PETScDMCollection.create_transfer_matrix(V12, V1)
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
L2_error = np.zeros((max_iterations+1, 1))
L2_error[iter] = L2/LL

H1_error = np.zeros((max_iterations+1,1))
H1_error[iter] = H1/HH
while (iter < max_iterations):
    # First Step: do f1 = du2/dn2, g1 = u2
    uu2 = Function(V2)
    if (np.absolute(rx2.max()) < np.absolute(rx2.min())):
        uu2.vector()[:] = -rx2
    else:
        uu2.vector()[:] = rx2

    uu21 = Function(V12)
    uu21.vector()[:] = B2*uu2.vector()

    g1 = Function(V1)
    g1.vector()[:] = BB1*uu21.vector()

    du21 = Function(V12)
    # Step 1.75: Project du2/dn2 -> u21 -> du1/dn1
    du2 = Function(V2)
    n2 = FacetNormal(my_fin.ellipse_mesh)
    du2 = assemble(inner(grad(u2),n2)*ds)
    du2 = project(du2, V2)
    du21.vector()[:] = B2*du2.vector()

    f1 = Function(V1)
    f1.vector()[:] = BB1*du21.vector()   
    # Step 2 solver for u1
    boundary = my_fin.rec_robin()
    malha = getattr(my_fin, 'rectangle_mesh')
    plt.figure()
    plot(f1)
    plot(malha)
    plt.savefig('malhaf1.png')
    plt.close()
    K1, _ = my_fin.assemble_lui_stiffness(g=g1, f=f1, mesh=malha, robin_boundary=boundary)
    bc1 = DirichletBC(V1, Constant(0.), my_fin.clamp())
    bc1.apply(K1)

    solver1 = standard_solver(K1, M1)
    solver1.solve(0)
    r1, c1, rx1, cx1 = solver1.get_eigenpair(0)
    
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

    du1 = Function(V1)
    n1 = FacetNormal(my_fin.rectangle_mesh)
    du1 = assemble(inner(grad(u1),n1)*ds)
    du1 = project(du1, V1)

    du12 = Function(V12)
    du12.vector()[:] = B1*du1.vector()

    f2 = Function(V2)
    f2.vector()[:] = BB2*du12.vector()

    # Step 4 solve for u2
    boundary = my_fin.elli_robin()
    K2, _ = my_fin.assemble_lui_stiffness(g2, f2, my_fin.ellipse_mesh, boundary)
    solver2 = standard_solver(K2,M2)
    solver2.solve(0)
    r2, c2, rx2, cx2 = solver2.get_eigenpair(0)

    #Step 5 Calculate the L2 norm for convergence
    uu21.vector()[:] = B2*uu2.vector()
    uu12.vector()[:] = B1*uu1.vector()
    error = ((uu12-uu21)**2)*dx
    L2 = sqrt(abs(assemble(error)))
    print('Iteration ' +str(iter) +':')
    print('Relative L2 error is:', L2/LL)
    
    semi_error = inner(grad(uu12-uu21),grad(uu12-uu21))*dx
    H1 = L2 +sqrt(abs(assemble(semi_error)))
    print('Relative H1 error is:', H1/HH)
    print('Rectangle Frequency:', sqrt(r1))
    print('Ellipse Frequency:', sqrt(r2))
    iter += 1
    L2_error[iter] = L2/LL
    H1_error[iter] = H1/HH
