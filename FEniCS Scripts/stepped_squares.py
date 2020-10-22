from dolfin import(SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
import mshr
import matplotlib.pyplot as plt
import numpy as np

from Membrane import Membrane, standard_solver

L1 = 1.5
L2 = 0.75
L3 = 3
OL = L3 - L1 - L2
h1 = 2
h2 = L3 + 1
h3 = (h2-h1)/2

number_of_refinements = 1

domain1 = mshr.Rectangle(Point(0., 0.), Point(L1, h1))
domain2 = mshr.Rectangle(Point(L1-OL, -h3), Point(L3, h1+h3))
domain = domain1 + domain2

membrane = Membrane(domain, domain1, domain2, mesh_resolution=15,
                    polynomial_degree=1)

# Define SubDomains
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class LeftDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], [0., L1-OL])

class LeftRobin(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and between(x[0], [L1-OL, L1])

class RightDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        if near(x[0], L1-OL):
            return on_boundary and (not between(x[1], [0., h1])) 
        else:
            return on_boundary

class RightRobin(SubDomain):
    def inside(self, x, on_boundary):
        x_cond = near(x[0], L1-OL)
        y_cond = between(x[1], [0, h1])
        cond = x_cond and y_cond
        return on_boundary and cond

class Overlap(SubDomain):
    def inside(self, x, on_boundary):
        x_cond = between(x[0], [L1-OL, L1])
        y_cond = between(x[1], [0, h1])
        return x_cond and y_cond

insides = Overlap()
outsides = OnBoundary()
left_outsides = LeftDirichlet()
left_robin = LeftRobin()
right_outsides = RightDirichlet()
right_robin = RightRobin()

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
L2, H1, u1, u2 = membrane.schwarz_algorithm(4, u1, u2, left_robin, left_outsides,
                                            right_robin, right_outsides, M1, M2)

print('Initial L2 Relative Error:', np.sqrt(L2[0]/LL))
print('Initial H1 Relative Error:', np.sqrt(H1[0]/HH))
for i in range(1,len(L2)):
    print('Iteration', i, 'L2 error:', np.sqrt(L2[i]/LL))
    print('Iteration', i, 'H1 error:', np.sqrt(H1[i]/HH))