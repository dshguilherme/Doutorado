from dolfin import(SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
import mshr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Membrane import (Membrane, standard_solver, membrane_iterator,
                         generate_relatory)

h = 1.
L = 1.5

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class LeftDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        x_left = near(x[0], 0.)
        y_top = x[1] > 1 - DOLFIN_EPS
        y_bot = x[1] < 0 +DOLFIN_EPS
        cond = x_left or (y_top or y_bot)
        return on_boundary and cond

class RightDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        x_right = near(x[0], L)
        y_top = x[1] > 1 - DOLFIN_EPS
        y_bot = x[1] < 0 + DOLFIN_EPS
        cond = x_right or (y_top or y_bot)        
        return on_boundary and cond
rows = 1
cols = 100
L2error = np.ndarray((rows,cols))
aaL2error = np.ndarray((rows,cols))
H1error = np.ndarray((rows,cols))
aaH1error = np.ndarray((rows,cols))

add = np.logspace(-4,0.6937,100,base=10)
oo = np.logspace(-2,0,50,base=10)
for i in range(rows):
    o = 0.1
    b = L*(1-o)/2 # Non Overlapping Side Size
    domain = mshr.Rectangle(Point(0., 0.), Point(L, h))
    domain1 = mshr.Rectangle(Point(0., 0.), Point(b+L*o, h))
    domain2 = mshr.Rectangle(Point(b, 0.), Point(L, h))

    class LeftRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], b+o*L)
    class RightRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], b)

    class Overlap(SubDomain):
        def inside(self, x, on_boundary):
            return between(x[0], [b, b+o*L])


# Define SubDomains
    insides = Overlap()
    outsides = OnBoundary()
    left_outsides = LeftDirichlet()
    left_robin = LeftRobin()
    right_outsides = RightDirichlet()
    right_robin = RightRobin()
    
    for j in range(cols):
        ad = add[i] #0.03*(j+1)
        membrane = Membrane(domain, domain1, domain2, mesh_resolution=36,
                             polynomial_degree=1, adjust=ad)
        path = 'rectangular/surfaces/'+str(o) +'/ad/' +str(ad)
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
                                        num_of_iterations=2,
                                        membrane=membrane, mode_num=0)
        L2error[i,j] = L2[-1]
        aaL2error[i,j] = L2[-1]/(1.5*o)
        H1error[i,j] = H1[-1]
        aaH1error[i,j] = H1[-1]/(1.5*o)
        generate_relatory(path, membrane, L2,
                             H1, SH1, u, u1, u2, r, r1, r2, vecs)

np.save('rectangular/surfaces/small/gradL2.npy', L2error)
np.save('rectangular/surfaces/small/gradaaL2.npy', aaL2error)
np.save('rectangular/surfaces/small/gradH1.npy', H1error)
np.save('rectangular/surfaces/small/gradaaH1.npy',aaH1error)

# fig, ax = plt.subplots()
# ax.plot(add.T,aaL2error, label='L2 norm after 2 iterations')
# ax.legend(loc='upper right')
# ax.set_ylabel('L2 Error Norm per Area', fontsize=18)
# ax.set_xlabel('k', fontsize=18)
# plt.grid(b=True)
# plt.savefig('rectangular/areadjusted/aaL2.png')
# plt.close()


# fig, ax = plt.subplots()
# ax.plot(add.T,aaH1error, label='H1 norm after 2 iterations')
# ax.legend(loc='upper right')
# ax.set_ylabel('H1 Error Norm per Area', fontsize=18)
# ax.set_xlabel('k', fontsize=18)
# plt.grid(b=True)
# plt.savefig('rectangular/areadjusted/aaH1.png')
# plt.close()

# fig, ax = plt.subplots()
# ax.plot(np.arange(0.05,1,0.05),L2error, label='L2 norm after 2 iterations')
# ax.legend(loc='upper right')
# ax.set_ylabel('Total L2 Error Norm', fontsize=18)
# ax.set_xlabel('Overlapping Area Percentage', fontsize=18)
# plt.grid(b=True)
# plt.savefig('rectangular/areadjusted/L2.png')
# plt.close()


# fig, ax = plt.subplots()
# ax.plot(np.arange(0.05,1,0.05),H1error, label='H1 norm after 2 iterations')
# ax.legend(loc='upper right')
# ax.set_ylabel('Total H1 Error Norm', fontsize=18)
# ax.set_xlabel('Overlapping Area Percentage', fontsize=18)
# plt.grid(b=True)
# plt.savefig('rectangular/areadjusted/H1.png')
# plt.close()