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



rows = 9
cols = 1
L2error = np.ndarray((rows,cols))
aaL2error = np.ndarray((rows,cols))
H1error = np.ndarray((rows,cols))
aaH1error = np.ndarray((rows,cols))
area_array = np.ndarray((rows,cols))
for i in range(rows):
    L1 = 1.
    h1 = 1.
    L2 = 1.
    h2 = 2.

    p1 = Point(0.,0.)
    p2 = Point(L1, h1)
    p4 = Point(L1, 0.)
    p5 = Point(L1, -(h2-h1))
    p6 = Point(L1+L2, h1)

    domain2 = mshr.Rectangle(p5, p6)
    b = 0.1*(i+1)
    p3 = Point(L1+b, h1)
    domain1 = mshr.Rectangle(p1, p2) +mshr.Polygon([p4,p3,p2])
    domain = domain1 + domain2

    membrane = Membrane(domain, domain1, domain2, mesh_resolution=30,
                        polynomial_degree=1, adjust=1)    

    class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class LeftDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            y_top = near(x[1], h1)
            x_left = near(x[0], 0.)
            y_bot = near(x[1], 0.)
            cond = (y_top or x_left) or (y_bot)
            return on_boundary and cond

    class RightDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            y_top = near(x[1], h1)
            y_bot = near(x[1], -(h2-h1))
            y_clamp = y_top or y_bot

            x_right = near(x[0], L1+L2)
            x_left = near(x[0], L1)
            y_left = x[1] < 0 +DOLFIN_EPS
            x_clamp = x_right or (x_left and y_left)

            return on_boundary and (y_clamp or x_clamp)

    class LeftRobin(SubDomain):
        def inside(self, x, on_boundary):
            x_right = between(x[0], [L1, L1+b])
            y_right = between(x[1], [0, h1])
            cond = x_right and y_right
            return on_boundary and cond

    class RightRobin(SubDomain):
        def inside(self, x, on_boundary):
            if near(x[1], h1):
                x_top = between(x[0], [L1, L1+b])
                return on_boundary and x_top
            elif near(x[0], L1):
                y_left = between(x[1], [0., h1])
                return on_boundary and y_left

    class Overlap(SubDomain):
        def inside(self, x, on_boundary):
            if between(x[1], [0, h1]):
                xx = (b/h1)*x[1] + L1
                x_cond = between(x[0], [L1, xx])
                return x_cond
            else:
                return False

    insides = Overlap()
    left_robin = LeftRobin()
    right_robin = RightRobin()
    outsides = OnBoundary()
    left_outsides = LeftDirichlet()
    right_outsides = RightDirichlet()
    path = 'L_shaped/areaadjusted/'+str(30)+'/'
    freqs, vecs = membrane.initial_solution(outsides, left_outsides,
                                right_outsides, mode_number=0)
    u = Function(membrane.V)
    u.vector()[:] = vecs[0]
    r = freqs[0]
    LL2, H1, SH1, u1, u2, r1, r2 = membrane_iterator(insides=insides, 
                                    outsides=outsides,
                                    left_outsides=left_outsides,
                                    left_robin=left_robin,
                                    right_outsides=right_outsides,
                                    right_robin=right_robin,
                                    num_of_iterations=3,
                                    membrane=membrane, mode_num=0)

    generate_relatory(path, membrane, LL2, H1, SH1, u,
                        u1, u2, r, r1, r2, vecs)
    overlap_area = b/2
    area_array[i] = b/2
    L2error[i] = LL2[-1]
    aaL2error[i] = LL2[-1]/(overlap_area)
    H1error[i] = H1[-1]
    aaH1error[i] = H1[-1]/(overlap_area)

total_area = 2*L2 + L1
area_array = area_array/total_area

print(area_array)

fig, ax = plt.subplots()
ax.plot(area_array,aaL2error, label='L2 norm after 3 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('L2 Error Norm per Area', fontsize=18)
ax.set_xlabel('Overlapping Area Percentage', fontsize=18)
plt.grid(b=True)
plt.savefig(path+'aaL2.png')
plt.close()


fig, ax = plt.subplots()
ax.plot(area_array,aaH1error, label='H1 norm after 3 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('H1 Error Norm per Area', fontsize=18)
ax.set_xlabel('Overlapping Area Percentage', fontsize=18)
plt.grid(b=True)
plt.savefig(path+'aaH1.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(area_array,L2error, label='L2 norm after 3 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('Total L2 Error Norm', fontsize=18)
ax.set_xlabel('Overlapping Area Percentage', fontsize=18)
plt.grid(b=True)
plt.savefig(path+'L2.png')
plt.close()


fig, ax = plt.subplots()
ax.plot(area_array,H1error, label='H1 norm after 3 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('Total H1 Error Norm', fontsize=18)
ax.set_xlabel('Overlapping Area Percentage', fontsize=18)
plt.grid(b=True)
plt.savefig(path+'H1.png')
plt.close()