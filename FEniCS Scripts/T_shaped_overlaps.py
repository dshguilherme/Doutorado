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

    h = 1.
    L = 1.
    c = 0.1*(i+1)
    r = h/4 

    P1 = Point(0., h)
    P2 = Point(L+c, h+h)
    left_rectangle = mshr.Rectangle(P1, P2)

    C1 = Point(L+c, h+h-r)
    C2 = Point(L+c, h+r)

    minus_circle = mshr.Circle(C1, r)
    plus_circle = mshr.Circle(C2, r)

    domain1 = left_rectangle #+plus_circle) #-minus_circle
    domain2 = mshr.Rectangle(Point(L, 0.), Point(L+L, h+h+h))
    domain = domain1+domain2

    size = 30
    membrane = Membrane(domain, domain1, domain2,
                        mesh_resolution=size, polynomial_degree=1, adjust=1)

    # Define SubDomains

    class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class LeftDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and between(x[0], [0, L])

    class LeftRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and between(x[0],[L, L+c])

    class RightDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            y_top = near(x[1], h+h+h)
            y_bot = near(x[1], 0.)
            y_cond = y_top or y_bot

            x_right = near(x[0], L+L)
            x_left = near(x[0], L) and not between(x[1], [h, h+h])
            x_cond = x_right or x_left

            cond = y_cond or x_cond

            return on_boundary and cond

    class RightRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and between(x[0], [h, h+h])

    class Overlap(SubDomain):
        def inside(self, x, on_boundary):
            if between(x[1], [h, h+h]):
                if between(x[1], [h, h+0.5*h]):
                    if between(x[0], [L, L+c]):
                        return True
                    else:
                        xx = (x[0] -C2.x())**2
                        yy = (x[1] -C2.y())**2
                        rr = r**2
                        return (xx + yy) <= rr
                else:
                    if between(x[0], [L, L+c-r]):
                        return True
                    else:
                        xx = (x[0] -C1.x())**2
                        yy = (x[1] -C1.y())**2
                        rr = r**2
                        circ_cond = (xx +yy >= rr)
                        return circ_cond and (x[0] < L+c +DOLFIN_EPS)

    insides = Overlap()
    outsides = OnBoundary()
    left_outsides = LeftDirichlet()
    left_robin = LeftRobin()
    right_outsides = RightDirichlet()
    right_robin = RightRobin()

    # Solution
    path = 'T_shaped/'+'areaadjusted/'+str(size)+'/'
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
                                    num_of_iterations=3,
                                    membrane=membrane, mode_num=0)

    generate_relatory(path, membrane, L2, H1, SH1, u,
                        u1, u2, r, r1, r2, vecs)
    overlap_area = c*h
    area_array[i] = overlap_area
    L2error[i] = L2[-1]
    aaL2error[i] = L2[-1]/overlap_area
    H1error[i] = H1[-1]
    aaH1error[i] = H1[-1]/overlap_area

total_area = 3*L*h
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