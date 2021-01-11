from dolfin import(SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
import mshr
import matplotlib.pyplot as plt
import numpy as np

from Membrane import (Membrane, standard_solver, membrane_iterator,
                         generate_relatory)

h = 1.
L = 1.5
L2norms = list()
H1norms = list()
hsize = list()
domain = mshr.Rectangle(Point(0., 0.), Point(L, h))
for j in range(1):
    o = 0.1*(j+1)

    b = L*(1-o)/2 # Non Overlapping Side Size

    domain1 = mshr.Rectangle(Point(0., 0.), Point(b+L*o, h))
    domain2 = mshr.Rectangle(Point(b, 0.), Point(L, h))
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

    class LeftRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], b+o*L)

    class RightDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            x_right = near(x[0], L)
            y_top = x[1] > 1 - DOLFIN_EPS
            y_bot = x[1] < 0 + DOLFIN_EPS
            cond = x_right or (y_top or y_bot)        
            return on_boundary and cond

    class RightRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], b)

    class Overlap(SubDomain):
        def inside(self, x, on_boundary):
            return between(x[0], [b, b+o*L])

    # Define SubDomains
    ss = np.logspace(0,2,50,base=10)
    for h in range(50):
        size = ss[h]   
        membrane = Membrane(domain, domain1, domain2,
                        mesh_resolution=size, polynomial_degree=1,adjust=0.05)
    # Solution


        insides = Overlap()
        outsides = OnBoundary()
        left_outsides = LeftDirichlet()
        left_robin = LeftRobin()
        right_outsides = RightDirichlet()
        right_robin = RightRobin()
        path = 'rectangular/areadjusted/'+str(size)+'/'
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

        generate_relatory(path, membrane, L2, H1, SH1, u,
                            u1, u2, r, r1, r2, vecs)
        L2norms.append(L2[-1]/(1.5*o))
        H1norms.append(H1[-1]/(1.5*o))
        hsize.append(membrane.mesh.hmin())
        # areas.append(area)

np.save('rectangular/data/hrefinement/L2.npy', L2norms)
np.save('rectangular/data/hrefinement/H1.npy', H1norms)
np.save('rectangular/data/hrefinement/hsize.npy', hsize)

fig, ax = plt.subplots()
ax.plot(hsize,L2norms)
plt.grid(b=True)
ax.legend(loc='upper right')
ax.set_ylabel('L2 Error per Unit Area', fontsize=18)
ax.set_xlabel('minimum h size')
plt.savefig('rectangular/hrefinementL2.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(hsize,L2norms)
plt.grid(b=True)
ax.legend(loc='upper right')
ax.set_ylabel('L2 Error per Unit Area', fontsize=18)
ax.set_xlabel('minimum h size')
ax.set_xscale('log')
plt.savefig('rectangular/hrefinementL2log.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(hsize,L2norms)
plt.grid(b=True)
ax.legend(loc='upper right')
ax.set_ylabel('L2 Error per Unit Area', fontsize=18)
ax.set_xlabel('minimum h size')
ax.set_yscale('log')
ax.set_xscale('log')
plt.savefig('rectangular/hrefinementL2loglog.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(hsize,H1norms)
plt.grid(b=True)
ax.legend(loc='upper right')
ax.set_ylabel('H1 Error per Unit Area', fontsize=18)
ax.set_xlabel('minimum h size')
plt.savefig('rectangular/hrefinementH1.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(hsize,H1norms)
plt.grid(b=True)
ax.legend(loc='upper right')
ax.set_ylabel('H1 Error per Unit Area', fontsize=18)
ax.set_xlabel('minimum h size')
ax.set_xscale('log')
plt.savefig('rectangular/hrefinementH1log.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(hsize,H1norms)
plt.grid(b=True)
ax.legend(loc='upper right')
ax.set_ylabel('H1 Error per Unit Area', fontsize=18)
ax.set_xlabel('minimum h size')
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('rectangular/hrefinementH1loglog.png')
plt.close()