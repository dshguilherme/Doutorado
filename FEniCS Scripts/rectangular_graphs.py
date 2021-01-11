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

aaL2 = list()
L2norms = list()
aaH1 = list()
H1norms = list()
areas = list()
for overlap in range(1):
    o = 0.1*(overlap+1)
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
    
    membrane = Membrane(domain, domain1, domain2, mesh_resolution=12, 
                         polynomial_degree=1, adjust=.1)
    insides = Overlap()
    outsides = OnBoundary()
    left_outsides = LeftDirichlet()
    left_robin = LeftRobin()
    right_outsides = RightDirichlet()
    right_robin = RightRobin()
    path = 'rectangular/36/' #+'/mode_'+str(j)+'/'
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
    oL2, oH1, _ = membrane.big_error_norms(u,insides)
    generate_relatory(path, membrane, L2/oL2, H1/oH1, SH1, u,
                        u1, u2, r, r1, r2, vecs)
    dx = Measure('dx',domain=membrane.overlap_mesh)
    a = Constant(1.0)*dx
    area = assemble(a)
    areas.append(area/1.5)
    aaL2.append(L2[-1]/area)
    aaH1.append(H1[-1]/area)
    L2norms.append(L2[-1])
    H1norms.append(H1[-1])

# fig, ax = plt.subplots()
# ax.plot(areas,L2norms)
# ax.legend(loc='upper right')
# ax.set_ylabel('Relative Error', fontsize=18)
# ax.set_xlabel('Overlapping Area')
# plt.savefig('test/05adjustL2full.png')
# plt.close()

# fig, ax = plt.subplots()
# ax.plot(areas,aaL2)
# ax.legend(loc='upper right')
# ax.set_ylabel('Relative Error per Area', fontsize=18)
# ax.set_xlabel('Overlapping Area')
# plt.savefig('test/05adjustaaL2.png')
# plt.close()

# fig, ax = plt.subplots()
# ax.plot(areas,H1norms)
# ax.legend(loc='upper right')
# ax.set_ylabel('Relative Error', fontsize=18)
# ax.set_xlabel('Overlapping Area')
# plt.savefig('test/05adjustH1full.png')
# plt.close()

# fig, ax = plt.subplots()
# ax.plot(areas,aaH1)
# ax.legend(loc='upper right')
# ax.set_ylabel('Relative Error per Area', fontsize=18)
# ax.set_xlabel('Overlapping Area')
# plt.savefig('test/05adjustaaH1.png')
# plt.close()