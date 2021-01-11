from dolfin import(SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
import mshr
import matplotlib.pyplot as plt
import numpy as np

from Membrane import Membrane, standard_solver, membrane_iterator, generate_relatory

L1 = 1.
h1 = 1.
L2 = 1.
h2 = 2.
b = 1

p1 = Point(0.,0.)
p2 = Point(L1, h1)
p3 = Point(L1+b, h1)
p4 = Point(L1, 0.)

domain1 = mshr.Rectangle(p1, p2) +mshr.Polygon([p4,p3,p2])

p5 = Point(L1, -(h2-h1))
p6 = Point(L1+L2, h1)

domain2 = mshr.Rectangle(p5, p6)
domain = domain1 + domain2
L2norms = list()
H1norms = list()
for h in range(10):
    size = 5*(h+1)
    membrane = Membrane(domain, domain1, domain2, mesh_resolution=size,
                        polynomial_degree=1)

    # Define SubDomains
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

    class LeftRobin(SubDomain):
        def inside(self, x, on_boundary):
            x_right = between(x[0], [L1, L1+b])
            y_right = between(x[1], [0, h1])
            cond = x_right and y_right
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
    outsides = OnBoundary()
    left_outsides = LeftDirichlet()
    left_robin = LeftRobin()
    right_outsides = RightDirichlet()
    right_robin = RightRobin()
    path = 'L_shaped/'+str(size)+'/'
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
                                    num_of_iterations=4,
                                    membrane=membrane, mode_num=0)

    generate_relatory(path, membrane, LL2, H1, SH1, u,
                        u1, u2, r, r1, r2, vecs)
    L2norms.append(LL2[-1])
    H1norms.append(H1[-1])

fig, ax = plt.subplots()
ax.plot(L2norms, label='L2 norm after 10 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('Relative Error', fontsize=18)
ax.set_xlabel('Element Size')
ax.set_xticks([0, 1 , 2, 3, 4])#, 5, 6, 7, 8, 9])
ax.set_xticklabels(["h", "h/2", "h/3", "h/4", "h/5"])#, "h/6", "h/7", "h/8", "h/9", "h/10"])
plt.savefig('L_shaped/L2norms.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(H1norms, label='H1 norm after 10 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('Relative Error', fontsize=18)
ax.set_xlabel('Element Size')
ax.set_xticks([0, 1 , 2, 3, 4])#, 5, 6, 7, 8, 9])
ax.set_xticklabels(["h", "h/2", "h/3", "h/4", "h/5"])#, "h/6", "h/7", "h/8", "h/9", "h/10"])
plt.savefig('L_shaped/H1norms.png')
plt.close()