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

center = Point(0., 0.)
a = 131.32/2
b = 85.09/2
L = 20
h = 50.80
foot = 3.3

foot_x = center.x() - a - foot
foot_y = center.y() - 0.5*h 
head_x = foot_x + L
head_y = foot_y + h
frontier_x = -a*np.sqrt(1 - (h / (2*b) )**2 )
frontier_y = h/2
domain1 = mshr.Rectangle(Point(foot_x, foot_y), Point(frontier_x, frontier_y))
domain2 = mshr.Ellipse(center, a, b)
domain = domain1+domain2
left = domain1 - domain2
right = domain2 - domain1
domain3 = mshr.Rectangle(Point(foot_x,foot_y),Point(head_x+4*L, head_y))
domain = domain3+domain2
L2norms = list()
H1norms = list()
for i in range(10):
    size = 6*(i+1)
    membrane = Membrane(domain, domain3, right, 
                    mesh_resolution=size, polynomial_degree=1)

    # Define SubDomains



    class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class LeftDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            x_cond = x[0] >= frontier_x -DOLFIN_EPS
            return on_boundary and not x_cond
    class LeftRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] >= frontier_x - DOLFIN_EPS

    class RightDirichlet(SubDomain):
        def inside(self, x, on_boundary):
            x_cond = x[0] <= frontier_x +DOLFIN_EPS
            return on_boundary and not x_cond

    class RightRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] <= frontier_x +DOLFIN_EPS

    class Overlap(SubDomain):
        def inside(self, x, on_boundary):
            x_cond = between(x[0], [-a, frontier_x])
            yy = x[1]**2
            tmp = 1-(x[0]/a)**2
            dd = b*tmp*b
            y_cond = (yy <= dd)
            return x_cond and y_cond

    insides = Overlap()
    outsides = OnBoundary()
    left_outsides = LeftDirichlet()
    left_robin = LeftRobin()
    right_outsides = RightDirichlet()
    right_robin = RightRobin()

    # Solution
    path = 'fin/'+str(size)+'/'
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
                                    num_of_iterations=1,
                                    membrane=membrane, mode_num=0)

    generate_relatory(path, membrane, L2, H1, SH1, u,
                        u1, u2, r, r1, r2, vecs)
    L2norms.append(L2[-1]/(h*foot/2))
    H1norms.append(H1[-1]/(h*foot/2))
fig, ax = plt.subplots()
ax.plot(L2norms, label='L2 norm after 10 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('Relative Error', fontsize=18)
ax.set_xlabel('Element Size')
ax.set_xticks([0, 1 , 2, 3, 4])#, 5, 6, 7, 8, 9])
ax.set_xticklabels(["h", "h/2", "h/3", "h/4", "h/5"])#, "h/6", "h/7", "h/8", "h/9", "h/10"])
plt.savefig('fin/L2norms.png')
plt.close()

fig, ax = plt.subplots()
ax.plot(H1norms, label='H1 norm after 10 iterations')
ax.legend(loc='upper right')
ax.set_ylabel('Relative Error', fontsize=18)
ax.set_xlabel('Element Size')
ax.set_xticks([0, 1 , 2, 3, 4])#, 5, 6, 7, 8, 9])
ax.set_xticklabels(["h", "h/2", "h/3", "h/4", "h/5"])#, "h/6", "h/7", "h/8", "h/9", "h/10"])
plt.savefig('fin/H1norms.png')
plt.close()