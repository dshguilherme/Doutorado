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

domain1 = mshr.Rectangle(Point(foot_x, foot_y), Point(head_x, head_y))
domain2 = mshr.Ellipse(center, a, b)

domain = domain1+domain2

membrane = Membrane(domain, domain1, domain2, 
                mesh_resolution=25, polynomial_degree=1)

# Define SubDomains

frontier_x = -a*np.sqrt(1 - (h / (2*b) )**2 )
frontier_y = h/2


class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class LeftDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], foot_x)

class LeftRobin(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] >= frontier_x - DOLFIN_EPS

class RightDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return False

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
outsides = LeftDirichlet()
left_outsides = LeftDirichlet()
left_robin = LeftRobin()
right_outsides = RightDirichlet()
right_robin = RightRobin()

# Solution

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
                                num_of_iterations=15,
                                membrane=membrane, mode_num=0)

generate_relatory('fin/', membrane, L2, H1, SH1, u,
                     u1, u2, r, r1, r2)