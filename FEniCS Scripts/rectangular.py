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
o = 0.1

b = L*(1-o)/2 # Non Overlapping Side Size

domain = mshr.Rectangle(Point(0., 0.), Point(L, h))
domain1 = mshr.Rectangle(Point(0., 0.), Point(b+L*o, h))
domain2 = mshr.Rectangle(Point(b, 0.), Point(L, h))

membrane = Membrane(domain, domain1, domain2,
                     mesh_resolution=30, polynomial_degree=1)

# Define SubDomains

class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class LeftDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        x_left = near(x[0], 0.)
        y_top = near(x[1], h)
        y_bot = near(x[1], 0.)
        cond = (x_left or y_top) or y_bot
        return on_boundary and cond

class LeftRobin(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], b+o*L)

class RightDirichlet(SubDomain):
    def inside(self, x, on_boundary):
        x_right = near(x[0], L)
        y_top = near(x[1], h)
        y_bot = near(x[1], 0.)
        cond = (x_right or y_top) or y_bot        
        return on_boundary and cond

class RightRobin(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], b)

class Overlap(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], [b, b+o*L])

insides = Overlap()
outsides = OnBoundary()
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

generate_relatory('rectangular/', membrane, L2, H1, SH1, u,
                     u1, u2, r, r1, r2)