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
b = 0.3

p1 = Point(0.,0.)
p2 = Point(L1, h1)
p3 = Point(L1+b, h1)
p4 = Point(L1, 0.)

domain1 = mshr.Rectangle(p1, p2) +mshr.Polygon([p4,p3,p2])

p5 = Point(L1, -(h2-h1))
p6 = Point(L1+L2, h1)

domain2 = mshr.Rectangle(p5, p6)

domain = domain1 + domain2

membrane = Membrane(domain, domain1, domain2, mesh_resolution=40,
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
        y_left = between(x[1], [0, -(h2-h1)])
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

generate_relatory('L_shaped/', membrane, L2, H1, SH1, u,
                     u1, u2, r, r1, r2)