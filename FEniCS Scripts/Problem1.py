import numpy as np
import mshr
from dolfin import (SubDomain, Point, DOLFIN_EPS, near, between)

from Membrane import (Membrane, standard_solver, membrane_iterator,
                         generate_relatory)
from reporter import Relatory

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


membranes = list()
d = mshr.Rectangle(Point(0., 0.), Point(L,h))
overlaps = np.linspace(0.1, 1, num=10, endpoint=True)
for o in overlaps:
    b = L*(1-o)/2
    
    class LeftRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], b+o*L)
    

    class RightRobin(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], b)

    class Overlap(SubDomain):
        def inside(self, x, on_boundary):
            return between(x[0], [b, b+o*L])
    
    iside = Overlap()
    oside = OnBoundary()
    loside = LeftDirichlet()
    lrobin = LeftRobin()
    roside = RightDirichlet()
    rrobin = RightRobin()

    boundaries = list([iside, oside, loside, lrobin, roside, rrobin])
    
    r1 = mshr.Rectangle(Point(0., 0.), Point(b+L*o, h))
    r2 = mshr.Rectangle(Point(b, 0.), Point(L,h))
    mem = Membrane(d, r1, r2, mesh_resolution=30, boundaries=boundaries,
                    polynomial_degree=1, adjust=.1)
    
    membranes.append(mem)

relatory = Relatory(overlaps=overlaps, membranes=membranes)
relatory.schwarz_iterations()
relatory.generate_overlap_error_graphs()
# relatory.generate_overlap_k_surface()