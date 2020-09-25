'''
The strategy of this approach will be:
1. Compute the initial solution to the problem using T1, OT12, T2 for Omega1 and Omega2
2. Calculate the \
'''


from numpy import isclose
from dolfin import *
from multiphenics import *



# MESHES #

# Geometrical Data
L = 1.5 # Side of the Rectangle
h = 1.0 # Height of the Rectangle
o = 0.2 # Total overlapping Area
nx = 40 # Number of Nodes in x direction
ny = 40 # Number of Nodes in y direction
b = L*(1-o)/2 # Non Overlapping Side of the Rectangle

# Meshes
domain = Rectangle(Point(0.0, 0.0), Point(L,h))
domain_left = domain and Rectangle(Point(0.0,0.0), Point(b, h))
domain_center = domain and Rectangle(Point(b,0.0), Point(b+L*o,h))
domain_right = domain and Rectangle(Point(b+L*o, 0.0), Point(L,h))
domain.set_subdomain(1, domain_left)
domain.set_subdomain(2, domain_right)
domain.set_subdomain(3, domain_center)
mesh = generate_mesh(domain, 15)
# SubDomains

subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(). mesh.domains())

# Boundaries
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class OnInterfaceLeft(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], b)
        
class OnInterfaceRight(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], b+L*o)

boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
on_boundary = OnBoundary()
on_boundary.mark(boundaries,1)
on_interface = OnInterface()
on_interface.mark(boundaries,2)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= b + L*o
 
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= b

# Restrictions
boundary_restriction = MeshRestriction(mesh, on_boundary)
interace_restriction = MeshRestriction(mesh, on_interface)
left = Left()
left_restriction = MeshRestriction(mesh, left)
right = Right()
right_restriction = MeshRestriction(mesh, right)

# FUNCTION SPACES #
V = FunctionSpace(mesh, "Lagrange",1)
# Block function space
W = BlockFunctionSpace([V, V, V], restrict =[left, right, interface])

# TRIAL/TEST FUNCTIONS #
u1u2l = BlockTrialFunction(W)
(u1, u2, u3) = block_split(u1u2l)
v1v2m = BlockTestFunction(W)
(v1,v2,v3) = block_split(v1v2m)

# MEASURES #
dx = Measure('dx')(subdomain_data=subdomains)
ds = Measure('ds')(subdomain_data=boundaries)

# VARIATIONAL FORMS #

k = [[inner(grad(u1),grad(v1))*dx(1), 0                            , 0                              ],
     [0                             , inner(grad(u2),grad(v2))*dx(2), 0                             ],
     [0                             , 0                            , inner(grad(u3),grad(v3))*dx(3)]]
m = [[u1*v1*dx(1)   , 0          , 0          ],
     [0             , u2*v2*dx(2), 0          ],
     [0             , 0          , u3*v3*dx(3)]]
bc1 = DirichletBC(W.sub(0), Constant(0.0), boundaries, 1)
bc2 = DirichletBC(W.sub(1), Constant(0.0), boundaries, 1)
bc3 = DirichletBC(W.sub(2), Constant(0.0), boundaries, 1)
bcs = BlockDirichletBC([bc1, bc2, bc3])


# ASSEMBLY #

K = block_assemble(k)
M = block_assemble(m)
bcs.apply(K)
print("All done! Matrices are built")


