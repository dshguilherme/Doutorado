from fenics import *
from dolfin import *
from petsc4py import PETSc

# Implementation of the Overlapping Membrane using SubDomains
L = 1.5 # Side of the Rectangle
h = 1.0 # Height of the Rectangle
o = 0.7 # Total overlapping Area

nx = 24 # Number of Nodes in x direction
ny = 24 # Number of Nodes in y direction

b = L*(1-o)/2 # Non Overlapping Side of the Rectangle


# Let's start defining the subdomains
class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= L*o + b + DOLFIN_EPS

class RobinBoundary_1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] >= L*o +b -DOLFIN_EPS

class DirichletBoundary_1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < L*o +b -DOLFIN_EPS

class Omega_2(SubDomain):
    def inside(self,x,on_boundary):
        return x[0] >= b + DOLFIN_EPS

class RobinBoundary_2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] <= b + DOLFIN_EPS

class DirichletBoundary_2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < b + DOLFIN_EPS

class ClampedEdges(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
        
# Meshing
omega = RectangleMesh(Point(0.0,0.0), Point(L,h), nx, ny, "crossed")

# Domain and SubDomain Markers
domains = MeshFunction('size_t', omega, omega.topology().dim(),0)
domains.set_all(0)

omega1sub = Omega_1()
omega1sub.mark(domains,1)

omega2sub = Omega_2()
omega2sub.mark(domains,2)


boundaries = MeshFunction('size_t', omega, omega.topology().dim()-1,0)
boundaries.set_all(0)

clamp = ClampedEdges()
clamp.mark(boundaries,0)

luiboundary1 = RobinBoundary_1()
luiboundary1.mark(boundaries,1)

luiboundary2 = RobinBoundary_2()
luiboundary2.mark(boundaries,2)

# Trial / Test Functions
V = FunctionSpace(omega,"Lagrange",1)
u = TrialFunction(V)
v = TestFunction(V)

# Define new measures for the Subdomains and Boundaries
dx = Measure('dx', subdomain_data = domains)
ds = Measure('ds', subdomain_data = boundaries)

# Initial Bilinear Forms

k = inner(grad(u),grad(v))*dx
k1 = inner(grad(u),grad(v))*dx(1)
k2 = inner(grad(u),grad(v))*dx(2)

m = u*v*dx(0)
m1 = u*v*dx(1)
m2 = u*v*dx(2)

# Stiffness Matrices
K = PETScMatrix()
K1 = PETScMatrix()
K2 = PETScMatrix()

assemble(k, tensor=K)
assemble(k1, tensor=K1)
assemble(k2, tensor=K2)

# Apply Dirichlet Boundary Conditions at the Clamp
bcs = DirichletBC(V, 0.0, boundaries,0)
bcs.apply(K)
#bcs.apply(K1)
#bcs.apply(K2)

# Mass Matrices

M = PETScMatrix()
M1 = PETScMatrix()
M2 = PETScMatrix()

assemble(m, tensor=M)
assemble(m1, tensor=M1)
assemble(m2, tensor=M2)

# Eigensolver
solver0 = SLEPcEigenSolver(K,M)
solver0.parameters['spectrum'] = "smallest magnitude"
solver0.parameters['tolerance'] = 1e-6
solver0.parameters['problem_type'] = "pos_gen_non_hermitian"

solver1 = SLEPcEigenSolver(K1,M1)
solver1.parameters['spectrum'] = "smallest magnitude"
solver1.parameters['tolerance'] = 1e-6
solver1.parameters['problem_type'] = "pos_gen_non_hermitian"

solver2 = SLEPcEigenSolver(K2,M2)
solver2.parameters['spectrum'] = "smallest magnitude"
solver2.parameters['tolerance'] = 1e-6
solver2.parameters['problem_type'] = "pos_gen_non_hermitian"

# Solve the Initial Problem
solver0.solve(1)
solver1.solve(1)
solver2.solve(1)

# Extraction of EigenValues
r, c, rx, cx = solver0.get_eigenpair(1)
print("Target Frequency: ", sqrt(r))

r1, c1, rx1, cx1 = solver1.get_eigenpair(0)
print("Target Frequency: ", sqrt(r1))

r2, c2, rx2, cx2 = solver2.get_eigenpair(0)
print("Target Frequency: ", sqrt(r2))

# Write ParaViews of the Initial Solution
u0 = Function(V)
u1 = Function(V)
u2 = Function(V)

u0.vector()[:] = rx
u1.vector()[:] = rx1
u2.vector()[:] = rx2 

plot(u0)
vtkfile0 = File('SchwarzMembraneEigenvalueProblem/omega0.pvd')
vtkfile0 << u0
vtkfile1 = File('SchwarzMembraneEigenvalueProblem/omega1.pvd')
vtkfile1 << u1
vtkfile2 = File('SchwarzMembraneEigenvalueProblem/omega2.pvd')
vtkfile2 << u2
