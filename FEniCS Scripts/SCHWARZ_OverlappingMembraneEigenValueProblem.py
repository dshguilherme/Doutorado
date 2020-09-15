from fenics import *
from dolfin import *
from petsc4py import PETSc
from matplotlib import pyplot as plt

## Geometries and Meshes Definition
nx = 24
ny = 24
MAX_ITER = 10 # Number of maximum iterations of the algorithm

omega = RectangleMesh(Point(0.0,0.0), Point(1.5,1),nx,ny, "right/left") # Whole domain

o = 0.7 # Total Overlapping Area
b = 1.5*(1-o)/2 # Total non-Overlapping Area

omega1 = RectangleMesh(Point(0.0,0.0), Point(b+o*1.5,1), nx, ny, "right/left") # First SubDomain
omega2 = RectangleMesh(Point(b,0.0), Point(1.5,1), nx, ny, "right/left") # Second SubDomain

# Function Spaces
V = FunctionSpace(omega,"Lagrange",1)
V1 = FunctionSpace(omega1,"Lagrange",1)
V2 = FunctionSpace(omega2,"Lagrange",1)

# Defining Boundary Domains
 # Omega
class DirichletBoundaryOmega(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary
 # Boundary Markers
omega_markers = MeshFunction('size_t', omega, omega.topology().dim()-1,0)
omega_markers.set_all(9999)
bc_D = DirichletBoundaryOmega()
bc_D.mark(omega_markers, 0)
        
 # Omega 1       
class DirichletBoundaryOmega1(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and (x[0] < b +1.5*o)
class RobinBoundaryOmega1(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and (x[0] >= b+1.5*o -DOLFIN_EPS)
 # Boundary Markers
omega1_markers = MeshFunction('size_t', omega1, omega1.topology().dim()-1,0)
omega1_markers.set_all(9999)
bc_D1 = DirichletBoundaryOmega1()
bc_R1 = RobinBoundaryOmega1()
bc_D1.mark(omega1_markers, 0)
bc_R1.mark(omega1_markers, 1)

 # Omega 2
class DirichletBoundaryOmega2(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and (x[0] > b)
class RobinBoundaryOmega2(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and (x[0] < b+DOLFIN_EPS)
 # Boundary Markers
omega2_markers = MeshFunction('size_t', omega2, omega2.topology().dim()-1,0)
omega2_markers.set_all(9999)
bc_D2 = DirichletBoundaryOmega2()
bc_D2.mark(omega2_markers, 0)
bc_R2 = RobinBoundaryOmega2()
bc_R2.mark(omega2_markers, 1)

## Assemblies

# Trial/Test Functions
u = TrialFunction(V)
u1 = TrialFunction(V1)
u2 = TrialFunction(V2)

v = TestFunction(V)
v1 = TestFunction(V1)
v2 = TestFunction(V2)

# Bilinear Forms
k = inner(grad(u),grad(v))*dx
m = u*v*dx

k1 = inner(grad(u1),grad(v1))*dx
m1 = u1*v1*dx

k2 = inner(grad(u2),grad(v2))*dx
m2 = u2*v2*dx

# Matrices
 # Stiffness
K = PETScMatrix()
K1 = PETScMatrix()
K2 = PETScMatrix()

assemble(k, tensor=K)
assemble(k1, tensor=K1)
assemble(k2, tensor=K2) 

# Applying Boundary Conditions
bcs = DirichletBC(V, 0.0, bc_D)
bcs.apply(K)

bcs1 = DirichletBC(V1, 0.0, bc_D1)
bcs1.apply(K1)

bcs2 = DirichletBC(V2, 0.0, bc_D2)
bcs2.apply(K2)

 # Mass
M = PETScMatrix()
M1 = PETScMatrix()
M2 = PETScMatrix()

assemble(m, tensor=M)
assemble(m1, tensor=M1)
assemble(m2, tensor=M2)

# EigenSolver
 # Create the eigensolvers
omegasolver = SLEPcEigenSolver(K,M)
omegasolver.parameters['spectrum'] = "smallest magnitude"
omegasolver.parameters['tolerance'] = 1e-6
omegasolver.parameters['problem_type'] = "pos_gen_non_hermitian"

omega1solver = SLEPcEigenSolver(K1,M1)
omega1solver.parameters['spectrum'] = "smallest magnitude"
omega1solver.parameters['tolerance'] = 1e-6
omega1solver.parameters['problem_type'] = "pos_gen_non_hermitian"

omega2solver = SLEPcEigenSolver(K2,M2)
omega2solver.parameters['spectrum'] = "smallest magnitude"
omega2solver.parameters['tolerance'] = 1e-6
omega2solver.parameters['problem_type'] = "pos_gen_non_hermitian"
 
 # Solve the initial problem
omegasolver.solve(1)
omega1solver.solve(1)
omega2solver.solve(1)

# Eigenvalue Extraction
r, c, rx, cx = omegasolver.get_eigenpair(0)
print("Target Frequency: ", sqrt(r))

r1, c1, rx1, cx1 = omega1solver.get_eigenpair(0)
print("Omega 1 Frequency: ", sqrt(r1))
r2, c2, rx2, cx2 = omega2solver.get_eigenpair(0)

print("Omega 2 Frequency: ", sqrt(r2))

# Writing Paraviews of the Initial Solution
uu = Function(V)
uu.vector()[:] = rx

uu1 = Function(V1)
uu1.vector()[:] = rx1

uu2 = Function(V2)
uu2.vector()[:] = rx2

plot(uu)
vtkfile = File('SchwarzMembraneEigenvalueProblem/omega.pvd')
vtkfile << uu

plot(uu1)
vtkfile = File('SchwarzMembraneEigenvalueProblem/omega1.pvd')
vtkfile << uu1

plot(uu2)
vtkfile = File('SchwarzMembraneEigenvalueProblem/omega2.pvd')
vtkfile << uu2


## Schwarz Alternating Algorithm
iteration = 0
while (iteration <= MAX_ITER):
    # First Step: Do F1 = du2/dn2 G1 = u2
    g1 = Function(V1)
    LagrangeInterpolator.interpolate(uu2,g1)
    f1 = Function(V1)

    ff2 = project(uu2.dx(0) +uu2.dx(1),V2)
    LagrangeInterpolator.interpolate(ff2,f1)

    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)

    ## Solve for u1
    dx = Measure('dx', domain = omega1)
    ds = Measure('ds', subdomain_data = omega1_markers)
    ak1 = inner(grad(u1),grad(v1))*dx -(1-g1)*(u1.dx(0) +u1.dx(1))*v1*ds(1) +f1*u1*v1*ds(1)
    am1 = u1*v1*dx

    K1 = PETScMatrix()
    bcs1 = DirichletBC(V1, 0.0, bc_D1)
    assemble(ak1, tensor=K1)
    bcs1.apply(K1)

    M1 = PETScMatrix()
    assemble(am1, tensor=M1)

    omega1solver = SLEPcEigenSolver(K1,M1)
    omega1solver.parameters['spectrum'] = "smallest magnitude"
    omega1solver.parameters['tolerance'] = 1e-6
    omega1solver.parameters['problem_type'] = "pos_gen_non_hermitian"
    omega1solver.solve(0)
    r1, c1, rx1, cx1 = omega1solver.get_eigenpair(0)
    print("Omega 1 Frequency: ", sqrt(r1))

    uu1 = Function(V1)
    uu1.vector()[:] = rx1
    plot(uu1)
    vtkfile1 = File('SchwarzMembraneEigenvalueProblem/omega1_iter' +str(iteration)+'.pvd')
    vtkfile1 << uu1

    ## Do F1 = du1/dn1 G2 = u1
    g2 = Function(V2)
    LagrangeInterpolator.interpolate(uu1,g2)
    f2 = Function(V2)
    ff1 = project(uu1.dx(0)+uu1.dx(1),V1)
    LagrangeInterpolator.interpolate(ff1,f2)

    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)

    ## Solve for u2
    dx = Measure('dx', domain = omega2)
    ds = Measure('ds', subdomain_data = omega2_markers)
    ak2 = inner(grad(u2),grad(v2))*dx -(1-g2)*(u2.dx(0) +u2.dx(1))*v2*ds(1) +f2*u2*v2*ds(1)
    am2 = u2*v2*dx

    K2 = PETScMatrix()
    bcs2 = DirichletBC(V2, 0.0, bc_D2)
    assemble(ak2, tensor=K2)
    bcs2.apply(K2)

    M2 = PETScMatrix()
    assemble(am2, tensor=M2)

    omega2solver = SLEPcEigenSolver(K2,M2)
    omega2solver.parameters['spectrum'] = "smallest magnitude"
    omega2solver.parameters['tolerance'] = 1e-6
    omega2solver.parameters['problem_type'] = "pos_gen_non_hermitian"
    omega2solver.solve(0)
    r2, c2, rx2, cx2 = omega2solver.get_eigenpair(0)
    print("Omega 2 Frequency: ", sqrt(r2))
    iteration += 1

    uu2 = Function(V2)
    uu2.vector()[:] = rx2
    plot(uu2)
    vtkfile1 = File('SchwarzMembraneEigenvalueProblem/omega2_iter'+str(iteration)+'.pvd')
    vtkfile1 << uu2