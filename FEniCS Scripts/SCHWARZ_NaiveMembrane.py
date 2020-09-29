from dolfin import *
import mshr

# MESHES #

# Geometrical Data
L = 1.5 # Side of the Rectangle
h = 1.0 # Height of the Rectangle

o = 0.1 # Total overlapping Area
b = L*(1-o)/2 # Non Overlapping Side of the Rectangle

# Mesh
domain = mshr.Rectangle(Point(0., 0.), Point(L, h))
omega = mshr.generate_mesh(domain, 24)

domain_left = domain and mshr.Rectangle(Point(0., 0.), Point(b+L*o, h))
omega_left = mshr.generate_mesh(domain_left, 200)

domain_right = domain and mshr.Rectangle(Point(b, 0), Point(L, h))
omega_right = mshr.generate_mesh(domain_right, 200)

# Boundaries
class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

class OnBoundaryLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < b+L*o - DOLFIN_EPS

class RobinLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], b+L*o)

class OnBoundaryRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > b +DOLFIN_EPS

class RobinRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], b)

# DirichletBCs
omega_bcs = OnBoundary()
left_bcs = OnBoundaryLeft()
right_bcs = OnBoundaryRight()

# Initial Problem Assembly
V = FunctionSpace(omega, "Lagrange", 1)
V1 = FunctionSpace(omega_left, "Lagrange", 1)
V2 = FunctionSpace(omega_right, "Lagrange", 1)

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

# Matrix assembly

# Stiffness
K = PETScMatrix()
K1 = PETScMatrix()
K2 = PETScMatrix()

assemble(k, tensor=K)
bcs = DirichletBC(V, Constant(0.), omega_bcs)
bcs.apply(K)

assemble(k1, tensor=K1)
bcs_left = DirichletBC(V1, Constant(0.), left_bcs)
bcs_left.apply(K1)

assemble(k2, tensor=K2)
bcs_left = DirichletBC(V2, Constant(0.), right_bcs)
bcs_left.apply(K2)

# Mass
M = PETScMatrix()
M1 = PETScMatrix()
M2 = PETScMatrix()

assemble(m, tensor=M)
assemble(m1, tensor=M1)
assemble(m2, tensor=M2)

# EigenSolver

# Omega
solver0 = SLEPcEigenSolver(K,M)
solver0.parameters['spectrum'] = 'smallest magnitude'
solver0.parameters['tolerance'] = 1e-6
solver0.parameters['problem_type'] = 'pos_gen_non_hermitian'

solver0.solve(0)
r, c, rx, cx = solver0.get_eigenpair(0)
print("Target Frequency: ", sqrt(r))

uu = Function(V)
uu.vector()[:] = rx
vtkfile0 = File('SchwarzMembraneNaive/omega.pvd')
vtkfile0 << uu

# Left
solver1 = SLEPcEigenSolver(K1,M1)
solver1.parameters['spectrum'] = 'smallest magnitude'
solver1.parameters['tolerance'] = 1e-6
solver1.parameters['problem_type'] = 'pos_gen_non_hermitian'

solver1.solve(0)
r1, c1, rx1, cx1 = solver1.get_eigenpair(0)
print("Left Frequency: ", sqrt(r1))

uu1 = Function(V1)
uu1.vector()[:] = rx1
vtkfile1 = File('SchwarzMembraneNaive/left.pvd')
vtkfile1 << uu1

# Right
solver2 = SLEPcEigenSolver(K2,M2)
solver2.parameters['spectrum'] = 'smallest magnitude'
solver2.parameters['tolerance'] = 1e-6
solver2.parameters['problem_type'] = 'pos_gen_non_hermitian'

solver2.solve(0)
r2, c2, rx2, cx2 = solver2.get_eigenpair(0)
print("Right Frequency: ", sqrt(r2))

uu2 = Function(V2)
uu2.vector()[:] = rx2
vtkfile2 = File('SchwarzMembraneNaive/right.pvd')
vtkfile2 << uu2

## Schwarz Algorithm ##

# Defining the Robin Boundaries
left_boundaries = MeshFunction("size_t", omega_left,
                                omega_left.topology().dim()-1)
left_robin = RobinLeft()
left_robin.mark(left_boundaries, 1)

right_boundaries = MeshFunction("size_t", omega_right,
                                omega_right.topology().dim()-1)
right_robin = RobinRight()
right_robin.mark(right_boundaries, 1)

# iterate
iter = 1
MAX_ITER = 100
while(iter < MAX_ITER):
    # First Step: Do F1 = du/dn2, G1 = u2
    uu2.set_allow_extrapolation(True)
    f1 = project(uu2.dx(1) -uu2.dx(0), V=V1)
    g1 = project(uu2, V=V1)

    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)

    # Step 2: Solve for u1
    ds = Measure('ds', subdomain_data=left_boundaries)

    k1 = inner(grad(u1),grad(v1))*dx -(1-g1)*(-u1.dx(0) +u1.dx(1))*v1*ds(1) +f1*u1*v1*ds(1)
    K1 = PETScMatrix()
    assemble(k1, tensor=K1)
    bcs_left = DirichletBC(V1, Constant(0.), left_bcs)
    bcs_left.apply(K1)

    solver1 = SLEPcEigenSolver(K1,M1)
    solver1.parameters['spectrum'] = 'smallest magnitude'
    solver1.parameters['tolerance'] = 1e-6
    solver1.parameters['problem_type'] = 'pos_gen_non_hermitian'

    solver1.solve(0)
    r1, c1, rx1, cx1 = solver1.get_eigenpair(0)
    print("Left Iteration " + str(iter) + " Frequency: ", sqrt(r1))

    uu1 = Function(V1)
    uu1.vector()[:] = rx1
    vtkfile1 = File('SchwarzMembraneNaive/left' + str(iter) + '.pvd')
    vtkfile1 << uu1

    # Step 3: Do F2 = du/dn1, G2 = u1
    uu1.set_allow_extrapolation(True)
    f2 = project(uu1.dx(1)-uu1.dx(0), V=V2)
    g2 = project(uu1, V=V2)

    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)

    # Step 2: Solve for u2
    ds = Measure('ds', subdomain_data=right_boundaries)

    k2 = inner(grad(u2),grad(v2))*dx -(1-g2)*(u2.dx(0) +u2.dx(1))*v2*ds(1) +f2*u2*v2*ds(1)
    K2 = PETScMatrix()
    assemble(k2, tensor=K2)
    bcs_right = DirichletBC(V2, Constant(0.), right_bcs)
    bcs_right.apply(K2)

    solver2 = SLEPcEigenSolver(K2,M2)
    solver2.parameters['spectrum'] = 'smallest magnitude'
    solver2.parameters['tolerance'] = 1e-6
    solver2.parameters['problem_type'] = 'pos_gen_non_hermitian'

    solver2.solve(0)
    r2, c2, rx2, cx2 = solver2.get_eigenpair(0)
    print("Right Iteration " + str(iter) + " Frequency: ", sqrt(r2))

    uu2 = Function(V2)
    uu2.vector()[:] = rx2
    vtkfile2 = File('SchwarzMembraneNaive/right' + str(iter) +'.pvd')
    vtkfile2 << uu2
    iter += 1