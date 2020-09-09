from fenics import *
from dolfin import *
from petsc4py import PETSc

# Create the mesh
nx = 25
ny = 25
mesh = RectangleMesh(Point(0.0,0.0), Point(1.5,1), nx,ny, "right/left")

# Function Space
V = FunctionSpace(mesh,"Lagrange",1) # FunctionSpace of Lagrange Poly degree 1

# Setting up Homogeneous Dirichlet BC
bcs = DirichletBC(V, 0.0, 'on_boundary')

# Assembly
u = TrialFunction(V)
v = TestFunction(V)

k = inner(grad(u),grad(v))*dx
m = u*v*dx

K = PETScMatrix()
assemble(k, tensor=K)

M = PETScMatrix()
assemble(m, tensor=M)

# Apply Boundary Conditions
bcs.apply(K)

# Create the eigensolver
eigensolver = SLEPcEigenSolver(K,M)
eigensolver.parameters['spectrum'] = "smallest magnitude"
eigensolver.parameters['tolerance'] = 1e-6
eigensolver.parameters['problem_type'] = "pos_gen_non_hermitian"

eigensolver.solve(6)

r, c, rx, cx = eigensolver.get_eigenpair(1)
print("Smallest Omega: ", sqrt(r))

# Initialize Function and assign eigenvector
u = Function(V)
u.vector()[:] = rx 


#plot Eigenfunction
plot(u)
vtkfile = File('MembraneEigenValueProblem/solution.pvd')
vtkfile << u