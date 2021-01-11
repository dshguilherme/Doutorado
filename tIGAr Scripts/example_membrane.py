from tIGAr import *
from tIGAr.NURBS import *
import math
import matplotlib.pyplot as plt

a = 1. 
b = 1.5 
o = 0.1
p = 3



uKnots = [0.,]*(p+1) + [1.,]*(p+1)
vKnots = [0.,]*(p+1) + [1.,]*(p+1)

cpArray = array(
    [ [[0., 0.],    [0., a/3],    [0., 2*a/3],    [0., a]],
      [[b/3, 0.],   [b/3, a/3],   [b/3, 2*a/3],   [b/3, a]],
      [[2*b/3, 0.], [2*b/3, a/3], [2*b/3, 2*a/3], [2*b/3, a]],
      [[b, 0.],     [b, a/3],     [b, 2*a/3],     [b, a]]     
    ]    
)

w = array([[1.,]*4,]*4)

ikNURBS = NURBS_ik([uKnots, vKnots], cpArray)

# Refinement
REF_LEVEL = 4

numNewKnots = 1
for i in range(0, REF_LEVEL):
        numNewKnots *= 2

h = 1.0/float(numNewKnots)
numNewKnots -= 1
knotList = []
for i in range(0,numNewKnots):
    knotList +=([float(i+1)*h,])
newKnots = array(knotList)
print(newKnots)
ikNURBS.refine(0,newKnots)
ikNURBS.refine(1,newKnots)


splineMesh = NURBSControlMesh(ikNURBS)
splineGenerator = EqualOrderSpline(1,splineMesh)

field = 0
scalarSpline = splineGenerator.getScalarSpline(field)
for parametricDirection in [0,1]:
    for side in [0,1]:
        sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
        splineGenerator.addZeroDofs(field,sideDofs)

QUAD_DEG = 2*p

spline = ExtractedSpline(splineGenerator, QUAD_DEG)

u = TrialFunction(spline.V)
v = TestFunction(spline.V)


k = inner(spline.grad(u), spline.grad(v))*spline.dx
m = inner(u,v)*spline.dx

K = spline.assembleMatrix(k, diag=1.0/DOLFIN_EPS)
M = spline.assembleMatrix(m)

solver = SLEPcEigenSolver(K,M)
solver.parameters['spectrum'] = "smallest magnitude"
solver.solve()

for n in range(0,6):
    omega2, _, uVectorIGA, _ = solver.get_eigenpair(n)
    print("omega_"+str(n)+" = " +str(math.sqrt(omega2)))
    u = Function(spline.V)
    u.vector()[:] = spline.M*uVectorIGA
    filestr = File("paraview"+str(n)+".pvd")
    filestr << u
