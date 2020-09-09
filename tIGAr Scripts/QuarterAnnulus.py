from tIGAr import *
from tIGAr.NURBS import *
from igakit.nurbs import NURBS as NURBS_ik
from igakit.io import PetIGA
#from igakit.plot import plt
import matplotlib.pyplot as plt
import math
 
vKnots = [0.0,0.0,0.0,1.0,1.0,1.0]
uKnots = [0.0,0.0,1.0,1.0]
 
cpArray = array([[[1.0,0.0],[1.0,1.0],[0.0,1.0]],
                    [[2.0,0.0],[2.0,2.0],[0.0,2.0]]])
                    
w = array([[[1.0],[sqrt(2)/2],[1.0]],[[1.0],[sqrt(2)/2],[1.0]]])
ikNURBS = NURBS_ik([uKnots,vKnots],cpArray)
ikNURBS.elevate(0,1)
print(ikNURBS.degree)
print(ikNURBS.knots)
 
numNewKnots = 1
for i in range(0,6):
    numNewKnots *= 2
h = 1.0/float(numNewKnots)
numNewKnots -= 1
knotList = []
for i in range(0,numNewKnots):
    knotList += [float(i+1)*h,]
newKnots = array(knotList)
ikNURBS.refine(0,newKnots)
ikNURBS.refine(1,newKnots)
 
print(ikNURBS.knots)
#plt.figure()
#plt.surface(ikNURBS)
 
#plt.save('surf.png')
 
 
if(mpirank==0):
    PetIGA().write("out.dat",ikNURBS)
MPI.barrier(worldcomm)
 
if(mpirank==0):
    print("Generating extraction...")
    
splineMesh = NURBSControlMesh("out.dat",useRect=True)
 
splineGenerator = EqualOrderSpline(1,splineMesh)
 
field = 0
scalarSpline = splineGenerator.getScalarSpline(field)
for parametricDirection in [0,1]:
    for side in [0,1]:
        sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
        splineGenerator.addZeroDofs(field,sideDofs)
 
QUAD_DEG = 4
 
spline = ExtractedSpline(splineGenerator,QUAD_DEG)
#plot(spline.mesh)
#plt.savefig('mesh.png')
#so funciona plotar pro mawtplotlib se tiver em useRect=true (triangular mesh), entao exportar pro paraview
 
u = spline.rationalize(TrialFunction(spline.V))
v = spline.rationalize(TestFunction(spline.V))
 
x = spline.spatialCoordinates()
#tem que ver se isso ta certo, mas a ideia é que u nas boundaries desaparecem (0) por causa das cc aplicadas la em cima, entao acho que isso ta ok ?
soln = sin(pi*x[0])*sin(pi*x[1])*(x[0]*x[0]+x[1]*x[1]-1)*(x[0]*x[0]+x[1]*x[1]-4)
f = -spline.div(spline.grad(soln))
 
a = inner(spline.grad(u),spline.grad(v))*spline.dx
L = inner(f,v)*spline.dx
 
u_hom = Function(spline.V)
spline.solveLinearVariationalProblem(a==L,u_hom)
 
 
 
u_hom.rename("u","u")
File("results/u.pvd") << u_hom
 
nsd = 3
for i in range(0,nsd+1):
    name = "F"+str(i)
    spline.cpFuncs[i].rename(name,name)
    File("results/"+name+"-file.pvd") << spline.cpFuncs[i]
 
L2_error = math.sqrt(assemble(((spline.rationalize(u_hom)-soln)**2)*spline.dx))
L_error = assemble((spline.rationalize(u_hom)-soln)*spline.dx)
print(L2_error)
print(L_error)