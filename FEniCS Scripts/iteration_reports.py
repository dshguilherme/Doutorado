import numpy as np
import matplotlib.pyplot as plt

from membrane_generator import Rectangular, LShaped, TShaped, FinShaped

mesh_sizes = np.logspace(0, 2, num=10, base=10, endpoint=True)
ll2 = list()
hh1 = list()
hmin = list()
for size in mesh_sizes[2:len(mesh_sizes)]:
    rec_shaped = FinShaped(mesh_resolution=size, polynomial_degree=1, 
                                adjust=0.05,a=131.32/2, b=85.09/2, 
                                L=30, h=50.80, foot=3.3)

    rL2, rH1, L2, H1 = rec_shaped.report_results(num_of_iterations=2, mode_num=0, supress_results=True)
    hx = rec_shaped.mesh.hmin()
    hh = rec_shaped.mesh.hmax()
    tmp = (hx+hh)/2
    hmin.append(tmp)                      
    ll2.append(L2[-1]/(6))
    hh1.append(H1[-1]/(6))

fig, ax = plt.subplots()
ax.plot(hmin, ll2, label='L2 norm after 2 iterations')
ax.plot(hmin, hh1, label='H1 norm after 2 iterations')
ax.legend(loc='upper left')
ax.set_ylabel('Error Norms/Overlapping Area')
ax.set_xlabel('Mean element h size')
plt.grid(b=True, ls='-.')
plt.xscale('log')
plt.yscale('log')
plt.savefig('rec_refinement_l2.png')
plt.clf()
plt.cla()
plt.close()