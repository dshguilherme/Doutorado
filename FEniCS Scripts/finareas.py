import numpy as np
import matplotlib.pyplot as plt

from membrane_generator import FinShaped

fins = list()
L = np.linspace(start=30, stop=121.6, num=10, endpoint=True)

for ell in L:
    ff = fin_shaped = FinShaped(mesh_resolution=30, polynomial_degree=1,
                                adjust=1, a=131.32/2, b=85.09/2, 
                                L=ell,
                                h=50.80, foot=3.3)
    fins.append(ff)

Y = np.linspace(start=0.12526996, stop=0.63385829, num=10, endpoint=True)

Ls = list()
Lss = list()
Hs = list()
Hss = list()
A, _ ,_, _ = fins[0].calculate_areas()
count = 0
for fin in fins:
    LL, HH, L2, H1 = fin.report_results(num_of_iterations=10, mode_num=0, supress_results=True)
    Ls.append(L2[-1]/(A*Y[count]))
    Lss.append(LL[-1])
    Hs.append(H1[-1]/(A*Y[count]))
    Hss.append(HH[-1])
    count+=1


fig, ax = plt.subplots()
ax.plot(Y,Ls)
ax.legend(loc='upper right')
ax.set_ylabel('L2 error norm per area')
ax.set_xlabel('Overlapping area')
plt.grid(b=True, ls='-.')
plt.savefig('fin_l2_overlap.png')
plt.cla()
plt.clf()
plt.close()

fig, ax = plt.subplots()
ax.plot(Y,Hs)
ax.legend(loc='upper right')
ax.set_ylabel('H1 error norm per area')
ax.set_xlabel('Overlapping area')
plt.grid(b=True, ls='-.')
plt.savefig('fin_h1_overlap.png')
plt.cla()
plt.clf()
plt.close()

fig, ax = plt.subplots()
ax.plot(Y,Lss)
ax.legend(loc='upper right')
ax.set_ylabel('L2 error norm per area')
ax.set_xlabel('Overlapping area')
plt.grid(b=True, ls='-.')
plt.savefig('fin_ll_overlap.png')
plt.cla()
plt.clf()
plt.close()

fig, ax = plt.subplots()
ax.plot(Y,Hss)
ax.legend(loc='upper right')
ax.set_ylabel('H1 error norm per area')
ax.set_xlabel('Overlapping area')
plt.grid(b=True, ls='-.')
plt.savefig('fin_hh_overlap.png')
plt.cla()
plt.clf()
plt.close()