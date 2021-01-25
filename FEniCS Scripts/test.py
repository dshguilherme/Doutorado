import numpy as np

from membrane_generator import Rectangular, LShaped, TShaped, FinShaped

rec_LL = np.ndarray([50,10])
rec_L2 = np.ndarray([50,10])
rec_HH = np.ndarray([50,10])
rec_H1 = np.ndarray([50,10])

rec_overlaps = np.linspace(start=0.1, stop=1, num=10 ,  endpoint=True)
rec_adjust = np.logspace(-3, .5, num=50, endpoint=True, base=10)


ell_LL = np.ndarray([50,10])
ell_L2 = np.ndarray([50,10])
ell_HH = np.ndarray([50,10])
ell_H1 = np.ndarray([50,10])

ell_overlaps = np.linspace(start=0.1, stop=1, num=10 ,  endpoint=True)
ell_adjust = np.logspace(-3, .5, num=50, endpoint=True, base=10)

t_LL = np.ndarray([50,10])
t_L2 = np.ndarray([50,10])
t_HH = np.ndarray([50,10])
t_H1 = np.ndarray([50,10])

t_overlaps = np.linspace(start=0.1, stop=1, num=10 ,  endpoint=True)
t_adjust = np.logspace(-3, .5, num=50, endpoint=True, base=10)

fin_LL = np.ndarray([50,10])
fin_L2 = np.ndarray([50,10])
fin_HH = np.ndarray([50,10])
fin_H1 = np.ndarray([50,10])

fin_overlaps = np.linspace(start=30, stop=121.6, num=10 ,  endpoint=True)
fin_adjust = np.logspace(-3, .5, num=50, endpoint=True, base=10)

for k in range(50):
    for x in range(10):

        rec_shaped = Rectangular(mesh_resolution=30, polynomial_degree=1,
                                    adjust=rec_adjust[k], h=1, L=1.5,
                                    o=rec_overlaps[x])
        
        LL, HH, L2, H1 = rec_shaped.report_results(num_of_iterations=2,
                                         mode_num=0, supress_results=True)
        rec_LL[k,x] = LL[-1]
        rec_L2[k,x] = L2[-1]
        rec_HH[k,x] = HH[-1]
        rec_H1[k,x] = H1[-1]


        ell_shaped = LShaped(mesh_resolution=30, polynomial_degree=1,
                                adjust=ell_adjust[k], L1=1., L2=1., 
                                h1=1., h2=2., b=ell_overlaps[x])
        
        LL, HH, L2, H1 = ell_shaped.report_results(num_of_iterations=2,
                                         mode_num=0, supress_results=True)
        ell_LL[k,x] = LL[-1]
        ell_L2[k,x] = L2[-1]
        ell_HH[k,x] = HH[-1]
        ell_H1[k,x] = H1[-1]

        t_shaped = TShaped(mesh_resolution=30, polynomial_degree=1,
                            adjust=t_adjust[k], L=1, h=1,
                             c=t_overlaps[x])
        LL, HH, L2, H1 = t_shaped.report_results(num_of_iterations=2,
                                         mode_num=0, supress_results=True)
        t_LL[k,x] = LL[-1]
        t_L2[k,x] = L2[-1]
        t_HH[k,x] = HH[-1]
        t_H1[k,x] = H1[-1]

        fin_shaped = FinShaped(mesh_resolution=30, polynomial_degree=1,
                                adjust=fin_adjust[k], a=131.32/2, b=85.09/2, 
                                L=fin_overlaps[x],
                                h=50.80, foot=3.3)
        LL, HH, L2, H1 = fin_shaped.report_results(num_of_iterations=2,
                                         mode_num=0, supress_results=True)
        fin_LL[k,x] = LL[-1]
        fin_L2[k,x] = L2[-1]
        fin_HH[k,x] = HH[-1]
        fin_H1[k,x] = H1[-1]

np.save('rec_LL.npy', rec_LL)
np.save('rec_L2.npy', rec_L2)
np.save('rec_HH.npy', rec_HH)
np.save('rec_H1.npy', rec_H1)

np.save('ell_LL.npy', ell_LL)
np.save('ell_L2.npy', ell_L2)
np.save('ell_HH.npy', ell_HH)
np.save('ell_H1.npy', ell_H1)

np.save('t_LL.npy', t_LL)
np.save('t_L2.npy', t_L2)
np.save('t_HH.npy', t_HH)
np.save('t_H1.npy', t_H1)

np.save('fin_LL.npy', fin_LL)
np.save('fin_L2.npy', fin_L2)
np.save('fin_HH.npy', fin_HH)
np.save('fin_H1.npy', fin_H1)