import numpy as np
import pandas as pd

from SCHWARZ_TransferFunction import schwarz_transfer_membrane

overlaps = np.linspace(0.1, 0.9, num=9)
polynomial_order = np.linspace(1,4, num=4)
errorL2 = list()
errorH1 = list()
degree = list()
overlap = list()
for pol in polynomial_order:
    for ov in overlaps:
        p = int(pol)
        L2, H1 = schwarz_transfer_membrane(L=1.5, h=1, o=ov, mesh_resolution=10,
                                            number_of_refinements=1, max_iterations=10,
                                            polynomial_order=p)
        errorL2.append(L2)
        errorH1.append(H1)
        degree.append(p)
        overlap.append(ov)

df = pd.DataFrame(data={'L2': errorL2, 'H1': errorH1, 'Degree': degree,
                           'Overlap Ratio':overlap})
df.to_csv('lateral_test.csv')                                  