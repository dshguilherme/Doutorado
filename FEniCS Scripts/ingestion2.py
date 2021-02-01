import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


m = np.load('fin_H1.npy')
X = np.logspace(-3, .5, num=50, endpoint=True, base=10)
Y = np.linspace(start=0.12526996, stop=0.63385829, num=10, endpoint=True)
Z = np.divide(m.T, X)
X,Y = np.meshgrid(np.log10(X), Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Z)
ax.set_ylabel('Overlapping Area')
ax.set_xlabel('log(k)')
ax.set_zlabel(r'$k^{-1}$ * Relative Error Norm')
plt.savefig('surface_fin_H1.png')
plt.close()

