import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


m = np.load('ell_L2.npy')
X = np.logspace(-3, .5, num=50, endpoint=True, base=10)
Y = np.linspace(start=0.1, stop=1, num=10, endpoint=True)
Z = np.divide(m.T, X)
X,Y = np.meshgrid(np.log10(X), Y/6)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Z)
ax.set_ylabel('Overlapping Area')
ax.set_xlabel('log(k)')
ax.set_zlabel(r'$k^{-1}$ * Relative Error Norm')
plt.savefig('surface_ell_L2.png')
plt.close()

