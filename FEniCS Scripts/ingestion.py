import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.io

def plot_error_surface(filename, title):
    m = np.load(filename+'.npy')
    X = np.logspace(-2,0.6937,50,base=10)
    Z = np.divide(m.T,X)
    Y = np.logspace(-2,0,50,base=10)
    X,Y = np.meshgrid(np.log10(X),Y)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,Z)
    plt.title(title)
    ax.set_ylabel('Overlapping Area')
    ax.set_xlabel('log(k)')
    ax.set_zlabel(r'$k^{-1}$ * Error Norm')
    plt.savefig(filename+'.png')
    plt.close()

plot_error_surface('rectangular/surfaces/gradL2', title='Total L2 Error')
plot_error_surface('rectangular/surfaces/gradaaL2', title='L2 Error per unit area')
plot_error_surface('rectangular/surfaces/gradH1', title='Total H1 Error')
plot_error_surface('rectangular/surfaces/gradaaH1', title='H1 Error per unit area')