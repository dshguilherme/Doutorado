from dolfin import(SubDomain, MeshFunction, refine, FunctionSpace, 
                     TrialFunction, TestFunction, inner, grad, dx, 
                     PETScMatrix, assemble, DirichletBC, SLEPcEigenSolver,
                     Function, Measure, ds, Point, Constant, DOLFIN_EPS,
                     near, between, sqrt, project, File, plot,
                     PETScDMCollection, interpolate)
import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from Membrane import (Membrane, standard_solver, membrane_iterator,
                         generate_relatory)


class Relatory:
    def __init__(self, overlaps, membranes):
        self.overlaps = overlaps
        self.membranes = membranes

    
    def generate_overlap_error_graphs(self):

        LL2 = list()
        aaL2 = list()
        HH1 = list()
        aaH1 = list()
        
        for idx in range(len(self.overlaps)):
            membrane = self.membranes[idx]
            L2, H1, _, _, _, _, _ = membrane_iterator(
                                            num_of_iterations=2,
                                            membrane=membrane, mode_num=0)
            
            LL2.append(L2[-1])
            aaL2.append(L2[-1]/self.overlaps[idx])
            HH1.append(H1[-1])
            aaH1.append(H1[-1]/self.overlaps[idx])

       
        fig, ax = plt.subplots()
        ax.plot(self.overlaps, LL2, label='L2 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('L2 Error Norm', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig('l2byOA.png')
        plt.cla()
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.overlaps, HH1, label='H1 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('H1 Error Norm', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig('h1byOA.png')
        plt.cla()
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.overlaps, aaL2, label='L2 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('L2 Error Norm by Area', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig('aal2byOA.png')
        plt.cla()
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.overlaps, aaH1, label='H1 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('H1 Error Norm by Area', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig('aah1byOA.png')
        plt.cla()
        plt.clf()
        plt.close()
    
    def generate_overlap_k_surface(self):

        constants = np.logspace(-4,0.5,50, base=10)
        LL2 = np.ndarray((len(self.overlaps),50))
        HH1 = np.ndarray((len(self.overlaps),50))
        rows = 0
        cols = 0

        for k in constants:
            for idx in range(len(self.overlaps)):
                left = self.d1[idx]
                right = self.d2[idx]
                domain = self.d
                membrane = Membrane(domain, left, right, mesh_resolution=30, polynomial_degree=1,
                                    adjust=k)
                outsides = self.outsides[idx]
                insides = self.insides[idx]
                left_outsides = self.left_outsides[idx]
                left_robin = self.left_robin[idx]
                right_outsides = self.right_outsides[idx]
                right_robin = self.right_robin[idx]
                L2, H1, _, _, _, _, _ = membrane_iterator(insides=insides, 
                                                outsides=outsides,
                                                left_outsides=left_outsides,
                                                left_robin=left_robin,
                                                right_outsides=right_outsides,
                                                right_robin=right_robin,
                                                num_of_iterations=2,
                                                membrane=membrane, mode_num=0)
                LL2[rows,cols] = L2
                HH1[rows,cols] = H1
                rows+=1

            cols+=1

        X = constants
        Z1 = np.divide(LL2.T,X)
        Z2 = np.divide(HH1.T,X)
        Y = self.overlaps
        X,Y = np.meshgrid(np.log10(X),Y)
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X,Y,Z1)
        plt.title('L2 Error Norm')
        ax.set_ylabel('Overlapping Area')
        ax.set_xlabel('log(k)')
        ax.set_zlabel(r'$k^{-1} * Error Norm')
        plt.savefig('l2surface.png')
        plt.close()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X,Y,Z2)
        plt.title('H1 Error Norm')
        ax.set_ylabel('Overlapping Area')
        ax.set_xlabel('log(k)')
        ax.set_zlabel(r'$k^{-1} * Error Norm')
        plt.savefig('h1surface.png')
        plt.close()

    def schwarz_iterations(self):
        idx = np.argmin(self.overlaps)
        membrane = self.membranes[idx]
        freqs, vecs = membrane.initial_solution(mode_number=0)
        u = Function(membrane.V)
        u.vector()[:] = vecs[0]
        r = freqs[0]        
        L2, H1, SH1, u1, u2, r1, r2 = membrane_iterator(num_of_iterations=10,
                                        membrane=membrane, mode_num=0)        
        generate_relatory(membrane, L2, H1, SH1, u, u1, u2, r, r1, r2, vecs)



