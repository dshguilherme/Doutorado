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
    def __init__(self, filepath, geometry, overlaps, domains1, domains2, boundaries):
        self.type = geometry
        self.path = filepath
        self.overlaps = overlaps
        self.d1 = domains1
        self.d2 = domains2
        self.insides = boundaries[0]
        self.outsides = boundaries[1]
        self.left_outsides = boundaries[2]
        self.left_robin = boundaries[3]
        self.right_outsides = boundaries[4]
        self.right_robin = boundaries[5]

    
    def generate_overlap_error_graphs(self):

        LL2 = list()
        aaL2 = list()
        HH1 = list()
        aaH1 = list()
        
        for idx in range(len(self.overlaps)):
            left = d1[idx]
            right = d2[idx]
            domain = left + right
            membrane = Membrane(domain, left, right, mesh_resolution=36, polynomial_degree=1,
                                 adjust=0.01)
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
            
            LL2.append(L2)
            aaL2.append(L2/self.overlaps[idx])
            HH1.append(H1)
            aaH1.append(H1/self.overlaps[idx])

        
        fig, ax = plt.subplots()
        ax.plot(self.overlaps, LL2, label='L2 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('L2 Error Norm', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig(self.path +'l2byOA.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.overlaps, HH1, label='H1 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('H1 Error Norm', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig(self.path +'h1byOA.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.overlaps, aaL2, label='L2 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('L2 Error Norm by Area', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig(self.path +'aal2byOA.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.overlaps, aaH1, label='H1 norm after 2 iterations')
        ax.legend(loc='upper right')
        ax.set_ylabel('H1 Error Norm by Area', fontsize = 18)
        ax.set_xlabel('Overlap Area Percentage', fontsize = 18)
        plt.grid(b=True)
        plt.savefig(self.path +'aah1byOA.png')
        plt.close()
    
    def generate_overlap_k_surface(self):

        constants = np.logspace(-4,0.5,50, base=10)
        LL2 = np.ndarray((len(self.overlaps),50))
        HH1 = np.ndarray((len(self.overlaps),50))
        rows = 0
        cols = 0

        for k in constants:
            for idx in range(len(self.overlaps)):
                left = d1[idx]
                right = d2[idx]
                domain = left + right
                membrane = Membrane(domain, left, right, mesh_resolution=36, polynomial_degree=1,
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
        plt.savefig(self.path +'l2surface.png')
        plt.close()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X,Y,Z2)
        plt.title('H1 Error Norm')
        ax.set_ylabel('Overlapping Area')
        ax.set_xlabel('log(k)')
        ax.set_zlabel(r'$k^{-1} * Error Norm')
        plt.savefig(self.path +'h1surface.png')
        plt.close()

    def schwarz_iterations(self):
        k = 0.01
        idx = np.argmin(self.overlaps)
        left = self.d1[idx]
        right = self.d2[idx]
        domain = left+right
        membrane = Membrane(domain, left, right, mesh_resolution=36, polynomial_degree=1,
                                    adjust=k)
        outsides = self.outsides[idx]
        insides = self.insides[idx]
        left_outsides = self.left_outsides[idx]
        left_robin = self.left_robin[idx]
        right_outsides = self.right_outsides[idx]
        right_robin = self.right_robin[idx]
        L2, H1, SH1, u1, u2, r1, r2 = membrane_iterator(insides=insides, 
                                        outsides=outsides,
                                        left_outsides=left_outsides,
                                        left_robin=left_robin,
                                        right_outsides=right_outsides,
                                        right_robin=right_robin,
                                        num_of_iterations=2,
                                        membrane=membrane, mode_num=0)        
        freqs, vecs = membrane.initial_solution(outsides, left_outsides,
                                right_outsides, mode_number=0)
        u = Function(membrane.V)
        u.vector()[:] = vecs[0]
        r = freqs[0]
        generate_relatory(self.path+'/bestsample/', membrane, L2, H1, SH1, u,
                            u1, u2, r, r1, r2, vecs)




