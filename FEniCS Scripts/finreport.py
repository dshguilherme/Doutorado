import numpy as np

from membrane_generator import FinShaped

rec_shaped = FinShaped(mesh_resolution=30, polynomial_degree=1, 
                                adjust=0.05,a=131.32/2, b=85.09/2, 
                                L=60, h=50.80, foot=3.3)

rec_shaped.report_results(num_of_iterations=10, mode_num=0, supress_results=False)