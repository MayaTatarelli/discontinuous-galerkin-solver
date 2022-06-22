import numpy as np
import DGSolver
import sys
sys.path.append('../quickplotlib/lib/')
from quickplotlib import plotfxn, plot_matrix_sparsity_pattern
import matplotlib . pyplot as plt

number_of_runs = 10
num_elements_values = [5]

for i in range(1,number_of_runs):
	num_elements_values.append(num_elements_values[i-1]+10)
num_elements_values = np.array(num_elements_values)

# print(len(num_elements_values))
# print(num_elements_values)

error_values = -1.0*np.ones(number_of_runs)

for iElements in range(len(num_elements_values)):
	print("Running dg_solver with " + str(num_elements_values[iElements]) + " elements;")
	dg_solver = DGSolver.DGSolver(domain_left=0.0,
								  domain_right=(2.0*np.pi),
								  poly_degree=1,
								  number_of_elements=num_elements_values[iElements],
								  cfl=0.0075,
								  a=(2.0*np.pi),
								  rk4=True)
	error_values[iElements] = dg_solver.advance_solution_time()
	print("Error: " + str(error_values[iElements]))
	np.savetxt("error_values_P1_cfl0075_rk4.csv", error_values, delimiter=',') #, fmt='%d')

# print(error_values)
# np.savetxt("error_values_P2_cfl0015.csv", error_values, delimiter=',') #, fmt='%d')
# plt.figure()
# plt.plot(num_elements_values, error_values)
# plt.show()