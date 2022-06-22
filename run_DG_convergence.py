import numpy as np
import DGSolver
import sys
#sys.path.append('../quickplotlib/lib/')
#from quickplotlib import plotfxn, plot_matrix_sparsity_pattern
import matplotlib . pyplot as plt

number_of_runs = 7
num_elements_values = [5]

for i in range(1,number_of_runs):
	num_elements_values.append(num_elements_values[i-1]+10)
num_elements_values = np.array(num_elements_values)

# print(len(num_elements_values))
# print(num_elements_values)

error_values = -1.0*np.ones(number_of_runs)

#Variables to be set for DGSolver
#cur_poly_degree = 2
cur_cfl = 0.001
cur_cfl_name = "001"
cur_rk4 = True

for cur_poly_degree in range(2,5):
	#Set filename
	if(cur_rk4):
		file_name = "error_values_P{}_cfl{}_rk4.csv".format(cur_poly_degree,cur_cfl_name)
	else:
		file_name = "error_values_P{}_cfl{}.csv".format(cur_poly_degree,cur_cfl_name)
	print(file_name)

	for iElements in range(len(num_elements_values)):
		print("Running dg_solver with " + str(num_elements_values[iElements]) + " elements;")
		dg_solver = DGSolver.DGSolver(domain_left=0.0,
									domain_right=(2.0*np.pi),
									poly_degree=cur_poly_degree,
									number_of_elements=num_elements_values[iElements],
									cfl=cur_cfl,
									a=(2.0*np.pi),
									rk4=cur_rk4)
		error_values[iElements] = dg_solver.advance_solution_time()
		print("Error: " + str(error_values[iElements]))
		np.savetxt(file_name, error_values, delimiter=',') #, fmt='%d')

# print(error_values)
# np.savetxt("error_values_P2_cfl0015.csv", error_values, delimiter=',') #, fmt='%d')
# plt.figure()
# plt.plot(num_elements_values, error_values)
# plt.show()