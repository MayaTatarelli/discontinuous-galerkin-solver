import numpy as np
import DGSolver
import sys
sys.path.append('../quickplotlib/lib/')
from quickplotlib import plotfxn, plot_matrix_sparsity_pattern

#NEED TO SPLIT INTO TWO DIFFERENT FILES, ONE FOR TIME ADVANCEMENT AND ONE FOR CONVERGENCE

#TIME ADVANCEMENT
'''
x_values = np.loadtxt("physical_x_values.csv", delimiter=",")
x_values_flatten = x_values.flatten()

# print("1")
# print(x_values)
# print("2")
# print(type(x_values))
# print("3")
# print(x_values[0][1])
# print("4")
# print(x_values_flatten)
# print("5")
# print(x_values_flatten[1])
# print("6")
# print(type(x_values_flatten))
# exit()

y_values_at_all_timesteps_flatten = []
for i in range(0,11):
	if(i<10):
		filename = 'physical_solution_00000' + str(i) + '.csv'
	else:
		filename = 'physical_solution_0000' + str(i) + '.csv'

	y_values_at_one_time = np.loadtxt(filename, delimiter=",")
	y_values_at_one_time_flatten = y_values_at_one_time.flatten()
	y_values_at_all_timesteps_flatten.append(y_values_at_one_time_flatten)

#print(len(x_values_flatten))
#print(len(y_values_at_all_timesteps_flatten[0]))
n_time_steps = 10
subdir = "linear_advection_snapshots"
for i in range(0,n_time_steps+1):
	fig_file_name = subdir+"/"
	if(i<10):
		fig_file_name += "0000" + str(i)
	elif(i<100):
		fig_file_name += "000" + str(i)

	current_time_string = "Time Step: " + str(i)
	plotfxn(
		[x_values_flatten,x_values_flatten], 
		[y_values_at_all_timesteps_flatten[i],y_values_at_all_timesteps_flatten[i]],
	    ylabel='$u(x,t)$',xlabel='$x$', 
	    figure_filename=fig_file_name, 
	    figure_filetype="png", 
	    title_label=current_time_string,
	    legend_labels_tex=["solution","solution"])
'''