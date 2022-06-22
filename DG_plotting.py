import numpy as np
import DGSolver
import sys
sys.path.append('../quickplotlib/lib/')
from quickplotlib import plotfxn, plot_matrix_sparsity_pattern
import matplotlib . pyplot as plt

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

#CONVERGENCE

def get_ref_curve(error_vals,x_data,n):
    # n is order
    index_align = -1
    y_align = error_vals[index_align]
    x_align = x_data[index_align]
	# n = -(P+1)
    shift = np.log10(1.0/(y_align*((x_align*10.0)**float(n))))
	# Confirm the order of accuracy
	# shift = [-1.0,-2.0]
	# for n in range(1,3): 
	# 	name = r"$h^{-%i}$" % n
	# 	dx_n = (dx_store**n)*10.0**(shift[n-1]+float(n))
	# 	plt.loglog(1.0/dx_store,dx_n,label=name,color=clr[n+2],linewidth=1.0,linestyle='--')
    dx_n = (x_data**n)*10.0**(shift+float(n))
    return dx_n

number_of_runs = 10
num_elements_values = [5]

for i in range(1,number_of_runs):
	num_elements_values.append(num_elements_values[i-1]+10)
num_elements_values = np.array(num_elements_values)

x_values = (2.0*np.pi)/num_elements_values
print(x_values)
error_values = np.loadtxt("error_values_P1_cfl0075_rk4.csv", delimiter=",")
#When using error_values below, number_of_runs must be 8 (up to 75 elements only)
#error_values = np.array ([0.07675223434418506, 0.003379224280853759, 0.001316652195210326, 0.0008751557041560119, 0.0008060444479951529, 0.0006575629672622212, 0.000555763935542056, 0.00048141199163347006])
print(error_values)
poly_degree = 1

plt.figure()
i = poly_degree+1
dx_n = get_ref_curve(error_values, x_values, -i)
label_name = "$\\Delta x^{-(P+1)}$"
plt.loglog(1.0/x_values, 1.0/dx_n,linestyle="--",label=label_name)

plt.loglog(1.0/x_values, error_values)#,label=name_scheme, linestyle=linestyles[i_scheme])

plt.title("Nodal, P=1, CFL=0.0015, $N_{el}\\epsilon$[5,95]")
plt.ylabel("L2-error")
plt.xlabel("$h^{-1}$")
plt.legend()
plt.show()


'''
num_to_remove = 10
plt.figure()
i = poly_degree+1
dx_n = get_ref_curve(error_values[:-num_to_remove], x_values[:-num_to_remove], -i)
label_name = "$\\Delta x^{-%i}$" % i
plt.loglog(1.0/x_values[:-num_to_remove], 1.0/dx_n,linestyle="--",label=label_name)

plt.loglog(1.0/x_values[:-num_to_remove], error_values[:-num_to_remove])#,label=name_scheme, linestyle=linestyles[i_scheme])

plt.legend()
plt.show()
'''
'''
high_num_elements_values = [5]
for i in range(1,10):
	high_num_elements_values.append(high_num_elements_values[i-1]+10)
high_num_elements_values = np.array(high_num_elements_values)
print(high_num_elements_values)

high_x_values = (2.0*np.pi)/high_num_elements_values
high_error_values = np.loadtxt("error_values.csv", delimiter=",")
poly_degree = 2

plt.figure()
i = poly_degree+1
high_dx_n = get_ref_curve(high_error_values, high_x_values, -i)
label_name = "$\\Delta x^{-%i}$" % i
plt.loglog(1.0/high_x_values, 1.0/high_dx_n,linestyle="--",label=label_name)

plt.loglog(1.0/high_x_values, high_error_values)#,label=name_scheme, linestyle=linestyles[i_scheme])

plt.legend()
plt.show()
'''