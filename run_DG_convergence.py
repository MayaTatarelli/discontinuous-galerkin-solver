import numpy as np
import DGSolver
import sys
sys.path.append('../quickplotlib/lib/')
from quickplotlib import plotfxn, plot_matrix_sparsity_pattern
import matplotlib . pyplot as plt

number_of_runs = 20
num_elements_values = [2]

for i in range(1,number_of_runs):
	num_elements_values.append(num_elements_values[i-1]+2)
num_elements_values = np.array(num_elements_values)

# print(len(num_elements_values))
# print(num_elements_values)


error_values = np.zeros(number_of_runs)

for iElements in range(len(num_elements_values)):
	print("Running dg_solver with " + str(num_elements_values[iElements]) + " elements;")
	dg_solver = DGSolver.DGSolver(domain_left=0.0,
								  domain_right=(2.0*np.pi),
								  poly_degree=2,
								  number_of_elements=num_elements_values[iElements],
								  cfl=0.005,
								  a=(2.0*np.pi))
	error_values[iElements] = dg_solver.advance_solution_time()
	print("Error: " + str(error_values[iElements]))

print(error_values)
np.savetxt("error_values_lower_num_elmnts_2.csv", error_values, delimiter=',') #, fmt='%d')

plt.figure()
plt.plot(num_elements_values, error_values)
plt.show()

'''
#CONVERGENCE
x_values = (2.0*np.pi)/num_elements_values
print(x_values)
error_values = np.loadtxt("error_values_lower_num_elmnts_2.csv", delimiter=",")
print(error_values)
poly_degree = 2

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

plt.figure()
i = poly_degree+1
dx_n = get_ref_curve(error_values, x_values, -i)
label_name = "$\\Delta x^{-%i}$" % i
plt.loglog(1.0/x_values, 1.0/dx_n,linestyle="--",label=label_name)

plt.loglog(1.0/x_values, error_values)#,label=name_scheme, linestyle=linestyles[i_scheme])

plt.legend()
plt.show()
'''

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
