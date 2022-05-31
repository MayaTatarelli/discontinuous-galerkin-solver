import numpy as np
# import numpy.polynomial.legendre as legendre
import math
import sys
sys.path.append('../quickplotlib/lib/')
from quickplotlib import plotfxn, plot_matrix_sparsity_pattern
import matplotlib.pyplot as plt

from dg_variables import *
from dglib import *
from polylib import gLLNodesAndWeights, gLLDifferentiationMatrix

global poly_degree, number_of_quad_points, number_of_elements,\
       domain_left, domain_right, nodes, quadrature_weights, differentiation_matrix, basis_functions_store,\
       test_functions_store, derivative_test_functions_store, element_vertices,\
       elementwise_left_vertices, elementwise_right_vertices, elementwise_jacobian, elementwise_x

#==========================================================================#
#Test function used for testing 'inner_product_for_given_element'
def test_func(node):
    return node**2 #1
    #return math.e**(-node**2) #2

def integral_of_test_func(node):
    return (1.0/3.0)*node**3 #1
    #return 1.742395741884422 #2

def func_returns_1(node):
    return 1
#==========================================================================#

initialize_additional_dg_vars()
'''
diff_matrix =  gLLDifferentiationMatrix(number_of_quad_points)
#print(diff_matrix)

#Plot differentiation matrix
plot_matrix_sparsity_pattern(A=diff_matrix, colour_toggle='n',cutOff=1e-3, figure_filename='test_differentiation_matrix')

exit()
'''
xi = gLLNodesAndWeights(number_of_quad_points)[0]
print(number_of_quad_points)
print(len(xi))

plotfxn([xi,xi,xi,xi,xi,xi], [basis_functions_store[0],basis_functions_store[1],basis_functions_store[2],basis_functions_store[3],basis_functions_store[4],basis_functions_store[5]],
    ylabel='$\\phi$',xlabel='$\\xi$',
    figure_filename='basis_functions_may_25', figure_filetype="pdf", title_label="Legendre Polynomials", legend_labels_tex=['$p=0$','$p=1$','$p=2$','$p=3$','$p=4$','$p=5$'],nlegendcols=3)

'''
#Test building mass matrix
mass_matrix_0 = build_element_mass_matrix(0,poly_degree)

print(mass_matrix_0)

#Plot mass matrix
plot_matrix_sparsity_pattern(A=mass_matrix_0, colour_toggle='n',cutOff=1e-4, figure_filename='test_mass_matrix_0')
'''

'''
#jac = mapping_function_jacobian(-3, 1.5)
numerical_val = integrate(1, test_func, func_returns_1)
exact_val = integral_of_test_func(nodes[-1])-integral_of_test_func(nodes[0])
err = (numerical_val-exact_val)/exact_val

#print("jacobian value: %1.9e" % jac)
print("polynomial order: %i" % poly_degree)
print("numerical value: %1.9e" % numerical_val)
print("integration relative error: %1.9e" % err)
print(nodes)
print(quadrature_weights)
'''