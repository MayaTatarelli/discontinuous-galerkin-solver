#Test DGSolver class
import numpy as np
import DGSolver
import sys
sys.path.append('../quickplotlib/lib/')
from quickplotlib import plotfxn, plot_matrix_sparsity_pattern

#=========================================================================#
#New instance of DGSolver with default values
#dg_solver_1 = DGSolver.DGSolver()

'''
#Test Lagrange basis functions
plotfxn([dg_solver_1.nodes,dg_solver_1.nodes,dg_solver_1.nodes,dg_solver_1.nodes,dg_solver_1.nodes,dg_solver_1.nodes], 
		[dg_solver_1.basis_functions_store[0],dg_solver_1.basis_functions_store[1],dg_solver_1.basis_functions_store[2],dg_solver_1.basis_functions_store[3],dg_solver_1.basis_functions_store[4],dg_solver_1.basis_functions_store[5]],
    ylabel='$\\phi$',xlabel='$\\xi$',
    figure_filename='basis_functions_may_26', figure_filetype="pdf", title_label="Lagrange Polynomials", legend_labels_tex=['$p=0$','$p=1$','$p=2$','$p=3$','$p=4$','$p=5$'],nlegendcols=3)

#Test building mass matrix
mass_matrix_0 = dg_solver_1.build_element_mass_matrix(0)
#Plot mass matrix
plot_matrix_sparsity_pattern(A=mass_matrix_0, colour_toggle='n',cutOff=1e-4, figure_filename='test_DGSolver_class_mass_matrix_0')

#Test differentiation matrix
diff_matrix = dg_solver_1.differentiation_matrix
#Plot differentiation matrix
plot_matrix_sparsity_pattern(A=diff_matrix, colour_toggle='n',cutOff=1e-3, figure_filename='test_DGSolver_class_differentiation_matrix')

#Test building stiffness matrix
stiff_matrix_0 = dg_solver_1.build_element_stiffness_matrix(0)
plot_matrix_sparsity_pattern(A=stiff_matrix_0, colour_toggle='n',cutOff=1e-4, figure_filename='test_DGSolver_class_stiff_matrix_0')
'''
#=========================================================================#
#Another instance of DGSolver but with number_of_extra_quad_points set to 50
#dg_solver_2 = DGSolver.DGSolver(number_of_extra_quad_points=50)
'''
#Test building mass matrix
mass_matrix_0_2 = dg_solver_2.build_element_mass_matrix(0)
#Plot mass matrix
plot_matrix_sparsity_pattern(A=mass_matrix_0_2, colour_toggle='n',cutOff=1e-4, figure_filename='test_2_DGSolver_class_mass_matrix_0')

#Test differentiation matrix
diff_matrix_2 = dg_solver_2.differentiation_matrix
#Plot differentiation matrix
plot_matrix_sparsity_pattern(A=diff_matrix_2, colour_toggle='n',cutOff=1e-3, figure_filename='test_2_DGSolver_class_differentiation_matrix')

#Test building stiffness matrix
stiff_matrix_3_2 = dg_solver_2.build_element_stiffness_matrix(3)
plot_matrix_sparsity_pattern(A=stiff_matrix_3_2, colour_toggle='n',cutOff=1e-4, figure_filename='test_2_DGSolver_class_stiff_matrix_3')
'''
#=========================================================================#
'''
#Testing converting between physical and frequency space

def test_func(x):
	return np.sin(x)*np.cos(x)

dg_solver = DGSolver.DGSolver()
error = dg_solver.test_phys_freq_space_conversion(test_func)
print(error)
'''
#=========================================================================#
#Testing time advancement function

dg_solver = DGSolver.DGSolver(domain_left=0.0,
							  domain_right=(2.0*np.pi),
							  poly_degree=2,
							  number_of_elements=15,
							  cfl=0.005,
							  a=(2.0*np.pi))
#dg_solver.set_initial_condition()
#dg_solver.output_elementwise_x_file()
L2_error = dg_solver.advance_solution_time()
print(L2_error)
