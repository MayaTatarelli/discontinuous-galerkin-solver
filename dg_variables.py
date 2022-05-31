import numpy as np
global poly_degree, number_of_quad_points, number_of_elements,\
       domain_left, domain_right, nodes, quadrature_weights, differentiation_matrix, basis_functions_store,\
       test_functions_store, derivative_test_functions_store, element_vertices,\
       elementwise_left_vertices, elementwise_right_vertices, elementwise_jacobian, elementwise_x

poly_degree = 10
number_of_quad_points = poly_degree+1 # can vary depending on PDE nonlinearities; see Karniadakis book "aliasing section"
number_of_elements = 5
domain_left = 0.0
domain_right = 5.0

#To be initialized outside
nodes = np.empty(number_of_quad_points)
quadrature_weights = np.empty(number_of_quad_points)
# differentiation_matrix = np.empty((number_of_quad_points,number_of_quad_points))
basis_functions_store = []
test_functions_store = []
derivative_test_functions_store = []
element_vertices = np.empty(number_of_elements+1)
elementwise_left_vertices= np.empty(number_of_elements)
elementwise_right_vertices= np.empty(number_of_elements)
elementwise_jacobian= np.empty(number_of_elements)
elementwise_x = []