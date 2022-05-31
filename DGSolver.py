#import statements
import numpy as np
from polylib import *
#=========================================================================================#

#Discontinuous Galerkin Solver

class DGSolver:

	#constructor
	def __init__(self, poly_degree=10, number_of_extra_quad_points=0, 
				 number_of_elements=5, domain_left=0.0, domain_right=5.0):
		
		#set attributes with input parameters
		self.poly_degree = poly_degree
		self.number_of_quad_points = poly_degree+1+number_of_extra_quad_points # can vary depending on PDE nonlinearities; see Karniadakis book "aliasing section"
		self.number_of_elements = number_of_elements
		self.domain_left = domain_left
		self.domain_right = domain_right

		#initialize other attributes
		self.nodes = np.empty(self.number_of_quad_points)
		self.quadrature_weights = np.empty(self.number_of_quad_points)
		self.differentiation_matrix = np.empty((self.number_of_quad_points,self.number_of_quad_points))
		self.basis_functions_store = []
		self.test_functions_store = []
		self.derivative_test_functions_store = []
		self.element_vertices = np.empty(self.number_of_elements+1)
		self.elementwise_left_vertices= np.empty(self.number_of_elements)
		self.elementwise_right_vertices= np.empty(self.number_of_elements)
		self.elementwise_jacobian= np.empty(self.number_of_elements)
		self.elementwise_x = []
		self.elementwise_exact_solution_physical = np.empty(self.number_of_elements)
		self.elementwise_solution_physical = np.empty(self.number_of_elements) #not set in constructor
		self.elementwise_solution_frequency = np.empty(self.number_of_elements) #not set in constructor 

		#set other attributes   
		self.nodes, self.quadrature_weights = gLLNodesAndWeights(self.number_of_quad_points)
		self.differentiation_matrix = gLLDifferentiationMatrix(self.number_of_quad_points)

		for p in range(0,self.poly_degree+1):
			basis_p_at_all_nodes = basis(p, self.nodes, self.poly_degree)
			self.basis_functions_store.append(basis_p_at_all_nodes)
			self.test_functions_store.append(basis_p_at_all_nodes) # choose tests functions same as basis functions
			self.derivative_test_functions_store.append(self.differentiation(self.test_functions_store[p]))

		self.element_vertices = self.get_element_vertices_uniformly_spaced()
		self.elementwise_left_vertices = self.element_vertices[:-1]
		self.elementwise_right_vertices = self.element_vertices[1:]

		self.elementwise_jacobian = self.mapping_function_jacobian(self.elementwise_left_vertices,self.elementwise_right_vertices)

		for e in range(0,self.number_of_elements):
			self.elementwise_x.append(self.mapping_function(self.nodes, self.elementwise_left_vertices[e], self.elementwise_right_vertices[e]))
#=========================================================================================#
	#private: Get vertices of space discretization
	def get_element_vertices_uniformly_spaced(self):
	    return np.linspace(self.domain_left,self.domain_right,num=self.number_of_elements+1,dtype=np.float64)    

	#private: Calculate inner product of base and test functions for a given element
	def inner_product_for_given_element(self, element_index, func1, func2):
	    integral = 0.0
	    for i in range(0,self.number_of_quad_points):
	        integral += self.quadrature_weights[i]*func1[i]*func2[i] #using quad_points(standard space)
	    integral *= self.elementwise_jacobian[element_index]
	    return integral

	#private: Calculate deivative of test function
	def differentiation(self, func):
	    derivative = np.empty(self.number_of_quad_points)
	    for i in range (0, self.number_of_quad_points):
	        derivative[i] = 0.0
	        for j in range(0,self.number_of_quad_points):
	            derivative[i] += self.differentiation_matrix[i][j]*func[j]
	    return derivative
#=========================================================================================#
	#private: Mapping function from standard element domain to physical domain
	def mapping_function(self, quad_point, xL, xR):
	    mapping_function = 0.5*(1-quad_point)*xL + 0.5*(1+quad_point)*xR
	    return mapping_function
	#private: Jacobian of mapping function
	def mapping_function_jacobian(self, xL, xR):
	    mapping_function_jacobian = 0.5*(xR - xL)
	    return mapping_function_jacobian
#=========================================================================================#
	#Mass matrix
	def build_element_mass_matrix(self, element_index):
	    mass_matrix = np.zeros((self.poly_degree+1,self.poly_degree+1),dtype=np.float64)
	    for p in range(0,self.poly_degree+1):
	        for q in range(0,self.poly_degree+1):
	            mass_matrix[p][q] = self.inner_product_for_given_element(element_index,self.basis_functions_store[p],self.test_functions_store[q])

	    return mass_matrix
#=========================================================================================#
	#Stiffness matrix
	def build_element_stiffness_matrix(self, element_index):
	    stiffness_matrix = np.zeros((self.poly_degree+1,self.poly_degree+1),dtype=np.float64)
	    for p in range(0,self.poly_degree+1):
	        for q in range(0,self.poly_degree+1):
	            dphi_dx = self.derivative_test_functions_store[q]/self.elementwise_jacobian[element_index]
	            stiffness_matrix[p][q] = self.inner_product_for_given_element(element_index,self.basis_functions_store[p],dphi_dx)

	    return stiffness_matrix
#=========================================================================================#
	#Converting between physical and frequency space
	def element_frequency_to_physical(self, element_index, element_frequency_solution):
		#TODO: double check that this should be number_of_quad_points
		u_delta = np.zeros(self.number_of_quad_points, dtype=np.float64)
		for p in range(0,self.poly_degree+1):
			#TODO: determine if element_index is needed for elementwise_frequency_weights
			#and does basis function need to be in physical space
			u_delta += element_frequency_solution[p]*self.basis_functions_store[p]
		return u_delta #returning numerical solution in physical space

	def element_physical_to_frequency(self, element_index, element_physical_solution):
		inner_prod_phi_udelta = np.zeros(self.poly_degree, dtype=np.float64)
		for p in range(0,self.poly_degree+1):
			#does anything need to be done to u_delta since it is a function of chi(xi)
			inner_prod_phi_udelta[p] = self.inner_product_for_given_element(element_index,self.basis_functions_store[p],element_physical_solution)

		u_hat = np.linalg.solve(self.build_element_mass_matrix(element_index),np.transpose(inner_prod_phi_udelta))
		return u_hat #returning numerical solution in frequency space
#=========================================================================================#
	#L_2 error
	def L2_error(self, solution_difference):
		inner_prod_sum = 0.0
		for element_index in range(0,self.number_of_elements):
			inner_prod_sum += self.inner_product_for_given_element(element_index, solution_difference, solution_difference)
		error = np.sqrt(inner_prod_sum)
		return error

	#Function for testing converting from physical to frequency and vice versa
	#Give some function to test ex: sin(x)cos(x)
	#returns error between exact solution and numerical solution in physical space
	def test_phys_freq_space_conversion(self, func):
		for element_index in range(0,self.number_of_elements):
			self.elementwise_exact_solution_physical[element_index] = func(self.elementwise_x[element_index])
			self.elementwise_solution_frequency[element_index] = self.element_physical_to_frequency(element_index, self.elementwise_exact_solution_physical[element_index])
			self.elementwise_solution_physical[element_index] = self.element_frequency_to_physical(element_index, self.elementwise_solution_frequency[element_index])

		diff_numerical_and_exact_solution = self.elementwise_solution_physical - self.elementwise_exact_solution_physical
		error = self.L2_error(diff_numerical_and_exact_solution)
		return error


'''
class Template:

	class_data = 3 #this is a class variable (kind of like static)

	#constructor
	def __init__(self, other_data1, other_data2):
		#these are instance variables
		self.other_data1 = other_data1 
		self.other_data2 = other_data2


	#getters and setters
	def get_other_data1(self):
		return self.other_data1

	def set_other_data1(self,value):
		self.other_data1 = value
	
'''