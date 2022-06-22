#import statements
import numpy as np
import math
from polylib import *
#=========================================================================================#

#Discontinuous Galerkin Solver

class DGSolver:

	#constructor
	def __init__(self, poly_degree=10, number_of_extra_quad_points=0, 
				 number_of_elements=5, domain_left=0.0, domain_right=5.0, a=0.5, cfl=0.5,
				 rk4=False, gauss_legendre=False):
		
		#set attributes with input parameters
		self.poly_degree = poly_degree
		self.number_of_quad_points = poly_degree+1+number_of_extra_quad_points # can vary depending on PDE nonlinearities; see Karniadakis book "aliasing section"
		self.number_of_elements = number_of_elements
		self.domain_left = domain_left
		self.domain_right = domain_right
		self.a = a #This attribute will be moved to physics class along with flux methods
		self.cfl = cfl
		self.rk4 = rk4
		self.gauss_legendre = gauss_legendre
		self.left_dirichlet_boundary_condition_physical = 0.0
		self.right_dirichlet_boundary_condition_physical = 0.0

		if(self.rk4):
			print("Using RK4")
		if(self.gauss_legendre):
			print("Using Gauss-Legendre")
		else:
			print("Using Gauss-Lobatto-Legendre")

		#initialize other attributes
		self.nodes = np.empty(self.number_of_quad_points)
		self.quadrature_weights = np.empty(self.number_of_quad_points)
		self.differentiation_matrix = np.empty((self.number_of_quad_points,self.number_of_quad_points))
		self.basis_functions_store = []
		self.test_functions_store = []
		self.derivative_test_functions_store = []
		self.basis_functions_store_left = np.empty(self.poly_degree+1)
		self.basis_functions_store_right = np.empty(self.poly_degree+1)
		self.element_vertices = np.empty(self.number_of_elements+1)
		self.elementwise_left_vertices= np.empty(self.number_of_elements)
		self.elementwise_right_vertices= np.empty(self.number_of_elements)
		self.elementwise_jacobian= np.empty(self.number_of_elements)
		self.elementwise_x = []
		self.min_delta_x = 0.0
		self.adjusted_cfl = 0.0
		self.elementwise_exact_solution_physical = [] #np.empty(self.number_of_elements)
		self.elementwise_solution_physical = [] #np.empty(self.number_of_elements) #not set in constructor
		self.elementwise_solution_frequency = [] #np.empty(self.number_of_elements) #not set in constructor 

		#set other attributes   
		self.nodes, self.quadrature_weights = gLLNodesAndWeights(self.number_of_quad_points)
		self.differentiation_matrix = gLLDifferentiationMatrix(self.number_of_quad_points)
		
		for p in range(0,self.poly_degree+1):
			basis_p_at_all_nodes = basis(p, self.nodes, self.poly_degree)
			self.basis_functions_store.append(basis_p_at_all_nodes)
			self.test_functions_store.append(basis_p_at_all_nodes) # choose tests functions same as basis functions
			self.derivative_test_functions_store.append(self.differentiation(self.test_functions_store[p]))
			basis_p_at_element_boundaries = basis(p, np.array([-1.0, 1.0]), self.poly_degree)
			self.basis_functions_store_left[p] = basis_p_at_element_boundaries[0] #basis_p_at_all_nodes[0]
			self.basis_functions_store_right[p] = basis_p_at_element_boundaries[-1] #basis_p_at_all_nodes[-1]

		self.element_vertices = self.get_element_vertices_uniformly_spaced()
		self.elementwise_left_vertices = self.element_vertices[:-1]
		self.elementwise_right_vertices = self.element_vertices[1:]

		self.elementwise_jacobian = self.mapping_function_jacobian(self.elementwise_left_vertices,self.elementwise_right_vertices)

		for e in range(0,self.number_of_elements):
			self.elementwise_x.append(self.mapping_function(self.nodes, self.elementwise_left_vertices[e], self.elementwise_right_vertices[e]))

		#determine minimum delta x
		first_element_x_values = self.elementwise_x[0]
		delta_x = first_element_x_values[1:] - first_element_x_values[0:-1]
		self.min_delta_x = np.min(delta_x)
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
		# remove element index arg
		u_delta = np.zeros(self.number_of_quad_points, dtype=np.float64)
		for p in range(0,self.poly_degree+1):
			#TODO: determine if element_index is needed for elementwise_frequency_weights
			#and does basis function need to be in physical space
			u_delta += element_frequency_solution[p]*self.basis_functions_store[p]
		return u_delta #returning numerical solution in physical space

	#Converting between physical and frequency space
	def element_frequency_to_physical_at_boundaries(self, element_index, element_frequency_solution):
		#TODO: double check that this should be number_of_quad_points
		# remove element index arg
		u_delta_at_boundaries = np.zeros(2, dtype=np.float64)
		for p in range(0,self.poly_degree+1):
			#TODO: determine if element_index is needed for elementwise_frequency_weights
			#and does basis function need to be in physical space
			u_delta_at_boundaries += element_frequency_solution[p]*np.array([self.basis_functions_store_left[p],self.basis_functions_store_right[p]])
		return u_delta_at_boundaries #returning numerical solution in physical space at element boundaries

	def element_physical_to_frequency(self, element_index, element_physical_solution):
		inner_prod_phi_udelta = np.zeros(self.poly_degree+1, dtype=np.float64)
		for p in range(0,self.poly_degree+1):
			inner_prod_phi_udelta[p] = self.inner_product_for_given_element(element_index,self.basis_functions_store[p],element_physical_solution)
		u_hat = np.linalg.solve(self.build_element_mass_matrix(element_index),inner_prod_phi_udelta)
		return u_hat #returning numerical solution in frequency space
#=========================================================================================#
	#Physical and numerical flux for linear advection

	def convective_flux(self, element_solution):
		return self.a*element_solution

	def numerical_flux(self, element_solution_left, element_solution_right):
		#TODO: check indexing here and if this is correct
		flux_at_boundary = 0.5*(self.convective_flux(element_solution_left) + self.convective_flux(element_solution_right)) - 0.5*abs(self.a)*(element_solution_right-element_solution_left)
		return flux_at_boundary

	def initial_condition_function(self, x):
		# for element_index in range(0,self.number_of_elements):
		val = np.sin(x)
		return val

	def set_initial_condition(self):
		for element_index in range(0,self.number_of_elements):
			initial_physical_state = self.initial_condition_function(self.elementwise_x[element_index])
			self.elementwise_solution_physical.append(initial_physical_state)
			self.elementwise_solution_frequency.append(self.element_physical_to_frequency(element_index, initial_physical_state))

	def exact_solution_function(self, x, time):
		return self.initial_condition_function(x-self.a*time)
#=========================================================================================#
	def interpolate_solution(self):

		return 0
#=========================================================================================#
	def compute_right_hand_side(self,current_time,sol_frequency):
		# need to allocate / declare things here
		sol_physical = []
		flux_physical = []
		flux_frequency = []
		numerical_flux = []
		rhs_vec = []
		rhs_frequency_new = []
		for element_index in range(0,self.number_of_elements):
			sol_physical.append(self.element_frequency_to_physical(element_index,sol_frequency[element_index]))
		for element_index in range(0,self.number_of_elements):
			flux_physical.append(self.convective_flux(sol_physical[element_index]))
		for element_index in range(0,self.number_of_elements):
			flux_frequency.append(self.element_physical_to_frequency(element_index,flux_physical[element_index]))
		for element_index in range(0,self.number_of_elements):
			# inner element values:
			solution_L_plus,solution_R_minus = self.element_frequency_to_physical_at_boundaries(element_index,sol_frequency[element_index])
			#solution_L_plus = sol_physical[element_index][0]
			#solution_R_minus = sol_physical[element_index][-1] #current_element_solution_physical[-1]
			# neighbour element values:
			if(element_index!=0):
				#solution_L_minus = sol_physical[element_index-1][-1]
				solution_L_minus = self.element_frequency_to_physical_at_boundaries(element_index-1,sol_frequency[element_index-1])[-1]
			else:
				# solution_L_minus = self.left_dirichlet_boundary_condition_physical
				solution_L_minus = self.exact_solution_function(self.domain_left,current_time)
			if(element_index!=(self.number_of_elements-1)):
				# solution_R_plus = sol_physical[element_index+1][0]
				solution_R_plus = self.element_frequency_to_physical_at_boundaries(element_index+1,sol_frequency[element_index+1])[0]
			else:
				# solution_R_plus = self.right_dirichlet_boundary_condition_physical
				solution_R_plus = self.exact_solution_function(self.domain_right,current_time)
			# compute numerical fluxes and store:
			f_star_L = self.numerical_flux(solution_L_minus, solution_L_plus)
			f_star_R = self.numerical_flux(solution_R_minus, solution_R_plus)
			numerical_flux.append([f_star_L,f_star_R])
		for element_index in range(0,self.number_of_elements):
			# build RHS vector
			# -- boundary terms
			right_boundary_term = (flux_physical[element_index][-1] - numerical_flux[element_index][-1]) * self.basis_functions_store_right
			left_boundary_term = (flux_physical[element_index][0] - numerical_flux[element_index][0]) * self.basis_functions_store_left
			# -- volume terms (inside element)
			stiffness_term = self.build_element_stiffness_matrix(element_index).dot(flux_frequency[element_index])
			rhs_vec.append(-stiffness_term + (right_boundary_term - left_boundary_term))
		for element_index in range(0,self.number_of_elements):
			# get weights for RHS
			rhs_frequency_new.append(np.linalg.solve(self.build_element_mass_matrix(element_index),rhs_vec[element_index]))
		return rhs_frequency_new
#=========================================================================================#
	def step_in_time(self, delta_t, current_time):
		sol_freq_current = self.elementwise_solution_frequency
		
		if(self.rk4):
			# rk4:
			k1 = self.compute_right_hand_side(current_time, sol_freq_current)
			# -- step 2
			dummy_arg =[]
			for element_index in range(0,self.number_of_elements):
				dummy_arg.append(sol_freq_current[element_index]+0.5*delta_t*k1[element_index])
			k2 = self.compute_right_hand_side(current_time+0.5*delta_t, dummy_arg)
			# -- step 3
			dummy_arg =[]
			for element_index in range(0,self.number_of_elements):
				dummy_arg.append(sol_freq_current[element_index]+0.5*delta_t*k2[element_index])
			k3 = self.compute_right_hand_side(current_time+0.5*delta_t, dummy_arg)
			# -- step 4
			dummy_arg =[]
			for element_index in range(0,self.number_of_elements):
				dummy_arg.append(sol_freq_current[element_index]+delta_t*k3[element_index])
			k4 = self.compute_right_hand_side(current_time, dummy_arg)
		else:
			# e.e.
			rhs_freq = self.compute_right_hand_side(current_time,sol_freq_current)
		
		sol_freq_new = []

		for element_index in range(0,self.number_of_elements):
			if(self.rk4):
				# rk4:
				sol_freq_new.append(sol_freq_current[element_index] + (1.0/6.0)*delta_t*(k1[element_index]+2.0*k2[element_index]+2.0*k3[element_index]+k4[element_index]))
			else:
				# e.e.
				sol_freq_new.append(sol_freq_current[element_index] + delta_t*rhs_freq[element_index])

		# update solution in attributes
		for element_index in range(0,self.number_of_elements):
			self.elementwise_solution_frequency[element_index] = sol_freq_new[element_index]
		return 0

	def file_name(self, count):

		if(count<10):
			csv_name = 'physical_solution_00000' + str(count) + '.csv'
		elif(count<100):
			csv_name = 'physical_solution_0000' + str(count) + '.csv'
		elif(count<1000):
			csv_name = 'physical_solution_000' + str(count) + '.csv'
		elif(count<10000):
			csv_name = 'physical_solution_00' + str(count) + '.csv'
		elif(count<100000):
			csv_name = 'physical_solution_0' + str(count) + '.csv'
		else:
			csv_name = 'physical_solution_' + str(count) + '.csv'
		return csv_name

	def output_elementwise_x_file(self):
			x_values = np.stack(self.elementwise_x, axis=0)
			csv_name = 'physical_x_values.csv'
			np.savetxt(csv_name, x_values, delimiter=',') #, fmt='%d')

	def output_solution_files(self,count):
		for element_index in range(0,self.number_of_elements):
			self.elementwise_solution_physical[element_index] = self.element_frequency_to_physical(element_index, self.elementwise_solution_frequency[element_index])
		physical_solution = np.stack(self.elementwise_solution_physical, axis=0)
		csv_name = self.file_name(count)
		np.savetxt(csv_name, physical_solution, delimiter=',') #, fmt='%d')

	#TODO: calculate adjusted time step so that it ends exactly at final time
	def calculate_time_step(self, final_time):
		exact_time_step_from_cfl = (self.cfl * self.min_delta_x) / self.a
		upper_value_on_number_of_timesteps = math.ceil(final_time/exact_time_step_from_cfl)
		adjusted_time_step = final_time/upper_value_on_number_of_timesteps
		self.adjusted_cfl = (self.a * adjusted_time_step) / self.min_delta_x
		print(self.cfl)
		print(self.adjusted_cfl)
		print(adjusted_time_step)
		return adjusted_time_step

	def advance_solution_time(self, get_output_files=False, get_L2_error=True):
		self.set_initial_condition()
		initial_time = 0.0
		current_time = 1.0*initial_time
		output_solution_dt_interval = 0.1
		current_desired_time_for_output = 1.0*current_time+output_solution_dt_interval
		final_time = 1.0
		constant_time_step = self.calculate_time_step(final_time) #0.001
		output_count = 0
		self.output_solution_files(output_count)

		while (current_time < final_time):
			self.step_in_time(constant_time_step,current_time)
			current_time += constant_time_step
			# convert to physical
			# write the solution physical .txt files -- make solution physical a np array matrix
			# then do np.writetxt(physical_solution)
			is_output_time = ((current_time<=current_desired_time_for_output) and (current_time+constant_time_step>current_desired_time_for_output))
			if(get_output_files and is_output_time):
				output_count += 1
				self.output_solution_files(output_count)
				current_desired_time_for_output += output_solution_dt_interval
		if(get_L2_error):
			#print(constant_time_step)
			#print(self.adjusted_cfl)
			#print(current_time)
			return self.L2_error(final_time)
		else:
			return "Did not get L2_error"
#=========================================================================================#
	#L_2 error
	def L2_error(self, final_time):
		inner_prod_sum = 0.0
		# update this: self.elementwise_exact_solution_physical from elementwise_x or whatever you call physical space
		# update this: self.elementwise_solution_physical[element_index] from the u hat at t_final
		self.elementwise_exact_solution_physical = []
		for element_index in range(0,self.number_of_elements):
			self.elementwise_exact_solution_physical.append(self.exact_solution_function(self.elementwise_x[element_index], final_time))
			self.elementwise_solution_physical[element_index] = self.element_frequency_to_physical(element_index, self.elementwise_solution_frequency[element_index])
		
		for element_index in range(0,self.number_of_elements):
			diff = self.elementwise_solution_physical[element_index] - self.elementwise_exact_solution_physical[element_index]
			inner_prod_sum += self.inner_product_for_given_element(element_index, diff, diff)
		error = np.sqrt(inner_prod_sum)
		return error

	#Function for testing converting from physical to frequency and vice versa
	#Give some function to test ex: sin(x)cos(x)
	#returns error between exact solution and numerical solution in physical space
	def test_phys_freq_space_conversion(self, func):
		for element_index in range(0,self.number_of_elements):
			self.elementwise_exact_solution_physical.append(func(self.elementwise_x[element_index]))
			self.elementwise_solution_frequency.append(self.element_physical_to_frequency(element_index, self.elementwise_exact_solution_physical[element_index]))
			self.elementwise_solution_physical.append(self.element_frequency_to_physical(element_index, self.elementwise_solution_frequency[element_index]))

		error = self.L2_error()
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