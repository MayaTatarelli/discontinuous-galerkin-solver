import numpy as np
from polylib import gLLNodesAndWeights, basis
from dg_variables import *
#==========================================================================#
#           INIT VARS -- MOVE LATER
#==========================================================================#
def initialize_additional_dg_vars():
    global poly_degree, number_of_quad_points, number_of_elements,\
       domain_left,domainR, nodes, quadrature_weights,\
       basis_functions_store, test_functions_store, element_vertices,\
       elementwise_left_vertices, elementwise_right_vertices, elementwise_jacobian, elementwise_x
    
    nodes, quadrature_weights = gLLNodesAndWeights(number_of_quad_points)

    for p in range(0,poly_degree+1):
        basis_p_at_all_nodes = basis(p, nodes, poly_degree)
        basis_functions_store.append(basis_p_at_all_nodes)
        test_functions_store.append(basis_p_at_all_nodes) # choose tests functions same as basis functions

    element_vertices = get_element_vertices_uniformly_spaced(domain_left, domain_right, number_of_elements)
    elementwise_left_vertices = element_vertices[:-1]
    elementwise_right_vertices = element_vertices[1:]
    # print(element_vertices)
    # print(elementwise_left_vertices)
    # print(elementwise_right_vertices)

    elementwise_jacobian = mapping_function_jacobian(elementwise_left_vertices,elementwise_right_vertices)

    for e in range(0,number_of_elements):
        elementwise_x.append(mapping_function(nodes, elementwise_left_vertices[e], elementwise_right_vertices[e]))
    # end of function

#==================================================================================#
#Integration function using LGL quadrature
#Indexing to get xL and xR of given element from x: xL=elmnt-1, xR=elmnt
#Note: be careful whether or not input functions takes standard or x space variables (use mapping when it takes x-space)

def get_element_vertices_uniformly_spaced(domainL, domainR, num_elements):
    return np.linspace(domainL,domainR,num=num_elements+1,dtype=np.float64)    

#Nodes are the value of epsilon between -1 and 1
#Use mapping function to convert these from standard space to x-space (quadrature_weights stay the same)
#Coded for a given element
def inner_product_for_given_element(element_index, func1, func2):
    global number_of_quad_points, quadrature_weights, elementwise_jacobian
    integral = 0.0
    for i in range(0,number_of_quad_points):
        integral += quadrature_weights[i]*func1[i]*func2[i] #using quad_points(standard space)
    integral *= elementwise_jacobian[element_index]
    return integral
#===================================================================================#
#Mapping function from standard element domain to physical domain
def mapping_function(quad_point, xL, xR): #change later to element, and then get xL and xR from element number
    mapping_function = 0.5*(1-quad_point)*xL + 0.5*(1+quad_point)*xR
    return mapping_function
#Jacobian of mapping function
def mapping_function_jacobian(xL, xR): #again, change later to element
    mapping_function_jacobian = 0.5*(xR - xL)
    return mapping_function_jacobian
#===================================================================================#
#Mass matrix
def build_element_mass_matrix(element_index,P):
    global basis_functions_store, test_functions_store

    mass_matrix = np.zeros((P+1,P+1),dtype=np.float64)
    for p in range(0,P+1):
        for q in range(0,P+1):
            mass_matrix[p][q] = inner_product_for_given_element(element_index,basis_functions_store[p],test_functions_store[q])

    return mass_matrix