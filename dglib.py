import numpy as np
from dg_variables import *

#==================================================================================#
#Integration function using LGL quadrature
#Indexing to get xL and xR of given element from x: xL=elmnt-1, xR=elmnt
#Note: be careful whether or not input functions takes standard or x space variables (use mapping when it takes x-space)

'''
#Coded for multi element#
def integrate(elmnt, jac, u1_input, u2_input): #integrate(nodes, x, weights, elmnt, u_input): -- TO DO: Modify for multi elements
    global poly_degree, number_of_quad_points
    integral = 0.0
    for i in range(number_of_quad_points):
        #using quad_points(standard space)
        integral += weights[elmnt][i]*u1_input(nodes[elmnt][i])*u2_input(nodes[elmnt][i]) 
    integral *= jac
    return integral
'''

#Coded for one element#
def integrate(jac, u1_input, u2_input):
    global poly_degree, number_of_quad_points
    integral = 0.0
    for i in range(number_of_quad_points):
        integral += weights[i]*u1_input(nodes[i])*u2_input(nodes[i]) #using quad_points(standard space)
        #integral += weights[i]*u1_input(chi(nodes[i],-3,1.5)) #using x-space
    integral *= jac #chi_jacobian(x[elmnt-1], x[elmnt])
    return integral
#===================================================================================#
#Mapping function for epsilon
def chi(quad_point, xL, xR): #change later to element, and then get xL and xR from element number
    chi = 0.5*(1-quad_point)*xL + 0.5*(1+quad_point)*xR
    return chi

#Jacobian of mapping function
def chi_jacobian(xL, xR): #again, change later to element
    chi_jacobian = 0.5*(xR - xL)
    return chi_jacobian
#===================================================================================#
#Mass matrix
def build_mass_matrix(jac, u_inputs):
    for i in range(number_of_quad_points):
        for j in range(number_of_quad_points):
            mass_matrix[i][j] = integrate(jac,u_inputs[i],u_inputs[j])

    return mass_matrix

