from polylib import gLLNodesAndWeights

poly_degree = 9
number_of_quad_points = poly_degree+1

#Nodes are the value of epsilon between -1 and 1
#Use mapping function to convert these from standard space to x-space (weights stay the same)
nodes, weights = gLLNodesAndWeights(number_of_quad_points)

'''
#x-space variables
x = np.empty(number_of_quad_points)
for i in range(number_of_quad_points):
    x[i] = chi(nodes[i],)
'''
