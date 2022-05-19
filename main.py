import numpy as np
# import numpy.polynomial.legendre as legendre
import math
import sys
sys.path.append('../quickplotlib/lib/')
from quickplotlib import plotfxn

from dg_variables import *
from dglib import integrate
from dglib import chi_jacobian

#==========================================================================#
#Test function used for testing 'integrate'
def test_func(node):
    return node**2 #1
    #return math.e**(-node**2) #2

def integral_of_test_func(node):
    return (1.0/3.0)*node**3 #1
    #return 1.742395741884422 #2

def func_returns_1(node):
    return 1
#==========================================================================#

#jac = chi_jacobian(-3, 1.5)
numerical_val = integrate(1, test_func, func_returns_1)
exact_val = integral_of_test_func(nodes[-1])-integral_of_test_func(nodes[0])
err = (numerical_val-exact_val)/exact_val

#print("jacobian value: %1.9e" % jac)
print("polynomial order: %i" % poly_degree)
print("numerical value: %1.9e" % numerical_val)
print("integration relative error: %1.9e" % err)
print(nodes)
print(weights)