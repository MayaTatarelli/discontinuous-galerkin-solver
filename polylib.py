import numpy as np
from scipy.special import jacobi
#*****************************************************
#           POLYNOMIAL LIBRARY SUBROUTINES
#*****************************************************
#=====================================================
def basis(p, xi, P):
  # Lagrange basis functions
  xi_p = 1.0*xi[p]
  value_store = np.zeros(P+1)
  for i in range(0,P+1):
    # if (xi[i] == xi_p):
    #   value[i] = 1.0
    # else:
    #   value[i] = (xi[i]-1.0)*(xi[i]+1.0)*jacobi(P-1, 1, 1, monic=False)(xi[i])/((P+1)*P*lgP(P, xi_p)*(xi_p-xi[i]))
    #   value[i] = (xi[i]-1.0)*(xi[i]+1.0)*dLgP(P, xi[i])/((P+1)*P*lgP(P, xi_p)*(xi_p-xi[i]))
    value = 1.0
    for j in range(0,P+1):
      if j != p:
        value *= (xi[i]-xi[j])/(xi_p-xi[j])
    value_store[i] = value

  return value_store
#=====================================================
'''def basis(p, xi, P):
  # Legendre basis functions
  # ref: https://backend.orbit.dtu.dk/ws/portalfiles/portal/143813931/filestore_40_.pdf
  # -- equation 4, page 4
  if(p==0):
    value = 0.5*(1.0-xi)
  elif(p==P):
    value = 0.5*(1.0+xi)
  else:
    value = 0.25*(1.0+xi)*(1.0-xi)*jacobi(p-1, 1, 1, monic=False)(xi) # may need to set this as true, you'll see
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jacobi.html
  return value'''
#=====================================================
def lgP (n, xi):
  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
  """
  Evaluates P_{n}(xi) using an iterative algorithm
  """
  if n == 0:
    
    return np.ones (xi.size)
  
  elif n == 1:
    
    return xi

  else:

    fP = np.ones (xi.size); sP = xi.copy (); nP = np.empty (xi.size)

    for i in range (2, n + 1):

      nP = ((2 * i - 1) * xi * sP - (i - 1) * fP) / i

      fP = sP; sP = nP

    return nP
#=====================================================
def dLgP (n, xi):
  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
  """
  Evaluates the first derivative of P_{n}(xi)
  """
  return n * (lgP (n - 1, xi) - xi * lgP (n, xi))\
           / (1 - xi ** 2)
#=====================================================
def d2LgP (n, xi):
  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
  """
  Evaluates the second derivative of P_{n}(xi)
  """
  return (2 * xi * dLgP (n, xi) - n * (n + 1)\
                                    * lgP (n, xi)) / (1 - xi ** 2)
#=====================================================
def d3LgP (n, xi):
  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
  """
  Evaluates the third derivative of P_{n}(xi)
  """
  return (4 * xi * d2LgP (n, xi)\
                 - (n * (n + 1) - 2) * dLgP (n, xi)) / (1 - xi ** 2)
#=====================================================
def gLLNodesAndWeights (n, epsilon = 1e-15):
  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
  """
  Computes the GLL nodes and weights
  """
  if n < 2:
    
    print ('Error: n must be larger than 1')
  
  else:
    
    x = np.empty (n)
    w = np.empty (n)
    
    x[0] = -1; x[n - 1] = 1
    w[0] = w[0] = 2.0 / ((n * (n - 1))); w[n - 1] = w[0];
    
    n_2 = n // 2
    
    for i in range (1, n_2):
      
      xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) *\
           np.cos ((4 * i + 1) * np.pi / (4 * (n - 1) + 1))
      
      error = 1.0
      
      while error > epsilon:
        
        y  =  dLgP (n - 1, xi)
        y1 = d2LgP (n - 1, xi)
        y2 = d3LgP (n - 1, xi)
        
        dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)
        
        xi -= dx
        error = abs (dx)
      
      x[i] = -xi
      x[n - i - 1] =  xi
      
      w[i] = 2 / (n * (n - 1) * lgP (n - 1, x[i]) ** 2)
      w[n - i - 1] = w[i]

    if n % 2 != 0:

      x[n_2] = 0;
      w[n_2] = 2.0 / ((n * (n - 1)) * lgP (n - 1, np.array (x[n_2])) ** 2)
      
  return x, w
#=====================================================
def gLLDifferentiationMatrix (n, epsilon = 1e-15):
  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
  """
  Computes the GLL nodes and weights
  """
  if n < 2:
    
    print ('Error: n must be larger than 1')
  
  else:
    n_p = n-1
    xi = gLLNodesAndWeights(n_p+1, epsilon)[0]
    d = np.empty((n_p+1,n_p+1))
    
    for i in range (0, n_p+1):
      for j in range (0, n_p+1):
        if(i==0 and j==0):
          d[i][j] = -n_p*(n_p+1)/4.0
        elif(i==n_p and j==n_p):
          d[i][j] = n_p*(n_p+1)/4.0
        elif(i==j and i>=1 and i<=(n_p-1)):
          d[i][j] = 0.0
        else:
          d[i][j] = 1.0/(xi[i]-xi[j])*(lgP(n_p, xi[i])/lgP(n_p, xi[j])) 
      
  return d
#=====================================================
def gLNodesAndWeights(n):
  if(n == 3):
    val1 = np.sqrt(3.0/5.0)
    x = np.array([-val1, 0.0, val1])

    weight1 = 5.0/9.0
    w = np.array([weight1, 8.0/9.0, weight1])

  elif(n == 4):
    val1 = np.sqrt(3.0/7.0 + np.sqrt(6.0/5.0)*2.0/7.0)
    val2 = np.sqrt(3.0/7.0 - np.sqrt(6.0/5.0)*2.0/7.0)
    x = np.array([-val1, -val2, val2, val1])

    weight1 = (18.0 - np.sqrt(30.0)) / 36.0
    weight2 = (18.0 + np.sqrt(30.0)) / 36.0
    w = np.array([weight1, weight2, weight2, weight1])

  elif(n == 5):
    val1 = (1.0/3.0)*np.sqrt(5.0 + 2.0*np.sqrt(10.0/7.0))
    val2 = (1.0/3.0)*np.sqrt(5.0 - 2.0*np.sqrt(10.0/7.0))
    x = np.array([-val1, -val2, 0.0, val2, val1])

    weight1 = (322.0 - 13.0*np.sqrt(70.0)) / 900.0
    weight2 = (322.0 + 13.0*np.sqrt(70.0)) / 900.0
    w = np.array([weight1, weight2, 128.0/225.0, weight2, weight1])
  else:
    print("You are using too high a polynomial degree for Gauss-Legendre quadrature.")
    exit()
  return x, w