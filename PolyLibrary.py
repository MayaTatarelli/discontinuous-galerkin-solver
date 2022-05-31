class PolyLibrary:

	def __init__(self, n, xi):
		self.n = n #number of quadrature points
		self.xi = xi #nodes for quadrature

#=====================================================
	def basis(self, p, P):
	  # Legendre basis functions
	  # ref: https://backend.orbit.dtu.dk/ws/portalfiles/portal/143813931/filestore_40_.pdf
	  # -- equation 4, page 4
	  if(p==0):
	    value = 0.5*(1.0-self.xi)
	  elif(p==P):
	    value = 0.5*(1.0+self.xi)
	  else:
	    value = 0.25*(1.0+self.xi)*(1.0-self.xi)*jacobi(p-1, 1, 1, monic=False)(self.xi) # may need to set this as true, you'll see
	    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jacobi.html
	  return value
#=====================================================
	def lgP (self.n, self.xi):
	  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
	  """
	  Evaluates P_{self.n}(self.xi) using an iterative algorithm
	  """
	  if self.n == 0:
	    
	    return np.ones (self.xi.size)
	  
	  elif self.n == 1:
	    
	    return self.xi

	  else:

	    fP = np.ones (self.xi.size); sP = self.xi.copy (); nP = np.empty (self.xi.size)

	    for i in range (2, self.n + 1):

	      nP = ((2 * i - 1) * self.xi * sP - (i - 1) * fP) / i

	      fP = sP; sP = nP

	    return nP
#=====================================================
	def dLgP (self.n, self.xi):
	  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
	  """
	  Evaluates the first derivative of P_{self.n}(self.xi)
	  """
	  return self.n * (lgP (self.n - 1, self.xi) - self.xi * lgP (self.n, self.xi))\
	           / (1 - self.xi ** 2)
#=====================================================
	def d2LgP (self.n, self.xi):
	  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
	  """
	  Evaluates the second derivative of P_{self.n}(self.xi)
	  """
	  return (2 * self.xi * dLgP (self.n, self.xi) - self.n * (self.n + 1)\
	                                    * lgP (self.n, self.xi)) / (1 - self.xi ** 2)
#=====================================================
	def d3LgP (self.n, self.xi):
	  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
	  """
	  Evaluates the third derivative of P_{self.n}(self.xi)
	  """
	  return (4 * self.xi * d2LgP (self.n, self.xi)\
	                 - (self.n * (self.n + 1) - 2) * dLgP (self.n, self.xi)) / (1 - self.xi ** 2)
#=====================================================
	def gLLNodesAndWeights (self.n, epsilon = 1e-15):
	  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
	  """
	  Computes the GLL nodes and weights
	  """
	  if self.n < 2:
	    
	    print ('Error: self.n must be larger than 1')
	  
	  else:
	    
	    x = np.empty (self.n)
	    w = np.empty (self.n)
	    
	    x[0] = -1; x[self.n - 1] = 1
	    w[0] = w[0] = 2.0 / ((self.n * (self.n - 1))); w[self.n - 1] = w[0];
	    
	    n_2 = self.n // 2
	    
	    for i in range (1, n_2):
	      
	      self.xi = (1 - (3 * (self.n - 2)) / (8 * (self.n - 1) ** 3)) *\
	           np.cos ((4 * i + 1) * np.pi / (4 * (self.n - 1) + 1))
	      
	      error = 1.0
	      
	      while error > epsilon:
	        
	        y  =  dLgP (self.n - 1, self.xi)
	        y1 = d2LgP (self.n - 1, self.xi)
	        y2 = d3LgP (self.n - 1, self.xi)
	        
	        dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)
	        
	        self.xi -= dx
	        error = abs (dx)
	      
	      x[i] = -self.xi
	      x[self.n - i - 1] =  self.xi
	      
	      w[i] = 2 / (self.n * (self.n - 1) * lgP (self.n - 1, x[i]) ** 2)
	      w[self.n - i - 1] = w[i]

	    if self.n % 2 != 0:

	      x[n_2] = 0;
	      w[n_2] = 2.0 / ((self.n * (self.n - 1)) * lgP (self.n - 1, np.array (x[n_2])) ** 2)
	      
	  return x, w
#=====================================================
	def gLLDifferentiationMatrix (self.n, epsilon = 1e-15):
	  # Reference for : https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb#scrollTo=uesH7LNBkhaP
	  """
	  Computes the GLL nodes and weights
	  """
	  if self.n < 2:
	    
	    print ('Error: self.n must be larger than 1')
	  
	  else:
	    n_p = self.n-1
	    self.xi = gLLNodesAndWeights(n_p+1, epsilon)[0]
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
	          d[i][j] = 1.0/(self.xi[i]-self.xi[j])*(lgP(n_p, self.xi[i])/lgP(n_p, self.xi[j])) 
	      
	  return d
#=====================================================