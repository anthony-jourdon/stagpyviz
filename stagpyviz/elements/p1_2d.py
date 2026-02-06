import numpy as np

# Use relative import when imported as module, absolute when run directly
try:
    from .elements import Element2D, SurfaceElement
except ImportError:
    from elements import Element2D, SurfaceElement

class P1_2D(Element2D):
  """
  Triangular linear (P1) parametric element in 2D.
  Nodes are ordered as:
  
    2
    | \
    |  \
    0---1

  and the reference coordinates (xi, eta) range from 0 to 1 with the constraint
  xi + eta <= 1.
  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 3
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    Ni = np.zeros((self.basis_per_el),dtype=np.float64)
    Ni[0] = 1.0 - xi[0] - xi[1]
    Ni[1] = xi[0]
    Ni[2] = xi[1]
    return Ni
  
  def Ni_centroid(self):
    Ni = np.ones((self.basis_per_el), dtype=np.float64) / 3.0
    return Ni
  
  def evaluate_GNi(self):
    GNi = np.zeros((self.basis_per_el,2),dtype=np.float64)
    GNi[0,0] = -1.0 # dN0/dxi
    GNi[0,1] = -1.0 # dN0/deta
    GNi[1,0] =  1.0 # dN1/dxi
    GNi[1,1] =  0.0 # dN1/deta
    GNi[2,0] =  0.0 # dN2/dxi
    GNi[2,1] =  1.0 # dN2/deta
    return GNi
  
  def GNi_centroid(self):
    """
    Returns the shape function derivatives at the element centroid (xi=1/3, eta=1/3).
    GNi[i, d] = dN_i/dxi_d  where i=node, d=direction (0=xi, 1=eta)

    Returns
    -------
    np.ndarray
        Array of shape (3, 2) with shape function derivatives at centroid.
    """
    GNi = np.array([
      [-1.0, -1.0],
      [ 1.0,  0.0],
      [ 0.0,  1.0]
    ], dtype=np.float64)
    return GNi
  
class P1_2D_R3(SurfaceElement):
  """
  Triangular linear (P1) parametric element in 3D, representing a surface element.
  Nodes are ordered as:
  
    2
    | \
    |  \
    0---1

  and the reference coordinates (xi, eta) range from 0 to 1 with the constraint
  xi + eta <= 1.
  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 3
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    Ni = np.zeros((self.basis_per_el),dtype=np.float64)
    Ni[0] = 1.0 - xi[0] - xi[1]
    Ni[1] = xi[0]
    Ni[2] = xi[1]
    return Ni
  
  def Ni_centroid(self):
    Ni = np.ones((self.basis_per_el), dtype=np.float64) / 3.0
    return Ni
  
  def evaluate_GNi(self):
    GNi = np.zeros((self.basis_per_el,2),dtype=np.float64)
    GNi[0,0] = -1.0 # dN0/dxi
    GNi[0,1] = -1.0 # dN0/deta
    GNi[1,0] =  1.0 # dN1/dxi
    GNi[1,1] =  0.0 # dN1/deta
    GNi[2,0] =  0.0 # dN2/dxi
    GNi[2,1] =  1.0 # dN2/deta
    return GNi
  
  def GNi_centroid(self):
    return self.evaluate_GNi()