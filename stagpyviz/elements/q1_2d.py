import numpy as np

# Use relative import when imported as module, absolute when run directly
try:
    from .elements import Element2D
except ImportError:
    from elements import Element2D

class Q1_2D(Element2D):
  """
  Quadrilateral bilinear (Q1) parametric element in 2D.
  Nodes are ordered as:
  
    2 ---- 3
    |      |
    |      |
    0 ---- 1

  and the reference coordinates (xi, eta) range from -1 to 1.
  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 4
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    Ni = np.zeros((self.basis_per_el),dtype=np.float64)
    Ni[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
    Ni[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
    Ni[2] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])
    Ni[3] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
    return Ni
  
  def Ni_centroid(self):
    Ni = np.ones((self.basis_per_el), dtype=np.float64) * 0.25
    return Ni
  
  def evaluate_GNi(self, xi:np.ndarray):
    GNi = np.zeros((self.basis_per_el,2),dtype=np.float64)
    GNi[0,0] = -0.25 * (1.0 - xi[1]) # dN0/dxi
    GNi[0,1] = -0.25 * (1.0 - xi[0]) # dN0/deta
    GNi[1,0] =  0.25 * (1.0 - xi[1]) # dN1/dxi
    GNi[1,1] = -0.25 * (1.0 + xi[0]) # dN1/deta
    GNi[2,0] = -0.25 * (1.0 + xi[1]) # dN2/dxi
    GNi[2,1] =  0.25 * (1.0 - xi[0]) # dN2/deta
    GNi[3,0] =  0.25 * (1.0 + xi[1]) # dN3/dxi
    GNi[3,1] =  0.25 * (1.0 + xi[0]) # dN3/deta
    return GNi
  
  def GNi_centroid(self):
    """
    Returns the shape function derivatives at the element centroid (xi=0, eta=0).
    GNi[i, d] = dN_i/dxi_d  where i=node, d=direction (0=xi, 1=eta)

    Returns
    -------
    np.ndarray
        Array of shape (4, 2) with shape function derivatives at centroid.
    """
    GNi = np.array([
      [-0.25, -0.25],  # node 0
      [ 0.25, -0.25],  # node 1
      [-0.25,  0.25],  # node 2
      [ 0.25,  0.25],  # node 3
    ], dtype=np.float64)
    return GNi