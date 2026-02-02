import numpy as np

# Use relative import when imported as module, absolute when run directly
try:
    from .elements import Element3D
except ImportError:
    from elements import Element3D

class Wedge3D(Element3D):
  """
  Wedge (prism) linear (W1) parametric element in 3D.
  Composed of two triangular faces and three quadrilateral faces.
  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 6
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    Ni = np.zeros((self.basis_per_el),dtype=np.float64)
    # basis functions for triangle in (xi,eta) plane
    t1 = 1.0 - xi[0] - xi[1]
    t2 = xi[0]
    t3 = xi[1]
    # vertical basis functions in zeta direction
    vm = 0.5 * (1.0 - xi[2])
    vp = 0.5 * (1.0 + xi[2])
    # combine to get 3D basis functions
    Ni[0] = t1 * vm
    Ni[1] = t2 * vm
    Ni[2] = t3 * vm
    Ni[3] = t1 * vp
    Ni[4] = t2 * vp
    Ni[5] = t3 * vp
    return Ni
  
  def Ni_centroid(self):
    Ni = np.ones((self.basis_per_el), dtype=np.float64) / 6.0
    return Ni
  
  def evaluate_GNi(self, xi:np.ndarray):
    GNi = np.zeros((self.basis_per_el,3),dtype=np.float64)
    #dN0
    GNi[0,0] = -0.5 * (1.0 - xi[2])
    GNi[0,1] = -0.5 * (1.0 - xi[2])
    GNi[0,2] = -0.5 * (1.0 - xi[0] - xi[1])
    #dN1
    GNi[1,0] = 0.5 * (1.0 - xi[2])
    GNi[1,1] = 0.0
    GNi[1,2] = -0.5 * xi[0]
    #dN2
    GNi[2,0] = 0.0
    GNi[2,1] = 0.5 * (1.0 - xi[2])
    GNi[2,2] = -0.5 * xi[1]
    #dN3
    GNi[3,0] = -0.5 * (1.0 + xi[2])
    GNi[3,1] = -0.5 * (1.0 + xi[2])
    GNi[3,2] = 0.5 * (1.0 - xi[0] - xi[1])
    #dN4
    GNi[4,0] = 0.5 * (1.0 + xi[2])
    GNi[4,1] = 0.0
    GNi[4,2] = 0.5 * xi[0]
    #dN5
    GNi[5,0] = 0.0
    GNi[5,1] = 0.5 * (1.0 + xi[2])
    GNi[5,2] = 0.5 * xi[1]
    return GNi
  
  def GNi_centroid(self):
    xi  = np.array([1.0/3.0, 1.0/3.0, 0.0], dtype=np.float64)
    GNi = self.evaluate_GNi(xi)
    return GNi