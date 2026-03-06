import numpy as np

# Use relative import when imported as module, absolute when run directly
try:
    from .elements import Element3D
except ImportError:
    from elements import Element3D

class Wedge3D(Element3D):
  """
  Wedge (prism) linear :math:`\\mathcal{W}_1` parametric element in 3D, :math:`\\mathbb R^3`.
  Composed of two triangular faces and three quadrilateral faces.
  Inherits from :py:class:`Element3D <stagpyviz.Element3D>`.
  
  .. image:: ../figures/el_W1.png
    :align: center
    :alt: Wedge reference element
    :width: 300

  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 6
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    """
    Evaluate the shape functions at given reference coordinates :math:`\\boldsymbol \\xi = (\\xi, \\eta, \\zeta)`.
    The shape functions for a linear wedge are given by the product of the triangular shape functions in the 
    :math:`(\\xi, \\eta)` plane and the linear shape functions in the :math:`\\zeta` direction:

    .. math::
      \\begin{split}
        N_0(\\xi, \\eta, \\zeta) &= (1 - \\xi - \\eta) \\cdot \\frac{1 - \\zeta}{2} \\\\
        N_1(\\xi, \\eta, \\zeta) &= \\xi \\cdot \\frac{1 - \\zeta}{2} \\\\
        N_2(\\xi, \\eta, \\zeta) &= \\eta \\cdot \\frac{1 - \\zeta}{2} \\\\
        N_3(\\xi, \\eta, \\zeta) &= (1 - \\xi - \\eta) \\cdot \\frac{1 + \\zeta}{2} \\\\
        N_4(\\xi, \\eta, \\zeta) &= \\xi \\cdot \\frac{1 + \\zeta}{2} \\\\
        N_5(\\xi, \\eta, \\zeta) &= \\eta \\cdot \\frac{1 + \\zeta}{2}
      \\end{split}

    :param numpy.ndarray xi: Array of shape ``(3,)`` containing the reference coordinates.
    :return: Array of shape ``(6,)`` containing the shape function values at the given reference coordinates.
    :rtype: numpy.ndarray
    """
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
    """
    Return the shape function values at the centroid of the element.
    For a linear wedge, the shape functions at the centroid are all equal to 1/6.

    :return: Array of shape ``(6,)`` containing the shape function values at the element centroid.
    :rtype: numpy.ndarray
    """
    Ni = np.ones((self.basis_per_el), dtype=np.float64) / 6.0
    return Ni
  
  def evaluate_GNi(self, xi:np.ndarray):
    """
    Evaluate the derivatives of the shape functions with respect to the reference coordinates 
    at given reference coordinates :math:`\\boldsymbol \\xi = (\\xi, \\eta, \\zeta)`.

    .. math::
      \\frac{\\partial N_k}{\\partial \\boldsymbol \\xi} &= 
      \\frac{1}{2}
      \\begin{bmatrix}
        -(1 - \\zeta) & -(1 - \\zeta) & -(1 - \\xi - \\eta) \\\\
        1 - \\zeta & 0 & -\\xi \\\\
        0 & 1 - \\zeta & -\\eta \\\\
        -(1 + \\zeta) & -(1 + \\zeta) & 1 - \\xi - \\eta \\\\
        1 + \\zeta & 0 & \\xi \\\\
        0 & 1 + \\zeta & \\eta \\\\
      \\end{bmatrix} \\\\
      
    where the rows correspond to the shape functions :math:`N_k` 
    and the columns correspond to the reference coordinates :math:`(\\xi, \\eta, \\zeta)`.

    :param numpy.ndarray xi: Array of shape ``(3,)`` containing the reference coordinates.
    :return: Array of shape ``(6, 3)`` containing the derivatives of the shape functions with respect to the reference coordinates at the given reference coordinates.
    :rtype: numpy.ndarray
    """
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
    """
    Evaluate the derivatives of the shape functions with respect to the reference coordinates at the element centroid. 
    Calls the method :py:meth:`evaluate_GNi <stagpyviz.Wedge3D.evaluate_GNi>` with the 
    reference coordinates of the centroid :math:`(\\xi, \\eta, \\zeta) = (1/3, 1/3, 0)`. 

    :return: 
      Array of shape ``(6, 3)`` containing the derivatives of the shape functions with 
      respect to the reference coordinates at the element centroid.
    :rtype: numpy.ndarray
    """
    xi  = np.array([1.0/3.0, 1.0/3.0, 0.0], dtype=np.float64)
    GNi = self.evaluate_GNi(xi)
    return GNi