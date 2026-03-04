import numpy as np

# Use relative import when imported as module, absolute when run directly
try:
    from .elements import Element2D
except ImportError:
    from elements import Element2D

class Q1_2D(Element2D):
  """
  Quadrilateral bilinear :math:`\\mathcal{Q}_1` parametric element in 2D, :math:`\\mathbb R^2`.
  Reference coordinates :math:`\\boldsymbol \\xi = (\\xi, \\eta) \\in [-1, 1]^2`.
  Inherits from :py:class:`Element2D <stagpyviz.Element2D>`.

  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 4
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    """
    Evaluate the shape functions at given reference coordinates :math:`\\boldsymbol \\xi = (\\xi, \\eta)`.
    The shape functions for a bilinear quadrilateral are given by:

    .. math::
      \\begin{split}
        N_0(\\xi, \\eta) &= \\frac{1}{4} (1 - \\xi) (1 - \\eta) \\\\
        N_1(\\xi, \\eta) &= \\frac{1}{4} (1 + \\xi) (1 - \\eta) \\\\
        N_2(\\xi, \\eta) &= \\frac{1}{4} (1 - \\xi) (1 + \\eta) \\\\
        N_3(\\xi, \\eta) &= \\frac{1}{4} (1 + \\xi) (1 + \\eta)
      \\end{split}

    :param numpy.ndarray xi: Array of shape ``(2,)`` containing the reference coordinates.
    :return: Array of shape ``(4,)`` containing the shape function values at the given reference coordinates.
    :rtype: numpy.ndarray
    """
    Ni = np.zeros((self.basis_per_el),dtype=np.float64)
    Ni[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1])
    Ni[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1])
    Ni[2] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1])
    Ni[3] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1])
    return Ni
  
  def Ni_centroid(self):
    """
    Return the shape function values at the centroid of the element.
    For the chosen bilinear quadrilateral, the shape functions at the centroid are all equal to 1/4.

    :return: Array of shape ``(4,)`` containing the shape function values at the element centroid.
    :rtype: numpy.ndarray
    """
    Ni = np.ones((self.basis_per_el), dtype=np.float64) * 0.25
    return Ni
  
  def evaluate_GNi(self, xi:np.ndarray):
    """
    Evaluate the derivatives of the shape functions with respect to the reference coordinates 
    at given reference coordinates :math:`\\boldsymbol \\xi = (\\xi, \\eta)`.
    The derivatives of the shape functions for a bilinear quadrilateral are given by:

    .. math::
      \\frac{\\partial N_k}{\\partial \\boldsymbol \\xi} = \\frac{1}{4}
      \\begin{bmatrix}
        -(1 - \\eta) & -(1 - \\xi) \\\\
        (1 - \\eta) & -(1 + \\xi) \\\\
        -(1 + \\eta) & -(1 - \\xi) \\\\
        (1 + \\eta) & -(1 + \\xi)
      \\end{bmatrix}

    where the rows correspond to the shape functions :math:`N_k` 
    and the columns correspond to the reference coordinates :math:`(\\xi, \\eta)`.

    :param numpy.ndarray xi: Array of shape ``(2,)`` containing the reference coordinates.
    :return: 
      Array of shape ``(4, 2)`` containing the derivatives of the shape functions 
      with respect to the reference coordinates at the given reference coordinates.
    :rtype: numpy.ndarray
    """
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
    Evaluate the derivatives of the shape functions with respect to the reference coordinates at the element centroid.
    See :py:meth:`evaluate_GNi <stagpyviz.Q1_2D.evaluate_GNi>` for the shape function derivatives 
    with respect to the reference coordinates.
    
    :return: 
      Array of shape ``(4, 2)`` containing the derivatives of the shape functions 
      with respect to the reference coordinates at the element centroid.
    :rtype: numpy.ndarray
    """
    GNi = np.array([
      [-0.25, -0.25],  # node 0
      [ 0.25, -0.25],  # node 1
      [-0.25,  0.25],  # node 2
      [ 0.25,  0.25],  # node 3
    ], dtype=np.float64)
    return GNi