import numpy as np

# Use relative import when imported as module, absolute when run directly
try:
    from .elements import Element2D, SurfaceElement
except ImportError:
    from elements import Element2D, SurfaceElement

class P1_2D(Element2D):
  """
  Triangular linear :math:`\\mathcal{P}_1` parametric element in 2D, :math:`\\mathbb R^2`.
  Nodes are ordered couterclockwise
  and the reference coordinates :math:`\\xi, \\eta` range from 0 to 1 with the constraint :math:`\\xi + \\eta \\leq 1`.
  Inherits from :py:class:`Element2D <stagpyviz.Element2D>`.

  .. image:: ../figures/el_P1.png
    :align: center
    :alt: P1 triangle reference element
    :width: 300

  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 3
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    """
    Evaluate the shape functions at given reference coordinates :math:`\\xi = (\\xi, \\eta)`.
    Shape functions for a linear triangle are:

    .. math::
      \\begin{split}
        N_0(\\xi, \\eta) &= 1 - \\xi - \\eta \\\\
        N_1(\\xi, \\eta) &= \\xi \\\\
        N_2(\\xi, \\eta) &= \\eta
      \\end{split}

    :param numpy.ndarray xi: Array of shape ``(2,)`` containing the reference coordinates.
    :return: Array of shape ``(3,)`` containing the shape function values at the given reference coordinates.
    :rtype: numpy.ndarray
    """
    Ni = np.zeros((self.basis_per_el),dtype=np.float64)
    Ni[0] = 1.0 - xi[0] - xi[1]
    Ni[1] = xi[0]
    Ni[2] = xi[1]
    return Ni
  
  def Ni_centroid(self):
    """
    Compute the shape function values at the element centroid.
    For the chosen linear triangle, the shape functions at the centroid are all equal to 1/3.

    :return: Array of shape ``(3,)`` containing the shape function values at the element centroid.
    :rtype: numpy.ndarray
    """
    Ni = np.ones((self.basis_per_el), dtype=np.float64) / 3.0
    return Ni
  
  def evaluate_GNi(self):
    """
    Compute the derivatives of the shape functions with respect to the reference coordinates.
    For a linear triangle, the derivatives are constant and given by:

      .. math::
        \\frac{\\partial N_k}{\\partial \\xi_i} = 
        \\begin{bmatrix}
          -1 & -1 \\\\
          1 &  0 \\\\
          0 &  1
        \\end{bmatrix}

    :return: Array of shape ``(3, 2)`` containing the derivatives of the shape functions with respect to the reference coordinates.
    :rtype: numpy.ndarray
    """
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
    Computes the shape function derivatives at the element centroid.
    The derivatives of the shape functions with respect to the reference coordinates are constant for a linear triangle, 
    so this method simply returns the same values as :py:meth:`evaluate_GNi <stagpyviz.P1_2D.evaluate_GNi>`.

    :return: Array of shape ``(3, 2)`` containing the derivatives of the shape functions with respect to the reference coordinates at the element centroid.
    :rtype: numpy.ndarray
    """
    GNi = np.array([
      [-1.0, -1.0],
      [ 1.0,  0.0],
      [ 0.0,  1.0]
    ], dtype=np.float64)
    return GNi

  def compute_area(self, xe:np.ndarray) -> np.ndarray:
    """
    Compute the area of the triangle defined by the coordinates of its vertices.
    The area is computed using the formula:

    .. math::
      A = \\frac{1}{2} |\\det(J)|
    
    where :math:`J` is the Jacobian matrix of the transformation from the reference element 
    to the physical element, evaluated at the element centroid.

    :param numpy.ndarray xe: 
      For a single element: array of shape ``(3, 2)`` containing the coordinates of the triangle vertices.
      For multiple elements: array of shape ``(n_elements, 3, 2)`` containing the coordinates of the triangle vertices for each element.
    :return: Area of the triangle(s).
    :rtype: np.ndarray
    """
    GNi  = self.GNi_centroid()
    J    = self.evaluate_Jacobian(GNi, xe)
    detJ = self.evaluate_detJ(J)
    area = 0.5 * np.abs(detJ)
    return area
  
class P1_2D_R3(SurfaceElement):
  """
  Triangular linear :math:`\\mathcal{P}_1` parametric element in 3D, representing a surface element in :math:`\\mathbb R^3`.
  Nodes are ordered couterclockwise
  and the reference coordinates :math:`\\xi, \\eta` range from 0 to 1 with the constraint :math:`\\xi + \\eta \\leq 1`.
  Inherits from :py:class:`SurfaceElement <stagpyviz.SurfaceElement>`.
  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 3
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    """
    Evaluate the shape functions at given reference coordinates :math:`\\xi = (\\xi, \\eta)`.
    See :py:meth:`evaluate_Ni <stagpyviz.P1_2D.evaluate_Ni>` for the shape function definitions.
    """
    Ni = np.zeros((self.basis_per_el),dtype=np.float64)
    Ni[0] = 1.0 - xi[0] - xi[1]
    Ni[1] = xi[0]
    Ni[2] = xi[1]
    return Ni
  
  def Ni_centroid(self):
    """
    Compute the shape function values at the element centroid.
    See :py:meth:`Ni_centroid <stagpyviz.P1_2D.Ni_centroid>` for the shape function values at the centroid.
    """
    Ni = np.ones((self.basis_per_el), dtype=np.float64) / 3.0
    return Ni
  
  def evaluate_GNi(self):
    """
    Compute the derivatives of the shape functions with respect to the reference coordinates.
    See :py:meth:`evaluate_GNi <stagpyviz.P1_2D.evaluate_GNi>` 
    for the shape function derivatives with respect to the reference coordinates.
    """
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
    Compute the shape function derivatives at the element centroid.
    See :py:meth:`GNi_centroid <stagpyviz.P1_2D.GNi_centroid>` for the shape function derivatives at the centroid.
    """
    return self.evaluate_GNi()
  
  def compute_area(self, xe:np.ndarray):
    """
    Compute the area of the triangle defined by the coordinates of its vertices.
    The area is computed using the formula:

    .. math::
      A = \\frac{1}{2} |\\det(J)|
    
    where :math:`J` is the Jacobian matrix of the transformation from the reference element 
    to the physical element, evaluated at the element centroid.

    :param numpy.ndarray xe: 
      For a single element: array of shape ``(3, 3)`` containing the coordinates of the triangle vertices.
      For multiple elements: array of shape ``(n_elements, 3, 3)`` containing the coordinates of the triangle vertices for each element.
    :return: Area of the triangle(s).
    :rtype: np.ndarray
    """
    return P1_2D.compute_area(self, xe)