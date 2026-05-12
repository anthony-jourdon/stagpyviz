import numpy as np
from time import perf_counter

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

  .. image:: ../figures/el_Q1.png
    :align: center
    :alt: Q1 quadrilateral reference element
    :width: 300

  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 4
    return
  
  def quadrature(self, nqp) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the quadrature weights and points for the specified number of quadrature points.
    Supported values for nqp are:

    - ``1``: 1 point quadrature rule (centroid), call to :py:meth:`quadrature_1pt <stagpyviz.Q1_2D.quadrature_1pt>`
    - ``4``: 2x2 point quadrature rule, call to :py:meth:`quadrature_4pt <stagpyviz.Q1_2D.quadrature_4pt>`
    
    :param int nqp: Number of quadrature points. Supported values are ``1`` and ``4``.
    :return: Tuple containing the quadrature weights and points.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    match nqp:
      case 1:
        return self.quadrature_1pt()
      case 4:
        return self.quadrature_4pt()
      case _:
        raise ValueError(f"Unsupported number of quadrature points: {nqp}. Supported values are 1 and 4.")
    return 

  def quadrature_1pt(self) -> tuple[np.ndarray, np.ndarray]:
    """
    1 point quadrature rule for a quadrilateral element in 2D, 
    which corresponds to evaluating the integrand at the centroid of the reference element.
    
    .. math::
      w_0 = 4, \\quad \\boldsymbol \\xi_0 = 
      \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix}

    with :math:`w_0` the quadrature weight 
    and :math:`\\boldsymbol \\xi_0` the quadrature point.
    
    :return: 
      Tuple containing the quadrature weights and points.
      The weight is returned as a 1D array of shape ``(1,)``, 
      and the points are returned as a 2D array of shape ``(1, 2)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    weight = np.array([4.0], dtype=np.float64)
    point  = np.zeros((1,2), dtype=np.float64)
    return weight, point
  
  def quadrature_4pt(self) -> tuple[np.ndarray, np.ndarray]:
    """
    :math:`2 \\times 2` point quadrature rule for a quadrilateral element in 2D.

    .. math::
      w_q = 1, \\quad \\boldsymbol \\xi_q = 
      \\begin{bmatrix} 
        \\pm 1/\\sqrt{3}\\\\ 
        \\pm 1/\\sqrt{3}
      \\end{bmatrix}

    with :math:`w_q` the quadrature weights (all equal to 1 for this rule)
    and :math:`\\boldsymbol \\xi_q` the quadrature points.

    :return:
      Tuple containing the quadrature weights and points.
      The weights are returned as a 1D array of shape ``(4,)``, 
      and the points are returned as a 2D array of shape ``(4, 2)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    one_sqrt3      = np.float64(0.5773502691896258)
    weights = np.full(4, 1.0, dtype=np.float64)
    points  = np.array([
      [-one_sqrt3, -one_sqrt3], # point 0
      [ one_sqrt3, -one_sqrt3], # point 1
      [-one_sqrt3,  one_sqrt3], # point 2
      [ one_sqrt3,  one_sqrt3]  # point 3
    ], dtype=np.float64)
    return weights, points
  
  def evaluate_Ni(self, xi:np.ndarray) -> np.ndarray:
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
    if xi.ndim == 1:
      npoints = 1
      xi = xi.reshape((1,2))
    elif xi.ndim == 2:
      npoints = xi.shape[0]
    else:
      raise ValueError("xi must be a 1D array of shape (2,) or a 2D array of shape (npoints, 2)")
    Ni = np.zeros((npoints,self.basis_per_el),dtype=np.float64)
    Ni[:,0] = 0.25 * (1.0 - xi[:,0]) * (1.0 - xi[:,1])
    Ni[:,1] = 0.25 * (1.0 + xi[:,0]) * (1.0 - xi[:,1])
    Ni[:,2] = 0.25 * (1.0 - xi[:,0]) * (1.0 + xi[:,1])
    Ni[:,3] = 0.25 * (1.0 + xi[:,0]) * (1.0 + xi[:,1])
    return Ni
  
  def Ni_centroid(self) -> np.ndarray:
    """
    Return the shape function values at the centroid of the element.
    For the chosen bilinear quadrilateral, the shape functions at the centroid are all equal to 1/4.

    :return: Array of shape ``(4,)`` containing the shape function values at the element centroid.
    :rtype: numpy.ndarray
    """
    Ni = np.full((self.basis_per_el), 0.25, dtype=np.float64)
    return Ni
  
  def evaluate_GNi(self, xi:np.ndarray) -> np.ndarray:
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
    if xi.ndim == 1:
      npoints = 1
      xi = xi.reshape((1,2))
    elif xi.ndim == 2:
      npoints = xi.shape[0]
    else:
      raise ValueError("xi must be a 1D array of shape (2,) or a 2D array of shape (npoints, 2)")
    GNi = np.zeros((npoints,self.basis_per_el,2),dtype=np.float64)
    GNi[:,0,0] = -0.25 * (1.0 - xi[:,1]) # dN0/dxi
    GNi[:,0,1] = -0.25 * (1.0 - xi[:,0]) # dN0/deta
    GNi[:,1,0] =  0.25 * (1.0 - xi[:,1]) # dN1/dxi
    GNi[:,1,1] = -0.25 * (1.0 + xi[:,0]) # dN1/deta
    GNi[:,2,0] = -0.25 * (1.0 + xi[:,1]) # dN2/dxi
    GNi[:,2,1] =  0.25 * (1.0 - xi[:,0]) # dN2/deta
    GNi[:,3,0] =  0.25 * (1.0 + xi[:,1]) # dN3/dxi
    GNi[:,3,1] =  0.25 * (1.0 + xi[:,0]) # dN3/deta
    return GNi
  
  def GNi_centroid(self) -> np.ndarray:
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
