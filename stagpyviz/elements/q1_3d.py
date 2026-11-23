import numpy as np
from time import perf_counter

# Use relative import when imported as module, absolute when run directly
try:
  from .elements import Element3D
except ImportError:
  from elements import Element3D

class Q1_3D(Element3D):
  """
  Quadrilateral hexahedral :math:`\\mathcal{Q}_1` parametric element in 3D, :math:`\\mathbb R^3`.
  Reference coordinates :math:`\\boldsymbol \\xi = (\\xi, \\eta, \\zeta) \\in [-1, 1]^3`.
  Inherits from :py:class:`Element3D <stagpyviz.Element3D>`.

  .. image:: ../figures/el_Q1_3D.png
    :align: center
    :alt: Q1 hexahedral reference element
    :width: 300

  """
  def __init__(self):
    super().__init__()
    self.basis_per_el = 8
    return

  @property
  def vertices(self) -> np.ndarray:
    """
    Return the coordinates of the vertices of the reference hexahedral element in 3D.

    :return: Array of shape ``(8, 3)`` containing the coordinates of the vertices.
    :rtype: numpy.ndarray
    """
    vert = np.array([
      [-1, -1, -1],
      [ 1, -1, -1],
      [-1,  1, -1],
      [ 1,  1, -1],
      [-1, -1,  1],
      [ 1, -1,  1],
      [-1,  1,  1],
      [ 1,  1,  1]
    ], dtype=np.float64)
    return vert

  def quadrature(self, nqp) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the quadrature weights and points for the specified number of quadrature points.
    Supported values for nqp are:

    - ``1``: 1 point quadrature rule (centroid), call to :py:meth:`quadrature_1pt <stagpyviz.Q1_3D.quadrature_1pt>`
    - ``8``: 2x2x2 point quadrature rule, call to :py:meth:`quadrature_8pt <stagpyviz.Q1_3D.quadrature_8pt>`
    
    :param int nqp: Number of quadrature points. Supported values are ``1`` and ``8``.
    :return: Tuple containing the quadrature weights and points.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    match nqp:
      case 1:
        return self.quadrature_1pt()
      case 8:
        return self.quadrature_8pt()
      case _:
        raise ValueError(f"Unsupported number of quadrature points: {nqp}. Supported values are 1 and 8.")
    return

  def quadrature_1pt(self) -> tuple[np.ndarray, np.ndarray]:
    """
    1 point quadrature rule for a hexahedral element in 3D, 
    which corresponds to evaluating the integrand at the centroid of the reference element.
    
    .. math::
      w_0 = 8, \\quad \\boldsymbol \\xi_0 = 
      \\begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\end{bmatrix}

    with :math:`w_0` the quadrature weight 
    and :math:`\\boldsymbol \\xi_0` the quadrature point.
    
    :return: 
      Tuple containing the quadrature weights and points.
      The weight is returned as a 1D array of shape ``(1,)``, 
      and the points are returned as a 2D array of shape ``(1, 3)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    weight = np.array([8.0], dtype=np.float64)
    point  = np.zeros((1,3), dtype=np.float64)
    return weight, point

  def quadrature_8pt(self) -> tuple[np.ndarray, np.ndarray]:
    """
    :math:`2 \\times 2 \\times 2` point quadrature rule for a hexahedral element in 3D.

    .. math::
      w_q = 1, \\quad \\boldsymbol \\xi_q = 
      \\begin{bmatrix} 
        \\pm 1/\\sqrt{3}\\\\ 
        \\pm 1/\\sqrt{3}\\\\ 
        \\pm 1/\\sqrt{3}
      \\end{bmatrix}

    with :math:`w_q` the quadrature weights (all equal to 1 for this rule)
    and :math:`\\boldsymbol \\xi_q` the quadrature points.

    :return:
      Tuple containing the quadrature weights and points.
      The weights are returned as a 1D array of shape ``(8,)``,
      and the points are returned as a 2D array of shape ``(8, 3)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    one_sqrt3 = np.float64(0.5773502691896258)
    weights   = np.full(8, 1.0, dtype=np.float64)
    points    = np.array([
      [-one_sqrt3, -one_sqrt3, -one_sqrt3],
      [ one_sqrt3, -one_sqrt3, -one_sqrt3],
      [-one_sqrt3,  one_sqrt3, -one_sqrt3],
      [ one_sqrt3,  one_sqrt3, -one_sqrt3],
      [-one_sqrt3, -one_sqrt3,  one_sqrt3],
      [ one_sqrt3, -one_sqrt3,  one_sqrt3],
      [-one_sqrt3,  one_sqrt3,  one_sqrt3],
      [ one_sqrt3,  one_sqrt3,  one_sqrt3]
    ], dtype=np.float64)
    return weights, points

  def evaluate_Ni(self, xi:np.ndarray) -> np.ndarray:
    """
    Evaluate the shape functions at given reference coordinates 
    :math:`\\boldsymbol \\xi = (\\xi, \\eta, \\zeta)`.
    The shape functions for a bilinear hexahedron are given by:

    .. math::
      N_i(\\xi, \\eta, \\zeta) = \\frac{1}{8} 
      (1 + \\xi \\, \\xi_i) 
      (1 + \\eta \\, \\eta_i) 
      (1 + \\zeta \\, \\zeta_i), 
      \\quad i = 0, 1, ..., 7

    where :math:`(\\xi_i, \\eta_i, \\zeta_i)` are the reference coordinates of 
    the vertices of the reference element.
      
    :param numpy.ndarray xi: Array of shape ``(3,)`` containing the reference coordinates.
    :return: Array of shape ``(8,)`` containing the shape function values at the given reference coordinates.
    :rtype: numpy.ndarray
    """
    if xi.ndim == 1:
      npoints = 1
      xi = xi.reshape((1,3))
    elif xi.ndim == 2:
      npoints = xi.shape[0]
    else:
      raise ValueError("xi must be a 1D array of shape (2,) or a 2D array of shape (npoints, 2)")
    Ni = np.zeros((npoints,self.basis_per_el),dtype=np.float64)
    vert = self.vertices
    for i in range(self.basis_per_el):
      Ni[:,i] = 0.125 * (1 + xi[:,0]*vert[i,0]) * (1 + xi[:,1]*vert[i,1]) * (1 + xi[:,2]*vert[i,2])
    return Ni

  def Ni_centroid(self) -> np.ndarray:
    """
    Return the shape function values at the centroid of the element.
    For the chosen bilinear hexahedron, the shape functions at the centroid are all equal to 1/8.

    :return: Array of shape ``(8,)`` containing the shape function values at the element centroid.
    :rtype: numpy.ndarray
    """
    Ni = np.full((self.basis_per_el), 0.125, dtype=np.float64)
    return Ni

  def evaluate_GNi(self, xi:np.ndarray) -> np.ndarray:
    """
    Evaluate the derivatives of the shape functions with respect to the reference coordinates 
    at given reference coordinates :math:`\\boldsymbol \\xi = (\\xi, \\eta, \\zeta)`.
    The derivatives of the shape functions for a bilinear hexahedron are given by:

    .. math::
      \\frac{\\partial N_k}{\\partial \\boldsymbol \\xi} = \\frac{1}{8}
      \\begin{bmatrix}
        \\xi_k (1 + \\eta \\, \\eta_k) (1 + \\zeta \\, \\zeta_k) & 
        (1 + \\xi \\, \\xi_k) \\eta_k (1 + \\zeta \\, \\zeta_k) & 
        (1 + \\xi \\, \\xi_k) (1 + \\eta \\, \\eta_k) \\zeta_k
      \\end{bmatrix}

    where :math:`(\\xi_k, \\eta_k, \\zeta_k)` are the reference coordinates of 
    the vertices of the reference element, 
    the rows correspond to the shape functions :math:`N_k` 
    and the columns correspond to the reference coordinates :math:`(\\xi, \\eta, \\zeta)`.

    :param numpy.ndarray xi: Array of shape ``(3,)`` containing the reference coordinates.
    :return: 
      Array of shape ``(8, 3)`` containing the derivatives of the shape functions 
      with respect to the reference coordinates at the given reference coordinates.
    :rtype: numpy.ndarray
    """
    if xi.ndim == 1:
      npoints = 1
      xi = xi.reshape((1,3))
    elif xi.ndim == 2:
      npoints = xi.shape[0]
    else:
      raise ValueError("xi must be a 1D array of shape (3,) or a 2D array of shape (npoints, 3)")
    GNi = np.zeros((npoints,self.basis_per_el,3),dtype=np.float64)
    vert = self.vertices
    for i in range(self.basis_per_el):
      GNi[:,i,0] = 0.125 * vert[i,0] * (1 + xi[:,1]*vert[i,1]) * (1 + xi[:,2]*vert[i,2])
      GNi[:,i,1] = 0.125 * (1 + xi[:,0]*vert[i,0]) * vert[i,1] * (1 + xi[:,2]*vert[i,2])
      GNi[:,i,2] = 0.125 * (1 + xi[:,0]*vert[i,0]) * (1 + xi[:,1]*vert[i,1]) * vert[i,2]
    return GNi

  def GNi_centroid(self) -> np.ndarray:
    """
    Evaluate the derivatives of the shape functions with respect to the reference coordinates at the element centroid.
    See :py:meth:`evaluate_GNi <stagpyviz.Q1_2D.evaluate_GNi>` for the shape function derivatives 
    with respect to the reference coordinates.
    
    :return: 
      Array of shape ``(8, 3)`` containing the derivatives of the shape functions 
      with respect to the reference coordinates at the element centroid.
    :rtype: numpy.ndarray
    """
    GNi = np.array([
      [-0.125, -0.125, -0.125],
      [ 0.125, -0.125, -0.125],
      [-0.125,  0.125, -0.125],
      [ 0.125,  0.125, -0.125],
      [-0.125, -0.125,  0.125],
      [ 0.125, -0.125,  0.125],
      [-0.125,  0.125,  0.125],
      [ 0.125,  0.125,  0.125]
    ], dtype=np.float64)
    return GNi