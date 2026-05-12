import numpy as np
from time import perf_counter

# Use relative import when imported as module, absolute when run directly
try:
    from .elements import Element3D
    from .p1_2d import P1_2D
except ImportError:
    from elements import Element3D
    from p1_2d import P1_2D

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
  
  def quadrature(self, nqp):
    """
    Return the quadrature weights and points for the specified number of quadrature points.
    Supported values for nqp are:

    - ``1``: 1 point quadrature rule (centroid), call to :py:meth:`quadrature_1pt <stagpyviz.Wedge3D.quadrature_1pt>`
    - ``6``: 3x2 point quadrature rule, call to :py:meth:`quadrature_6pt <stagpyviz.Wedge3D.quadrature_6pt>`

    :param int nqp: Number of quadrature points. Supported values are ``1`` and ``6``.
    :return: Tuple containing the quadrature weights and points.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    match nqp:
      case 1:
        return self.quadrature_1pt()
      case 6:
        return self.quadrature_6pt()
      case _:
        raise ValueError(f"Unsupported number of quadrature points: {nqp}. Supported values are 1 and 6.")
    return 
  
  def quadrature_1pt(self):
    """
    1 point quadrature rule for a wedge element in 3D, 
    which corresponds to evaluating the integrand at the centroid of the reference element.

    .. math::
      w_0 = 1, \\quad \\boldsymbol \\xi_0 = 
      \\begin{bmatrix} 
        1/3 \\\\ 
        1/3 \\\\
        0
      \\end{bmatrix}

    with :math:`w_0` the quadrature weight and :math:`\\boldsymbol \\xi_0` the quadrature point.

    :return: Tuple containing the quadrature weights and points.
      The weight is returned as a 1D array of shape ``(1,)``,
      and the points are returned as a 2D array of shape ``(1, 3)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    weight = np.array([1.0], dtype=np.float64)
    point = np.array([
      [1.0/3.0, 1.0/3.0, 0.0]
    ], dtype=np.float64)
    return weight, point
  
  def quadrature_6pt(self):
    """
    :math:`3 \\times 2` point quadrature rule for a wedge element in 3D,
    which corresponds to the tensor product of the 3 point quadrature rule for a triangle 
    and the 2 Gauss point quadrature rule for a line.

    .. math::
      w_{ij} = w_i^{\\triangle} \\cdot w_j^{\\text{line}}, \\quad \\boldsymbol \\xi_{ij} = 
      \\begin{bmatrix} 
        \\xi_i^{\\triangle} \\\\ 
        \\eta_i^{\\triangle} \\\\
        \\zeta_j^{\\text{line}}
      \\end{bmatrix}

    :return: Tuple containing the quadrature weights and points.
      The weights are returned as a 1D array of shape ``(6,)``,
      and the points are returned as a 2D array of shape ``(6, 3)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    one_sixth = np.float64(0.16666666666666666)
    two_third = np.float64(0.6666666666666666)
    one_sqrt3 = np.float64(0.5773502691896258)

    weights = np.full(6, 1.0/6.0, dtype=np.float64)
    points = np.array([
      [one_sixth, one_sixth, -one_sqrt3],
      [two_third, one_sixth, -one_sqrt3],
      [one_sixth, two_third, -one_sqrt3],
      [one_sixth, one_sixth, one_sqrt3],
      [two_third, one_sixth, one_sqrt3],
      [one_sixth, two_third, one_sqrt3]
    ], dtype=np.float64)
    return weights, points
  
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
    Ni_triangle = P1_2D().evaluate_Ni(xi)
    if xi.ndim == 1:
      npoints = 1
      xi = xi.reshape((1,3))
      Ni_triangle = Ni_triangle.reshape((1,3))
    elif xi.ndim == 2:
      npoints = xi.shape[0]
    else:
      raise ValueError(f"xi must be a 1D array of shape (3,) or a 2D array of shape (npoints, 3), got {xi.shape}")
    Ni = np.zeros((npoints,self.basis_per_el), dtype=np.float64)
    vm = 0.5 * (1.0 - xi[:,2])
    vp = 0.5 * (1.0 + xi[:,2])
    for k in range(3):
      Ni[:,k]   = Ni_triangle[:,k] * vm
      Ni[:,k+3] = Ni_triangle[:,k] * vp
    if npoints == 1:
      Ni = Ni.reshape((self.basis_per_el))
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

    :param numpy.ndarray xi: 
      Array of shape ``(npoints, 3)`` containing the reference coordinates.
    :return: 
      Array of shape ``(npoints, 6, 3)`` containing the derivatives of the 
      shape functions with respect to the reference coordinates at the given reference coordinates.
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
    #dN0
    GNi[:,0,0] = -0.5 * (1.0 - xi[:,2])
    GNi[:,0,1] = -0.5 * (1.0 - xi[:,2])
    GNi[:,0,2] = -0.5 * (1.0 - xi[:,0] - xi[:,1])
    #dN1
    GNi[:,1,0] = 0.5 * (1.0 - xi[:,2])
    GNi[:,1,1] = 0.0
    GNi[:,1,2] = -0.5 * xi[:,0]
    #dN2
    GNi[:,2,0] = 0.0
    GNi[:,2,1] = 0.5 * (1.0 - xi[:,2])
    GNi[:,2,2] = -0.5 * xi[:,1]
    #dN3
    GNi[:,3,0] = -0.5 * (1.0 + xi[:,2])
    GNi[:,3,1] = -0.5 * (1.0 + xi[:,2])
    GNi[:,3,2] = 0.5 * (1.0 - xi[:,0] - xi[:,1])
    #dN4
    GNi[:,4,0] = 0.5 * (1.0 + xi[:,2])
    GNi[:,4,1] = 0.0
    GNi[:,4,2] = 0.5 * xi[:,0]
    #dN5
    GNi[:,5,0] = 0.0
    GNi[:,5,1] = 0.5 * (1.0 + xi[:,2])
    GNi[:,5,2] = 0.5 * xi[:,1]
    if npoints == 1:
      GNi = GNi.reshape((self.basis_per_el, 3))
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