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
  
  def quadrature(self, nqp:int):
    """
    Return the quadrature weights and points for the specified number of quadrature points.
    Supported values for nqp are:

    - ``1``: 1 point quadrature rule (centroid), call to :py:meth:`quadrature_1pt <stagpyviz.P1_2D.quadrature_1pt>`
    - ``3``: 3 point quadrature rule, call to :py:meth:`quadrature_3pt <stagpyviz.P1_2D.quadrature_3pt>`

    :param int nqp: Number of quadrature points. Supported values are ``1`` and ``3``.
    :return: Tuple containing the quadrature weights and points.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    match nqp:
      case 1:
        return self.quadrature_1pt()
      case 3:
        return self.quadrature_3pt()
      case _:
        raise ValueError(f"Unsupported number of quadrature points: {nqp}. Supported values are 1 and 3.")
    return
  
  def quadrature_1pt(self):
    """
    1 point quadrature rule for a triangular element in 2D, 
    which corresponds to evaluating the integrand at the centroid of the reference element.

    .. math::
      w_0 = \\frac{1}{2}, \\quad \\boldsymbol \\xi_0 = 
      \\begin{bmatrix} 
        1/3 \\\\ 
        1/3 
      \\end{bmatrix}

    with :math:`w_0` the quadrature weight 
    and :math:`\\boldsymbol \\xi_0` the quadrature point.

    :return:
      Tuple containing the quadrature weights and points.
      The weight is returned as a 1D array of shape ``(1,)``,
      and the points are returned as a 2D array of shape ``(1, 2)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    weight = np.array([0.5], dtype=np.float64) # area of the reference triangle
    points = np.full((1, 2), 1.0/3.0, dtype=np.float64) # centroid of the triangle
    return weight, points
  
  def quadrature_3pt(self):
    """
    3 point quadrature rule for a triangular element in 2D,
    which corresponds to evaluating the integrand at the three points defined by:

    .. math::
      \\begin{split}
        w_q &= \\frac{1}{6} \\\\
        \\boldsymbol \\xi_0 &= 
        \\begin{bmatrix}
          1/6 \\\\
          1/6
        \\end{bmatrix} \\quad
        \\boldsymbol \\xi_1 = 
        \\begin{bmatrix}
          2/3 \\\\
          1/6
        \\end{bmatrix} \\quad
        \\boldsymbol \\xi_2 = 
        \\begin{bmatrix}
          1/6 \\\\
          2/3
        \\end{bmatrix}
      \\end{split}

    with :math:`w_q` the quadrature weights (all equal to 1/6 for this rule)
    and :math:`\\boldsymbol \\xi_q` the quadrature points.

    :return:
      Tuple containing the quadrature weights and points.
      The weights are returned as a 1D array of shape ``(3,)``,
      and the points are returned as a 2D array of shape ``(3, 2)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    weight = np.full((3,), 1.0/6.0, dtype=np.float64) # each point has weight 1/6, total area is 1/2
    points = np.array([
      [1.0/6.0, 1.0/6.0], # point 1
      [2.0/3.0, 1.0/6.0], # point 2
      [1.0/6.0, 2.0/3.0]  # point 3
    ], dtype=np.float64)
    return weight, points

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
    if xi.ndim == 1:
      Ni = np.zeros((self.basis_per_el),dtype=np.float64)
      Ni[0] = 1.0 - xi[0] - xi[1]
      Ni[1] = xi[0]
      Ni[2] = xi[1]
    elif xi.ndim == 2:
      Ni = np.zeros((xi.shape[0], self.basis_per_el), dtype=np.float64)
      Ni[:,0] = 1.0 - xi[:,0] - xi[:,1]
      Ni[:,1] = xi[:,0]
      Ni[:,2] = xi[:,1]
    return Ni
  
  def Ni_centroid(self):
    """
    Compute the shape function values at the element centroid.
    For the chosen linear triangle, the shape functions at the centroid are all equal to 1/3.

    :return: Array of shape ``(3,)`` containing the shape function values at the element centroid.
    :rtype: numpy.ndarray
    """
    Ni = np.full((self.basis_per_el), 1.0/3.0, dtype=np.float64) 
    return Ni
  
  def evaluate_GNi(self, xi:np.ndarray) -> np.ndarray:
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
    if xi.ndim == 1:
      npoints = 1
      xi = xi.reshape((1,2))
    elif xi.ndim == 2:
      npoints = xi.shape[0]
    else:
      raise ValueError("xi must be a 1D array of shape (2,) or a 2D array of shape (npoints, 2)")
    GNi = np.zeros((npoints,self.basis_per_el,2),dtype=np.float64)
    GNi[:,0,0] = -1.0 # dN0/dxi
    GNi[:,0,1] = -1.0 # dN0/deta
    GNi[:,1,0] =  1.0 # dN1/dxi
    GNi[:,1,1] =  0.0 # dN1/deta
    GNi[:,2,0] =  0.0 # dN2/dxi
    GNi[:,2,1] =  1.0 # dN2/deta
    return GNi
  
  def GNi_centroid(self) -> np.ndarray:
    """
    Computes the shape function derivatives at the element centroid.
    The derivatives of the shape functions with respect to the reference coordinates are constant for a linear triangle, 
    so this method simply returns the same values as :py:meth:`evaluate_GNi <stagpyviz.P1_2D.evaluate_GNi>`.

    :return: Array of shape ``(3, 2)`` containing the derivatives of the shape functions with respect to the reference coordinates at the element centroid.
    :rtype: numpy.ndarray
    """
    xi = np.array([1.0/3.0, 1.0/3.0], dtype=np.float64) # centroid coordinates
    return self.evaluate_GNi(xi)

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
  
  def quadrature(self, nqp:int):
    """
    Return the quadrature weights and points for the specified number of quadrature points.
    Supported values for nqp are:

    - ``1``: 1 point quadrature rule (centroid), call to :py:meth:`quadrature_1pt <stagpyviz.P1_2D.quadrature_1pt>`
    - ``3``: 3 point quadrature rule, call to :py:meth:`quadrature_3pt <stagpyviz.P1_2D.quadrature_3pt>`
    
    :param int nqp: Number of quadrature points. Supported values are ``1`` and ``3``.
    :return: Tuple containing the quadrature weights and points.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    match nqp:
      case 1:
        return P1_2D.quadrature_1pt(self)
      case 3:
        return P1_2D.quadrature_3pt(self)
      case _:
        raise ValueError(f"Unsupported number of quadrature points: {nqp}. Supported values are 1 and 3.")
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    """
    Evaluate the shape functions at given reference coordinates :math:`\\xi = (\\xi, \\eta)`.
    See :py:meth:`evaluate_Ni <stagpyviz.P1_2D.evaluate_Ni>` for the shape function definitions.
    """
    Ni = P1_2D.evaluate_Ni(self,xi)
    return Ni
  
  def Ni_centroid(self):
    """
    Compute the shape function values at the element centroid.
    See :py:meth:`Ni_centroid <stagpyviz.P1_2D.Ni_centroid>` for the shape function values at the centroid.
    """
    Ni = P1_2D.Ni_centroid(self)
    return Ni
  
  def evaluate_GNi(self, xi:np.ndarray) -> np.ndarray:
    """
    Compute the derivatives of the shape functions with respect to the reference coordinates.
    See :py:meth:`evaluate_GNi <stagpyviz.P1_2D.evaluate_GNi>` 
    for the shape function derivatives with respect to the reference coordinates.
    """
    return P1_2D.evaluate_GNi(self, xi)
  
  def GNi_centroid(self) -> np.ndarray:
    """
    Compute the shape function derivatives at the element centroid.
    See :py:meth:`GNi_centroid <stagpyviz.P1_2D.GNi_centroid>` for the shape function derivatives at the centroid.
    """
    return P1_2D.GNi_centroid(self)