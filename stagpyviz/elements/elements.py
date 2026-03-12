from time import perf_counter
import numpy as np

class Element:
  """
  Base class for isoparametric elements.
  This class defines the interface for evaluating shape functions, their derivatives, and Jacobian-related quantities.
  
  This class is the parent class for some specific element types.

  .. note::
    In the following documentation, the term *reference* refers to the reference element (e.g., the unit triangle or unit square), 
    while *physical* refers to the actual element in the physical domain defined by its node coordinates.

  .. note::
    Currently the implementation of Jacobian related quatities (Jacobian matrix and physical shape functions derivatives) 
    only supports the reference derivatives of the shape functions evaluated at a single point (e.g., the centroid) for all elements.
    
    A simple improvement would be to allow passing the reference derivatives evaluated at different points across the elements (but still one point per element).
    
    A more complex improvement would be to allow passing the reference derivatives evaluated at multiple points across the elements (e.g., quadrature points).
  """
  def __init__(self):
    self.dim = None
    self.basis_per_el = None
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    raise NotImplementedError("evaluate_Ni method defined in subclasses.")
  
  def Ni_centroid(self):
    raise NotImplementedError("Ni_centroid method defined in subclasses.")
  
  def evaluate_GNi(self, xi:np.ndarray):
    raise NotImplementedError("evaluate_GNi method defined in subclasses.")
  
  def GNi_centroid(self):
    raise NotImplementedError("GNi_centroid method defined in subclasses.")
  
  def evaluate_invJ(self, xi:np.ndarray, xe:np.ndarray):
    raise NotImplementedError("evaluate_invJ method defined in subclasses.")
  
  def evaluate_Jacobian(self, GNi:np.ndarray, xe:np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian matrix :math:`\\boldsymbol J` for a single element or multiple elements such that:

    .. math::
      J_{ij} = \\sum_k \\frac{\\partial N_k}{\\partial \\xi_j} x_{k,i}

    where :math:`\\partial_{\\xi_j} N_k` are the derivatives of the shape functions with respect to the reference coordinates,
    and :math:`x_{k,i}` are the physical coordinates of the node :math:`k`.

    :param numpy.ndarray GNi: Array of shape ``(nnodes, reference dim)`` containing the derivatives of the shape functions with respect to the reference coordinates.
    :param numpy.ndarray xe: Array of shape ``(nnodes, physical dim)`` for a single element, or ``(n_cells, nnodes, physical dim)`` for multiple elements, containing the physical coordinates of the nodes.
    :return: 
      The Jacobian matrix :math:`\\boldsymbol J` for the element(s). 
      For a single element, array of the shape ``(reference dim, physical dim)``. 
      For multiple elements, array of the shape ``(n_cells, reference dim, physical dim)``.
    :rtype: numpy.ndarray
    """
    t0 = perf_counter()
    GNi_idx = 'kj'
    xe_idx  = 'ki'
    J_idx   = 'ij'
    if xe.ndim == 3:
      xe_idx  = 'e'+xe_idx
      J_idx   = 'e'+J_idx
    if GNi.ndim == 3:
      GNi_idx = 'q'+GNi_idx
      J_idx   = 'q'+J_idx
    J = np.einsum(f'{GNi_idx},{xe_idx}->{J_idx}', GNi, xe)
    t1 = perf_counter()
    print(f"Computed Jacobian matrix of shape {J.shape} in {t1-t0:g} seconds")
    return J
  
  def evaluate_dNidx(self, invJ:np.ndarray, GNi:np.ndarray) -> np.ndarray:
    """
    Compute the physical derivatives of the shape functions with respect to the physical coordinates such that:

    .. math::
      \\frac{\\partial N_k}{\\partial x_i} = \\sum_j J^{-1}_{ji} \\frac{\\partial N_k}{\\partial \\xi_j}
    
    where :math:`J^{-1}_{ji}` are the components of the inverse Jacobian matrix (transposed), 
    and :math:`\\partial_{\\xi_j} N_k` are the derivatives of the shape functions with respect to the reference coordinates.

    :param numpy.ndarray invJ: Array of shape ``(reference dim, physical dim)`` for a single element, or ``(n_cells, reference dim, physical dim)`` for multiple elements, containing the inverse Jacobian matrix (transposed).
    :param numpy.ndarray GNi: Array of shape ``(nnodes, dim)`` containing the derivatives of the shape functions with respect to the reference coordinates.
    :return: 
      The physical derivatives of the shape functions with respect to the physical coordinates. 
      For a single element, array of the shape ``(nnodes, physical dim)``. 
      For multiple elements, array of the shape ``(n_cells, nnodes, physical dim)``.
    :rtype: numpy.ndarray
    """
    if invJ.ndim == 2:
      dNidx = np.matmul(invJ.T, GNi)
    elif invJ.ndim == 3:
      # Compute physical derivatives for all elements
      # GNi is (nnodes, dim), invJ is (n_cells, dim, dim)
      # We need: dNidx[e,k,i] = sum_j invJ[e,j,i] * GNi[k,j] (note the invJ transpose)
      # Result: (n_cells, nnodes, dim)
      dNidx = np.einsum('eji,kj->eki', invJ, GNi)
    return dNidx
  
  def evaluate_element_centroid(self, xe:np.ndarray) -> np.ndarray:
    """
    Compute the element centroid in physical coordinates such that:

    .. math::
      c_{i} = \\sum_k N_k x_{k,i}
    
    where :math:`N_k` are the shape functions evaluated at the element centroid,
    and :math:`x_{k,i}` are the physical coordinates of the node :math:`k`.

    :param numpy.ndarray xe: 
      Array of shape ``(nnodes, physical dim)`` for a single element, 
      or ``(n_cells, nnodes, physical dim)`` for multiple elements, 
      containing the physical coordinates of the nodes.
    :return: 
      The element centroid in physical coordinates. 
      For a single element, an array of shape ``(physical dim,)``. 
      For multiple elements, an array of shape ``(n_cells, physical dim)``.
    :rtype: numpy.ndarray
    """
    Ni = self.Ni_centroid()
    if xe.ndim == 2:
      centroid = np.dot(Ni, xe)
    elif xe.ndim == 3:
      centroid = np.einsum('k,eki->ei', Ni, xe)
    else:
      raise ValueError("xe must be 2D or 3D array.")
    return centroid

class Element2D(Element):
  """
  Class for 2D elements (e.g., triangles, quadrilaterals) representing domains in 2D space, :math:`\\mathbb R^2`.
  """
  def __init__(self):
    super().__init__()
    self.dim = 2
    return

  def evaluate_detJ(self, J:np.ndarray) -> np.ndarray:
    """
    Compute the determinant of the Jacobian matrix :math:`\\boldsymbol J` for a single element or multiple elements such that:

    .. math::
      \\det(\\boldsymbol J) = J_{00} J_{11} - J_{01} J_{10}

    :param numpy.ndarray J: Array of shape ``(2, 2)`` for a single element, or ``(n_cells, 2, 2)`` for multiple elements, containing the Jacobian matrix.
    :return: 
      The determinant of the Jacobian matrix :math:`\\det(\\boldsymbol J)` for the element(s). 
      For a single element, a float. 
      For multiple elements, an array of shape ``(n_cells,)``.
    :rtype: float or numpy.ndarray
    """
    if J.ndim < 2 or J.shape[-2:] != (2, 2):
      raise ValueError("J must have shape (..., 2, 2).")
    detJ:np.ndarray = J[...,0,0]*J[...,1,1] - J[...,0,1]*J[...,1,0]
    return detJ
  
  def evaluate_invJ(self, J:np.ndarray, detJ:float|np.ndarray) -> np.ndarray:
    """
    Compute the inverse of the Jacobian matrix :math:`\\boldsymbol J` for a single element or multiple elements such that:

    .. math::
      \\boldsymbol J^{-1} = \\frac{1}{\\det(\\boldsymbol J)} 
      \\begin{bmatrix} 
      J_{11} & -J_{01} \\\\
      -J_{10} & J_{00} 
      \\end{bmatrix}

    :param numpy.ndarray J: Array of shape ``(2, 2)`` for a single element, or ``(n_cells, 2, 2)`` for multiple elements, containing the Jacobian matrix.
    :param float or numpy.ndarray detJ: The determinant of the Jacobian matrix for the element(s). For a single element, a float. For multiple elements, an array of shape ``(n_cells,)``.
    :return: 
      The inverse of the Jacobian matrix :math:`\\boldsymbol J^{-1}` for the element(s). 
      For a single element, an array of shape ``(2, 2)``. 
      For multiple elements, an array of shape ``(n_cells, 2, 2)``.
    :rtype: numpy.ndarray
    """
    if J.ndim < 2 or J.shape[-2:] != (2, 2):
      raise ValueError("J must have shape (..., 2, 2).")
    J = np.zeros_like(J)
    invJ = np.zeros_like(J)
    invJ[..., 0, 0] =  J[..., 1, 1]
    invJ[..., 0, 1] = -J[..., 0, 1]
    invJ[..., 1, 0] = -J[..., 1, 0]
    invJ[..., 1, 1] =  J[..., 0, 0]
    invJ /= detJ[..., None, None]
    return invJ
  
class Element3D(Element):
  """
  Class for 3D elements (e.g., tetrahedra, hexahedra) representing domains in 3D space, :math:`\\mathbb R^3`.
  """
  def __init__(self):
    super().__init__()
    self.dim = 3
    return
  
  def evaluate_detJ(self, J:np.ndarray) -> float|np.ndarray:
    """
    Compute the determinant of the Jacobian matrix :math:`\\boldsymbol J` for a single element or multiple elements such that:

    .. math::
      \\det(\\boldsymbol J) = J_{00}(J_{11}J_{22} - J_{12}J_{21}) - J_{01}(J_{10}J_{22} - J_{12}J_{20}) + J_{02}(J_{10}J_{21} - J_{11}J_{20})

    :param numpy.ndarray J: Array of shape ``(..., 3, 3)`` containing the Jacobian matrix, where the last two dimensions are the matrix indices.
    :return: 
      The determinant of the Jacobian matrix :math:`\\det(\\boldsymbol J)` for the element(s). 
      For a single element, a float.
      For multiple elements, an array with shape ``J.shape[:-2]``.
    :rtype: float or numpy.ndarray
    """
    if J.ndim < 2 or J.shape[-2:] != (3, 3):
      raise ValueError("J must have shape (..., 3, 3).")
    t0 = perf_counter()
    detJ = (
      J[...,0,0]   * (J[...,1,1]*J[...,2,2] - J[...,1,2]*J[...,2,1])
      - J[...,0,1] * (J[...,1,0]*J[...,2,2] - J[...,1,2]*J[...,2,0])
      + J[...,0,2] * (J[...,1,0]*J[...,2,1] - J[...,1,1]*J[...,2,0])
    )
    t1 = perf_counter()
    print(f"Computed determinant of Jacobian matrix of shape {J.shape} in {t1-t0:g} seconds")
    return detJ
  
  def evaluate_invJ(self, J:np.ndarray, detJ:float|np.ndarray) -> np.ndarray:
    """
    Compute the inverse of the Jacobian matrix :math:`\\boldsymbol J` for a single element or multiple elements.

    :param numpy.ndarray J: Array of shape ``(3, 3)`` for a single element, or ``(n_cells, 3, 3)`` for multiple elements, containing the Jacobian matrix.
    :param float or numpy.ndarray detJ: The determinant of the Jacobian matrix for the element(s). For a single element, a float. For multiple elements, an array of shape ``(n_cells,)``.
    :return: 
      The inverse of the Jacobian matrix :math:`\\boldsymbol J^{-1}` for the element(s). 
      For a single element, an array of shape ``(3, 3)``. 
      For multiple elements, an array of shape ``(n_cells, 3, 3)``.
    :rtype: numpy.ndarray
    """
    if J.ndim < 2 or J.shape[-2:] != (3, 3):
      raise ValueError("J must have shape (..., 3, 3).")
    t0 = perf_counter()
    invJ = np.zeros_like(J)
    invJ[...,0,0] =  (J[...,1,1]*J[...,2,2]-J[...,1,2]*J[...,2,1])
    invJ[...,0,1] = -(J[...,0,1]*J[...,2,2]-J[...,0,2]*J[...,2,1])
    invJ[...,0,2] =  (J[...,0,1]*J[...,1,2]-J[...,0,2]*J[...,1,1])
    invJ[...,1,0] = -(J[...,1,0]*J[...,2,2]-J[...,1,2]*J[...,2,0])
    invJ[...,1,1] =  (J[...,0,0]*J[...,2,2]-J[...,0,2]*J[...,2,0])
    invJ[...,1,2] = -(J[...,0,0]*J[...,1,2]-J[...,0,2]*J[...,1,0])
    invJ[...,2,0] =  (J[...,1,0]*J[...,2,1]-J[...,1,1]*J[...,2,0])
    invJ[...,2,1] = -(J[...,0,0]*J[...,2,1]-J[...,0,1]*J[...,2,0])
    invJ[...,2,2] =  (J[...,0,0]*J[...,1,1]-J[...,0,1]*J[...,1,0])
    invJ /= detJ[..., None, None]
    t1 = perf_counter()
    print(f"Computed inverse of Jacobian matrix of shape {J.shape} in {t1-t0:g} seconds")
    return invJ
  
class SurfaceElement(Element):
  """
  Specialized class for 2D surface elements (triangles, quadrilaterals) embedded in 3D space, :math:`\\mathbb R^3`.
  """
  def __init__(self):
    super().__init__()
    self.dim = 3
    return
  
  def normal_vector_nonu(self, J:np.ndarray) -> np.ndarray:
    """
    Compute the **non-unit** normal vector to the surface element defined by the Jacobian matrix :math:`\\boldsymbol J` 
    for a single element or multiple elements.
    The normal vector :math:`\\mathbf n` is computed as the cross product of the two tangent vectors defined by the columns of the Jacobian matrix:

    .. math::
      \\begin{split}
        \\boldsymbol J &= 
        \\begin{bmatrix} 
          \\mathbf t_{\\xi} & \\mathbf t_{\\eta} 
        \\end{bmatrix} \\in \\mathbb R^{3 \\times 2} \\\\
        \\mathbf n &= \\mathbf t_{\\xi} \\times \\mathbf t_{\\eta}
      \\end{split}

    where :math:`\\mathbf t_{\\xi}` and :math:`\\mathbf t_{\\eta}` are the tangent vectors defined by 
    the columns of the Jacobian matrix, and :math:`\\times` denotes the cross product.

    :param numpy.ndarray J: Array of shape ``(3, 2)`` for a single element, or ``(n_cells, 3, 2)`` for multiple elements, containing the Jacobian matrix.
    :return: 
      The non-unit normal vector to the surface element. 
      For a single element, an array of shape ``(3,)``. 
      For multiple elements, an array of shape ``(n_cells, 3)``.
    :rtype: numpy.ndarray
    """
    if J.ndim == 2:
      # Single element
      tangent1 = J[:,0]
      tangent2 = J[:,1]
      normal = np.cross(tangent1, tangent2)
    elif J.ndim == 3:
      # Multiple elements
      tangent1 = J[:,:,0] # shape (n_elements, 3)
      tangent2 = J[:,:,1] # shape (n_elements, 3)
      # cross product
      normal = np.empty_like(tangent1)
      normal[:,0] = tangent1[:,1]*tangent2[:,2] - tangent1[:,2]*tangent2[:,1]
      normal[:,1] = tangent1[:,2]*tangent2[:,0] - tangent1[:,0]*tangent2[:,2]
      normal[:,2] = tangent1[:,0]*tangent2[:,1] - tangent1[:,1]*tangent2[:,0]
    else:
      raise ValueError("J must be 2D or 3D array.")
    return normal
  
  def normal_vector(self, J:np.ndarray) -> np.ndarray:
    """
    Compute the unit normal vector to the surface element.
    The non-unit normal vector is computed using the method :py:meth:`normal_vector_nonu <stagpyviz.SurfaceElement.normal_vector_nonu>`, 
    and then normalized by its magnitude (which is the determinant of the Jacobian matrix) to obtain the unit normal vector.

    :param numpy.ndarray J: Array of shape ``(3, 2)`` for a single element, or ``(n_cells, 3, 2)`` for multiple elements, containing the Jacobian matrix.
    :return: 
      The unit normal vector to the surface element. 
      For a single element, an array of shape ``(3,)``. 
      For multiple elements, an array of shape ``(n_cells, 3)``.
    :rtype: numpy.ndarray
    """
    normal = self.normal_vector_nonu(J)
    norm   = self.evaluate_detJ(J)
    if J.ndim == 2:
      normal /= norm
    elif J.ndim == 3:
      normal /= norm[:, np.newaxis]
    return normal
  
  def evaluate_detG(self, J:np.ndarray) -> float|np.ndarray:
    """
    Compute the determinant of the metric tensor :math:`\\boldsymbol G` for a single element or multiple elements such that:

    .. math::
      \\det(\\boldsymbol G) = \\mathbf n \\cdot \\mathbf n

    where :math:`\\mathbf n` is the non-unit normal vector to the surface element computed using the method :py:meth:`normal_vector_nonu <stagpyviz.SurfaceElement.normal_vector_nonu>`.

    :param numpy.ndarray J: Array of shape ``(3, 2)`` for a single element, or ``(n_cells, 3, 2)`` for multiple elements, containing the Jacobian matrix.
    :return: 
      The determinant of the metric tensor :math:`\\det(\\boldsymbol G)` for the element(s). 
      For a single element, a float. 
      For multiple elements, an array of shape ``(n_cells,)``.
    :rtype: float or numpy.ndarray
    """
    normal = self.normal_vector_nonu(J)
    if J.ndim == 2:
      detG = np.dot(normal, normal)
    elif J.ndim == 3:
      detG = np.einsum('ei,ei->e', normal, normal)
    else:
      raise ValueError("J must be 2D or 3D array.")
    return detG

  def evaluate_detJ(self, J:np.ndarray) -> float|np.ndarray:
    """
    Compute the determinant of the Jacobian matrix :math:`\\boldsymbol J` for a surface element such that:

    .. math::
      \\det(\\boldsymbol J) = \\sqrt{\\det(\\boldsymbol G)}

    where :math:`\\det(\\boldsymbol G)` is the determinant of the metric tensor computed using the method :py:meth:`evaluate_detG <stagpyviz.SurfaceElement.evaluate_detG>`.

    :param numpy.ndarray J: Array of shape ``(3, 2)`` for a single element, or ``(n_cells, 3, 2)`` for multiple elements, containing the Jacobian matrix.
    :return: 
      The determinant of the Jacobian matrix for the element(s). 
      For a single element, a float. 
      For multiple elements, an array of shape ``(n_cells,)``.
    :rtype: float or numpy.ndarray
    """
    detG = self.evaluate_detG(J)
    detJ = np.sqrt(detG)
    return detJ

  def evaluate_metric_tensor(self, J:np.ndarray) -> np.ndarray:
    """
    Compute the metric tensor :math:`\\boldsymbol G` for a surface element such that:

    .. math::
      G_{ij} = \\mathbf t_{\\xi_i} \\cdot \\mathbf t_{\\xi_j}

    where :math:`\\mathbf t_{\\xi_i}` and :math:`\\mathbf t_{\\xi_j}` are the tangent vectors 
    defined by the columns of the Jacobian matrix.
    
    :param numpy.ndarray J: Array of shape ``(3, 2)`` for a single element, or ``(n_cells, 3, 2)`` for multiple elements, containing the Jacobian matrix.
    :return: 
      The metric tensor :math:`\\boldsymbol G` for the element(s). 
      For a single element, an array of shape ``(2, 2)``. 
      For multiple elements, an array of shape ``(n_cells, 2, 2)``.
    :rtype: numpy.ndarray
    """
    if J.ndim == 2:
      G = np.matmul(J.T, J)
    elif J.ndim == 3:
      G = np.einsum('eki,ekj->eij', J, J)
    else:
      raise ValueError("J must be 2D or 3D array.")
    return G

  def evaluate_dNidx(self, J:np.ndarray, GNi:np.ndarray) -> np.ndarray:
    """
    Compute the physical derivatives of the shape functions with respect to the physical coordinates for a surface element such that:

    .. math::
      \\frac{\\partial N_k}{\\partial x_i} = 
      \\sum_l \\left( \\sum_m J_{im} G^{-1}_{ml} \\right) \\frac{\\partial N_k}{\\partial \\xi_l}

    where :math:`J_{im}` are the components of the Jacobian matrix, 
    :math:`G^{-1}_{ml}` are the components of the inverse of the metric tensor, 
    and :math:`\\partial_{\\xi_l} N_k` are the derivatives of the shape functions with respect to the reference coordinates.
    
    :param numpy.ndarray J: Array of shape ``(3, 2)`` for a single element, or ``(n_cells, 3, 2)`` for multiple elements, containing the Jacobian matrix.
    :param numpy.ndarray GNi: Array of shape ``(nnodes, 2)`` containing the derivatives of the shape functions with respect to the reference coordinates.
    :return:
      The physical derivatives of the shape functions with respect to the physical coordinates.
      For a single element, an array of shape ``(nnodes, 3)``.
      For multiple elements, an array of shape ``(n_cells, nnodes, 3)``.
    :rtype: numpy.ndarray

    """
    G = self.evaluate_metric_tensor(J)
    detG = self.evaluate_detG(J)
    invG = Element2D().evaluate_invJ(G, detG)
    # Compute physical derivatives dN/dx = J * invG * GNi
    if J.ndim == 2:
      M = np.einsum('ik,kj->ij', J, invG)  # shape (3,2)
      dNidx = np.einsum('ji,ki->kj', M, GNi)  # shape (nnodes, 3)
    elif J.ndim == 3:
      M = np.einsum('eik,ekj->eij', J, invG)  # shape (n_elements, 3, 2)
      dNidx = np.einsum('eji,ki->ekj', M, GNi)  # shape (n_elements, nnodes, 3)
    return dNidx