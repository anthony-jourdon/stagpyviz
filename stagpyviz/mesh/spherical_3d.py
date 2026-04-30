import pyvista as pvs
import numpy as np
from time import perf_counter

class UnstructuredSphere(pvs.UnstructuredGrid):
  """
  Unstructured grid class for spherical meshes. 
  Provides methods for Cartesian - Spherical coordinates transformations, as well as vector and gradient transformations.
  Inherits from `pyvista.UnstructuredGrid`_, so all methods and properties of the parent class are available.

  :Attributes:

  .. py:attribute:: points_spherical

    The points of the mesh in spherical coordinates. 
    Shape is ``(number_of_points, 3)`` with columns corresponding to ``(radius, colatitude, longitude)``.
    
    :type: numpy.ndarray

  .. py:attribute:: centroids

    The centroids of the cells in Cartesian coordinates.
    Shape is ``(number_of_cells, 3)`` with columns corresponding to ``(x, y, z)``.
    
    :type: numpy.ndarray

  .. py:attribute:: centroids_spherical

    The centroids of the cells in spherical coordinates.
    Shape is ``(number_of_cells, 3)`` with columns corresponding to ``(radius, colatitude, longitude)``.
    
    :type: numpy.ndarray

  :Methods:

  """
  def __init__(self, *args, deep:bool=False, **kwargs) -> None:
    super().__init__(*args, deep=deep, **kwargs)
    self._points_spherical    = None
    self._centroids           = None
    self._centroids_spherical = None
    return
  
  def is_point_field(self, field:np.ndarray) -> bool:
    """
    Check if the input field is a point field.

    :param numpy.ndarray field: The field to check.
    :return: True if the field is a point field, False otherwise.
    :rtype: bool
    """
    if field.shape[0] == self.number_of_points:
      return True
    return False
  
  def is_cell_field(self, field:np.ndarray) -> bool:
    """
    Check if the input field is a cell field.

    :param numpy.ndarray field: The field to check.
    :return: True if the field is a cell field, False otherwise.
    :rtype: bool
    """
    if field.shape[0] == self.number_of_cells:
      return True
    return False
  
  def create_point_field(self, bs:int=1, dtype:np.dtype=np.float64) -> np.ndarray:
    """
    Create a point field initialized with zeros.
    
    :param int bs: The number of components of the field. Default is 1 (scalar field).
    :param numpy.dtype dtype: The data type of the field. Default is ``np.float64``.
    :return: 
      A point field initialized with zeros.
      Scalar fields have shape ``(number_of_points,)``, vector fields have shape ``(number_of_points, bs)``.
    :rtype: numpy.ndarray
    """
    if bs == 1:
      return np.zeros( (self.number_of_points), dtype=dtype )
    else:
      return np.zeros( (self.number_of_points, bs), dtype=dtype )
    
  def create_cell_field(self, bs:int=1, dtype:np.dtype=np.float64) -> np.ndarray:
    """
    Create a cell field initialized with zeros.

    :param int bs: The number of components of the field. Default is 1 (scalar field).
    :param numpy.dtype dtype: The data type of the field. Default is ``np.float64``.
    :return: 
      A cell field initialized with zeros.
      Scalar fields have shape ``(number_of_cells,)``, vector fields have shape ``(number_of_cells, bs)``.
    :rtype: numpy.ndarray
    """
    if bs == 1:
      return np.zeros( (self.number_of_cells), dtype=dtype )
    else:
      return np.zeros( (self.number_of_cells, bs), dtype=dtype )
    
  def cartesian_to_spherical(
      self,
      x:np.ndarray|float,
      y:np.ndarray|float,
      z:np.ndarray|float
    ) -> tuple[
      np.ndarray|float,
      np.ndarray|float,
      np.ndarray|float
    ]:
    """
    Transform Cartesian coordinates to Spherical coordinates such that:

    .. math::
      \\begin{split}
        r &= \\sqrt{x^2 + y^2 + z^2} \\\\
        \\theta &= \\arctan\\left(\\frac{\\sqrt{x^2 + y^2}}{z}\\right) \\\\
        \\phi &= \\arctan\\left(\\frac{y}{x}\\right)
      \\end{split}

    :param numpy.ndarray|float x: The x coordinate(s).
    :param numpy.ndarray|float y: The y coordinate(s).
    :param numpy.ndarray|float z: The z coordinate(s).
    :return: 
      A tuple containing the spherical coordinates ``(R, theta, phi)``.
      ``R`` is the radial distance, 
      ``theta`` is the colatitude (angle from the :math:`z`-axis), 
      and ``phi`` is the longitude (angle from the :math:`x`-axis in the :math:`xy`-plane).
    :rtype: tuple[numpy.ndarray|float, numpy.ndarray|float, numpy.ndarray|float]
    """
    R     = np.sqrt(x**2+y**2+z**2)
    colat = np.arctan2(np.sqrt(x**2+y**2),z)
    lon   = np.arctan2(y,x)
    return R,colat,lon

  def spherical_to_cartesian(
      self,
      R:np.ndarray|float, 
      lat:np.ndarray|float, 
      lon:np.ndarray|float
    ) -> tuple[
      np.ndarray|float,
      np.ndarray|float,
      np.ndarray|float
    ]:
    """
    Transform Spherical coordinates to Cartesian coordinates such that:

    .. math::
      \\begin{split}
        x &= r \\sin(\\theta) \\cos(\\phi) \\\\
        y &= r \\sin(\\theta) \\sin(\\phi) \\\\
        z &= r \\cos(\\theta)
      \\end{split}

    :param numpy.ndarray|float R: The radial distance(s).
    :param numpy.ndarray|float lat: The colatitude(s) (angle(s) from the :math:`z`-axis).
    :param numpy.ndarray|float lon: The longitude(s) (angle(s) from the :math:`x`-axis in the :math:`xy`-plane).
    :return: 
      A tuple containing the Cartesian coordinates ``(x, y, z)``.
    :rtype: tuple[numpy.ndarray|float, numpy.ndarray|float, numpy.ndarray|float]
    """
    x = R*np.cos(lat)*np.cos(lon)
    y = R*np.cos(lat)*np.sin(lon)
    z = R*np.sin(lat)
    return x,y,z
  
  @property 
  def points_spherical(self) -> np.ndarray:
    if self._points_spherical is None:
      self._points_spherical = np.zeros( (self.number_of_points, 3), dtype=np.float64 )
      self._points_spherical[:,0], self._points_spherical[:,1], self._points_spherical[:,2] = self.cartesian_to_spherical(
        self.points[:,0],
        self.points[:,1],
        self.points[:,2]
      )
    return self._points_spherical
  
  @property
  def centroids(self) -> np.ndarray:
    if self._centroids is None:
      elidx = self.cell_connectivity.reshape((self.number_of_cells, self.elements.basis_per_el))
      elcoords = self.points[elidx, :]
      Ni_centroid = self.elements.Ni_centroid()
      self._centroids = np.einsum('k,eki->ei',Ni_centroid,elcoords)
      #self.cell_centers().points
    return self._centroids

  @property
  def centroids_spherical(self) -> np.ndarray:
    centroids = self.centroids
    if self._centroids_spherical is None:
      self._centroids_spherical = np.zeros( (self.number_of_cells, 3), dtype=np.float64 )
      self._centroids_spherical[:,0], self._centroids_spherical[:,1], self._centroids_spherical[:,2] = self.cartesian_to_spherical(
        centroids[:,0],
        centroids[:,1],
        centroids[:,2]
      )
    return self._centroids_spherical
  
  def rotation_matrix(self,theta:np.ndarray,phi:np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix for Cartesian - Spherical vector transformations at given 
    colatitude (:math:`\\theta`) and longitude (:math:`\\phi`) angles such that:

    .. math::
      \\boldsymbol{R} = \\begin{bmatrix}
        \\sin(\\theta) \\cos(\\phi) & \\sin(\\theta) \\sin(\\phi) & \\cos(\\theta) \\\\
        \\cos(\\theta) \\cos(\\phi) & \\cos(\\theta) \\sin(\\phi) & -\\sin(\\theta) \\\\
        -\\sin(\\phi) & \\cos(\\phi) & 0
      \\end{bmatrix}

    :param numpy.ndarray theta: The colatitude angles (angle from the :math:`z`-axis).
    :param numpy.ndarray phi: The longitude angles (angle from the :math:`x`-axis in the :math:`xy`-plane).
    :return: 
      The rotation matrix for Cartesian - Spherical vector transformations at the given angles. 
      Shape is ``(N, 3, 3)`` where N is the length of the input angle arrays.
    :rtype: numpy.ndarray

    """
    R = np.zeros((theta.shape[0],3,3), dtype=np.float64)
    R[:,0,0] =  np.sin(theta)*np.cos(phi)
    R[:,0,1] =  np.sin(theta)*np.sin(phi)
    R[:,0,2] =  np.cos(theta)

    R[:,1,0] = np.cos(theta)*np.cos(phi)
    R[:,1,1] = np.cos(theta)*np.sin(phi)
    R[:,1,2] = -np.sin(theta)

    R[:,2,0] = -np.sin(phi)
    R[:,2,1] =  np.cos(phi)
    R[:,2,2] =  0.0
    return R

  def rotation_matrix_vertices(self) -> np.ndarray:
    """
    Computes the rotation matrix for Cartesian - Spherical vector transformations at the vertices of the mesh.
    Calls :py:meth:`rotation_matrix <stagpyviz.UnstructuredSphere.rotation_matrix>` using the attribute :py:attr:`points_spherical`.
    """
    return self.rotation_matrix(self.points_spherical[:,1],self.points_spherical[:,2])
  
  def rotation_matrix_centroids(self) -> np.ndarray:
    """
    Computes the rotation matrix for Cartesian - Spherical vector transformations at the centroids of the cells of the mesh.
    Calls :py:meth:`rotation_matrix <stagpyviz.UnstructuredSphere.rotation_matrix>` using the attribute :py:attr:`centroids_spherical`.
    """
    return self.rotation_matrix(self.centroids_spherical[:,1],self.centroids_spherical[:,2])
  
  def vector_cartesian_to_spherical(self, v_cartesian:np.ndarray) -> np.ndarray:
    """
    Transform a vector field from Cartesian to Spherical coordinates such that:

    .. math::
      \\mathbf{v}_{r} = \\boldsymbol{R} \\mathbf{v}_{x}
    
    where :math:`\\mathbf{v}_{r}` is the vector field in spherical coordinates,
    :math:`\\mathbf{v}_{x}` is the vector field in Cartesian coordinates,
    and :math:`\\boldsymbol{R}` is the rotation matrix of the transformomation between Cartesian and Spherical coordinates 
    (see :py:meth:`rotation_matrix <stagpyviz.UnstructuredSphere.rotation_matrix>`).

    :param numpy.ndarray v_cartesian: 
      The vector field in Cartesian coordinates. 
      Shape should be ``(N, 3)`` where N can be either the number of points or the number of cells of the mesh.
    :return: The vector field in Spherical coordinates. Shape is the same as the input field.
    :rtype: numpy.ndarray
    """
    t0 = perf_counter()
    if self.is_point_field(v_cartesian):
      R = self.rotation_matrix_vertices()
    elif self.is_cell_field(v_cartesian):
      R = self.rotation_matrix_centroids()
    else:
      raise ValueError(f"Input vector field has incompatible shape. Mesh ncells: {self.number_of_cells}, npoints: {self.number_of_points}, field shape: {v_cartesian.shape}")
    v_spherical = np.einsum('ijk,ik->ij', R, v_cartesian)
    t1 = perf_counter()
    print(f"cartesian to spherical vector transformation performed in {t1-t0:g} seconds")
    return v_spherical
  
  def vector_spherical_to_cartesian(self, v_spherical:np.ndarray) -> np.ndarray:
    """
    Transform a vector field from Spherical to Cartesian coordinates such that:

    .. math::
      \\mathbf{v}_{x} = \\boldsymbol{R}^T \\mathbf{v}_{r}

    where :math:`\\mathbf{v}_{x}` is the vector field in Cartesian coordinates,
    :math:`\\mathbf{v}_{r}` is the vector field in spherical coordinates,
    and :math:`\\boldsymbol{R}^T` is the transpose of the rotation matrix of the transformomation between 
    Cartesian and Spherical coordinates (see :py:meth:`rotation_matrix <stagpyviz.UnstructuredSphere.rotation_matrix>`).

    :param numpy.ndarray v_spherical: 
      The vector field in Spherical coordinates. 
      Shape should be ``(N, 3)`` where N can be either the number of points or the number of cells of the mesh.
    :return: The vector field in Cartesian coordinates. Shape is the same as the input field.
    :rtype: numpy.ndarray

    """
    t0 = perf_counter()
    if self.is_point_field(v_spherical):
      R = self.rotation_matrix_vertices()
    elif self.is_cell_field(v_spherical):
      R = self.rotation_matrix_centroids()
    else:
      raise ValueError(f"Input vector field has incompatible shape. Mesh ncells: {self.number_of_cells}, npoints: {self.number_of_points}, field shape: {v_spherical.shape}")
    R_T = np.transpose(R, (0,2,1))
    v_cartesian = np.einsum('ijk,ik->ij', R_T, v_spherical)
    t1 = perf_counter()
    print(f"spherical to cartesian vector transformation performed in {t1-t0:g} seconds")
    return v_cartesian

  def point_field_to_cell_field(self, field:np.ndarray) -> np.ndarray:
    """
    Transform a point field to a cell field by interpolating the nodal values at elements centroid.

    :param numpy.ndarray field: 
      The field to transform. Shape should be ``(number_of_points,)``.
      If a cell field is provided, it will be returned as is.
    :return: The transformed cell field. Shape is ``(number_of_cells,)``.
    :rtype: numpy.ndarray
    """
    elidx = self.cell_connectivity.reshape((self.number_of_cells, self.elements.basis_per_el))
    if self.is_cell_field(field):
      return field
    elif self.is_point_field(field):
      t0 = perf_counter()
      # Get the field values at the nodes of each element
      elidx = self.cell_connectivity.reshape((self.number_of_cells, self.elements.basis_per_el))
      Ni_centroid = self.elements.Ni_centroid()  # (nodes_per_el,)
      field_el = field[ elidx ]  # (number_of_cells, nodes_per_el)
      # Compute the average value of the field at the nodes of each element
      field_centroid = np.einsum('k,ek->e', Ni_centroid, field_el)  # (number_of_cells,)
      t1 = perf_counter()
      print(f"Field values at element centroids computed in {t1-t0:g} seconds")
      return field_centroid
    else:
      raise ValueError(f"Input field has incompatible shape. Mesh ncells: {self.number_of_cells}, npoints: {self.number_of_points}, field shape: {field.shape}")