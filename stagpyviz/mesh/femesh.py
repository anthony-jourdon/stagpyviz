import numpy as np
import pyvista as pvs
from time import perf_counter

try:
  from ..elements import Element
except ImportError:
  from stagpyviz.elements import Element

class FEMesh(pvs.PointGrid, pvs.DataSet):
  """
  Base class to represent meshes constructed from elements available 
  in the :py:class:`Element <stagpyviz.Element>` class.

  Implements operations common to all meshes, such as computing the elements-to-vertices mapping and
  computing gradients and integrals of fields.

  """
  def __init__(self, *args, deep:bool=False, **kwargs):
    super().__init__(*args, deep=deep, **kwargs)
    # Keep the element selected by subclasses (e.g., ShellMesh, YinYangMesh).
    # Fallback to the generic Element only when nothing was set upstream.
    if not hasattr(self, "elements") or self.elements is None:
      self.elements:Element = kwargs.pop("elements", Element())
    self.nodes_per_el = self.elements.basis_per_el

    self._elidx:np.ndarray|None = None
    self._centroids:np.ndarray|None = None
    self._cells_Jacobian:np.ndarray|None = None
    self._cells_volume:np.ndarray|None = None
    return

  @property
  def elidx(self) -> np.ndarray:
    """
    Elements-to-vertices mapping.
    Array of shape ``(number_of_cells, nodes_per_el)`` 
    containing the indices of the vertices of each element.
    """
    if self._elidx is None:
      self.create_elements2vertices()
    return self._elidx

  @property
  def centroids(self) -> np.ndarray:
    """
    Coordinates of the centroids of the elements in the mesh.
    Array of shape ``(number_of_cells, 3)``.
    """
    if self._centroids is None:
      elidx = self.elidx
      elcoords = self.points[elidx, :]
      Ni_centroid = self.elements.Ni_centroid()
      self._centroids = np.einsum('k,eki->ei',Ni_centroid,elcoords)
      #self.cell_centers().points
    return self._centroids

  def create_elements2vertices(self):
    serr  = "Method to create the elements-to-vertices mapping is not implemented for this base class.\n"
    serr += "Use subclasses to create and access the elements-to-vertices mapping."
    raise NotImplementedError(serr)

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

  @property
  def cells_Jacobian(self) -> np.ndarray:
    """
    3D array of shape ``(number_of_cells, 3, 3)`` containing the Jacobian matrices 
    of the transformation from the reference element to the physical element for each cell in the mesh.
    """
    if self._cells_Jacobian is None:
      elidx    = self.elidx
      elcoords = self.points[ elidx, : ]  # (number_of_cells, nodes_per_el, dim)
      # shape functions gradient evaluated at the centroid of the reference element
      GNi = self.elements.GNi_centroid() 
      J = self.elements.evaluate_Jacobian(GNi, elcoords)  # (number_of_cells, dim, dim)
      self._cells_Jacobian = J
    return self._cells_Jacobian

  @property
  def cells_volume(self) -> np.ndarray:
    """
    Array of shape ``(number_of_cells,)`` containing the volumes of the elements in the mesh.
    Volume computed using :py:meth:`Element.evaluate_volume() <stagpyviz.Element.evaluate_volume>`.
    """
    if self._cells_volume is None:
      elidx     = self.elidx
      el_coords = self.points[ elidx, : ]
      self._cells_volume = self.elements.evaluate_volume(el_coords,rule=1)
    return self._cells_volume

  def compute_gradient(self, field:np.ndarray) -> np.ndarray:
    """
    Compute the Cartesian gradient of a scalar field defined on the mesh 
    using the shape functions of the mesh elements.

    :param numpy.ndarray field:
      The scalar field defined on the mesh for which the gradient is to be computed.
      The array should have shape ``(number_of_points,)``.
    :return:
      The gradient of the field at the centroids of the elements.
      The array has shape ``(number_of_cells, 3)``, 
      where the last dimension corresponds to the x, y and z components of the gradient.
    :rtype: numpy.ndarray
    
    """
    t0 = perf_counter()
    elidx = self.elidx
    GNi   = self.elements.GNi_centroid()
    J     = self.cells_Jacobian  # (number_of_cells, 3, 3)
    detJ  = self.elements.evaluate_detJ(J)
    invJ  = self.elements.evaluate_invJ(J, detJ)
    dNdx  = self.elements.evaluate_dNidx(invJ, GNi)  # (number_of_cells, nodes_per_el, 3)
    # Gather field values at element nodes
    field_el = field[ elidx ]  # (number_of_cells, nodes_per_el)
    grad_field = np.einsum('eki,ek...->ei...', dNdx, field_el)
    t1 = perf_counter()
    print(f"Gradient computation performed in {t1-t0:g} seconds")
    return grad_field

  # only cell fields and 1 point quadrature rule is implemented here as it is the same 
  # for all element types, but point fields with higher order quadrature rules 
  # are implemented in the specific mesh classes
  def integrate_cell_field(self, field:np.ndarray) -> np.ndarray:
    if not self.is_cell_field(field):
      raise ValueError(f"Field of shape {field.shape} must match number of cells {self.number_of_cells} to be integrated with integrate_cell_field()")
    return np.einsum('e...,e->e...', field, self.cells_volume)

  def integrate_1pt_rule(self, field:np.ndarray) -> np.ndarray:
    if self.is_cell_field(field):
      return self.integrate_cell_field(field)
    elif self.is_point_field(field):
      t0 = perf_counter()
      elidx = self.elidx
      el_coords = self.points[ elidx, : ]  # (number_of_cells, nodes_per_el, 3)
      field_el = field[ elidx ]  # (number_of_cells, nodes_per_el)
      t1 = perf_counter()
      integral = self.elements.integrate_field(el_coords, field_el, rule=1, Jacobian=self.cells_Jacobian)
      print(f"Field integrated with 1 point rule in {t1-t0:g} seconds")
      return integral
    else:
      raise ValueError(f"Field of shape {field.shape} must be either a point field with shape ({self.number_of_points},) or a cell field with shape ({self.number_of_cells},) to be integrated with integrate_1pt_rule()")

  def integrate_over_cell(self,field:np.ndarray, **kwargs) -> np.ndarray:
    """
    Integrate a field over each cell of the mesh.
    Default integration rule is 1 point quadrature rule.
    For higher order integration, see mesh specific documentation.
    """
    return self.integrate_1pt_rule(field)

  def point_field_to_cell_field(self, field:np.ndarray) -> np.ndarray:
    """
    Transform a point field to a cell field by interpolating the nodal values at elements centroid.

    :param numpy.ndarray field: 
      The field to transform. Shape should be ``(number_of_points,)``.
      If a cell field is provided, it will be returned as is.
    :return: The transformed cell field. Shape is ``(number_of_cells,)``.
    :rtype: numpy.ndarray
    """
    elidx = self.elidx
    if self.is_cell_field(field):
      return field
    elif self.is_point_field(field):
      t0 = perf_counter()
      # Get the field values at the nodes of each element
      field_el = field[ elidx ]  # (number_of_cells, nodes_per_el)
      Ni_centroid = self.elements.Ni_centroid()  # (nodes_per_el,)
      # Compute the average value of the field at the nodes of each element
      field_centroid = np.einsum('k,ek...->e...', Ni_centroid, field_el)  # (number_of_cells,)
      t1 = perf_counter()
      print(f"Field values at element centroids computed in {t1-t0:g} seconds")
      return field_centroid
    else:
      raise ValueError(f"Input field has incompatible shape. Mesh ncells: {self.number_of_cells}, npoints: {self.number_of_points}, field shape: {field.shape}")
    
  def cell_data_to_point_data(self, pass_cell_data:bool=False) -> None:
    """
    Convert all cell data to point data on the mesh.
    
    :param bool pass_cell_data:
      If True, the cell data will be kept in the mesh after conversion.
      If False, the cell data will be removed from the mesh after conversion.
    """
    point_mesh = super().cell_data_to_point_data()
    for f in point_mesh.point_data:
      if pass_cell_data == False and f in self.cell_data:
        self.cell_data.pop(f)
      self.point_data[f] = point_mesh.point_data[f]
    return
  
  def point_data_to_cell_data(self, pass_point_data:bool=False) -> None:
    """
    Convert all point data to cell data on the mesh.
    
    :param bool pass_point_data:
      If True, the point data will be kept in the mesh after conversion.
      If False, the point data will be removed from the mesh after conversion.
    """
    cell_mesh = super().point_data_to_cell_data()
    for f in cell_mesh.cell_data:
      if pass_point_data == False and f in self.point_data:
        self.point_data.pop(f)
      self.cell_data[f] = cell_mesh.cell_data[f]
    return