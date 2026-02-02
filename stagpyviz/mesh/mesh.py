import pyvista as pvs
import numpy as np
from functools import cached_property
from time import perf_counter

try:
    from ..elements.q1_2d import Q1_2D
except ImportError:
  from elements.q1_2d import Q1_2D

class Hex2DMesh(pvs.StructuredGrid):

  def __init__(self, uinput=None, y=None, z=None, *args, deep: bool = False, **kwargs):
    super().__init__(uinput, y, z, *args, deep=deep, **kwargs)

    self.basis_per_el = 4
    self.dimensions = kwargs.get('dimensions', self.dimensions)
    self.nodes_dimensions = np.array([
      self.dimensions[0],
      self.dimensions[1],
    ], dtype=np.int64)
    self.cells_dimensions = np.array([
      self.dimensions[0]-1,
      self.dimensions[1]-1,
    ], dtype=np.int64)

    self.elements:Q1_2D = Q1_2D()
    self.elidx = None
    self.mesh_cell2point:pvs.StructuredGrid = None
    return

  def is_point_field(self, field:np.ndarray):
    if field.shape[0] == self.number_of_points:
      return True
    return False

  def is_cell_field(self, field:np.ndarray):
    if field.shape[0] == self.number_of_cells:
      return True
    return False
  
  def create_e2v(self):
    self.elidx = np.zeros((self.number_of_cells, self.basis_per_el), dtype=np.int64)
    e = 0
    for j in range(self.cells_dimensions[1]):
      for i in range(self.cells_dimensions[0]):
        n0 = j * self.nodes_dimensions[0] + i
        n1 = n0 + 1
        n2 = n0 + self.nodes_dimensions[0]
        n3 = n0 + self.nodes_dimensions[0] + 1
        self.elidx[e, :] = [n0, n1, n2, n3]
        e += 1
    return
  
  @property
  def element2vertex(self):
    if self.elidx is None:
      self.create_e2v()
    return self.elidx
  
  @cached_property
  def centroids(self):
    _centroids = self.cell_centers().points
    return _centroids
  
  def compute_gradient(self, field:np.ndarray) -> np.ndarray:
    # Get element connectivity: shape (n_cells, 4)
    elidx = self.element2vertex
    # Extract coordinates for all elements: shape (n_cells, 4, 2)
    # mesh.points is (n_points, 3) but we only need x,y
    elcoor = self.points[elidx, :2]  # (n_cells, 4, 2)
    # Shape function derivatives at centroid
    GNi = self.elements.GNi_centroid()
    # Compute Jacobian for all elements: J = xe^T @ GNi
    Jac = self.elements.evaluate_Jacobian(GNi, elcoor)
    # Compute determinant: shape (n_cells,)
    detJ = self.elements.evaluate_detJ(Jac)
    # Compute inverse Jacobian for all elements: shape (n_cells, 2, 2)
    invJ = self.elements.evaluate_invJ(Jac, detJ)
    # Compute dN/dx = GNi @ invJ^T for all elements
    dNdx = self.elements.evaluate_dNidx(invJ, GNi)
    # Get the number of components in the field 
    if len(field.shape) == 1: dof = 1
    else:                     dof = field.shape[1]

    grad_f = []
    for d in range(dof):
      # Extract field values at element nodes: shape (n_cells, 4)
      if dof == 1:
        el_field = field[elidx]  # (n_cells, 4)
      else:
        el_field = field[elidx, d]  # (n_cells, 4)
      # Compute gradient: grad = sum over nodes of dNdx * field
      # dNdx is (n_cells, 4, 2), el_field is (n_cells, 4)
      # Result: (n_cells, 2)
      grad_f.append(np.einsum('eik,ei->ek', dNdx, el_field))
    if dof == 1:
      grad_f = grad_f[0]
    return grad_f
  
  def cell_field_to_point_field(self, field_name:str|None=None, field:np.ndarray|None=None) -> np.ndarray:
    # Neither field nor field_name provided => Error
    if field is None and field_name is None:
      raise ValueError("Either field_name or field or both must be provided.")
    # Field provided, use it in priority
    if field is not None:
      fname = field_name if field_name is not None else 'unnamed_field'
      if not self.is_cell_field(field):
        if self.is_point_field(field):
          return field
        else:
          raise ValueError(f"Provided field of shape {field.shape} does not match number of cells ({self.number_of_cells}) or points ({self.number_of_points}).")
      self.cell_data[fname] = field
      # Average cell data to point data
      self.mesh_cell2point = self.cell_data_to_point_data()
      return self.mesh_cell2point[fname]
    # Field not provided, get from mesh using field_name
    else:
      # Ensure the averaging to points has been done
      if self.mesh_cell2point is None:
        self.mesh_cell2point = self.cell_data_to_point_data()
      if field_name not in self.cell_data and field_name not in self.point_data:
        raise ValueError(f"Field '{field_name}' not found in mesh, consider passing it as the 'field' argument.")
      # Already a point field
      if field_name in self.point_data:
        return self.point_data[field_name]
      if field_name in self.cell_data and field_name not in self.mesh_cell2point.point_data:
        # The mesh_cell2point needs an update
        self.mesh_cell2point = self.cell_data_to_point_data()
        return self.mesh_cell2point.point_data[field_name]
      if field_name not in self.mesh_cell2point.point_data:
        raise ValueError(f"Field '{field_name}' not found in projected point data.")
      return self.mesh_cell2point.point_data[field_name]
  
  def replace_cell_field_by_point_field(self, field_name:str):
    if field_name not in self.cell_data and field_name not in self.point_data:
      raise ValueError(f"Field '{field_name}' not found in mesh.")
    if self.mesh_cell2point is None:
      self.mesh_cell2point = self.cell_data_to_point_data()
    if field_name not in self.mesh_cell2point.point_data:
      self.mesh_cell2point = self.cell_data_to_point_data()
    # remove the cell field and replace by point field
    point_field = self.mesh_cell2point.point_data[field_name]
    self.cell_data.pop(field_name)
    self.point_data[field_name] = point_field
    return