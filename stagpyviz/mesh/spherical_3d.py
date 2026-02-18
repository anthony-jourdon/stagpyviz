import pyvista as pvs
import numpy as np
from time import perf_counter

class UnstructuredSphere(pvs.UnstructuredGrid):
  def __init__(self, *args, deep:bool=False, **kwargs) -> None:
    super().__init__(*args, deep=deep, **kwargs)
    self._points_spherical    = None
    self._centroids           = None
    self._centroids_spherical = None
    return
  
  def is_point_field(self, field:np.ndarray) -> bool:
    if field.shape[0] == self.number_of_points:
      return True
    return False
  
  def is_cell_field(self, field:np.ndarray) -> bool:
    if field.shape[0] == self.number_of_cells:
      return True
    return False
  
  def create_point_field(self, bs:int=1, dtype:np.dtype=np.float64) -> np.ndarray:
    if bs == 1:
      return np.zeros( (self.number_of_points), dtype=dtype )
    else:
      return np.zeros( (self.number_of_points, bs), dtype=dtype )
    
  def create_cell_field(self, bs:int=1, dtype:np.dtype=np.float64) -> np.ndarray:
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
      self._centroids = self.cell_centers().points
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
    return self.rotation_matrix(self.points_spherical[:,1],self.points_spherical[:,2])
  
  def rotation_matrix_centroids(self) -> np.ndarray:
    return self.rotation_matrix(self.centroids_spherical[:,1],self.centroids_spherical[:,2])
  
  def vector_cartesian_to_spherical(self, v_cartesian:np.ndarray) -> np.ndarray:
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
  
  def Jacobian_matrix(self, r:np.ndarray, theta:np.ndarray, phi:np.ndarray) -> np.ndarray:
    """
    Compute the Jacobian matrix for Cartesian - Spherical gradients transformations.
    """
    J = np.zeros( (r.shape[0], 3,3), dtype=np.float64 )
    J[:,0,0] = np.sin(theta)*np.cos(phi)
    J[:,0,1] = r*np.cos(theta)*np.cos(phi)
    J[:,0,2] = -r*np.sin(theta)*np.sin(phi)

    J[:,1,0] = np.sin(theta)*np.sin(phi)
    J[:,1,1] = r*np.cos(theta)*np.sin(phi)
    J[:,1,2] = r*np.sin(theta)*np.cos(phi)

    J[:,2,0] = np.cos(theta)
    J[:,2,1] = -r*np.sin(theta)
    J[:,2,2] = 0.0
    return J

  def spherical_gradient(self,cartesian_gradient:np.ndarray) -> np.ndarray:
    t0 = perf_counter()
    if self.is_point_field(cartesian_gradient):
      coords = self.points_spherical
    elif self.is_cell_field(cartesian_gradient):
      coords = self.centroids_spherical
    else:
      raise ValueError(f"Input gradient field has incompatible shape. Mesh ncells: {self.number_of_cells}, npoints: {self.number_of_points}, field shape: {cartesian_gradient.shape}")
    r     = coords[:,0]
    theta = coords[:,1]
    phi   = coords[:,2]
    J = self.Jacobian_matrix(r, theta, phi)
    sph_grad = np.einsum('eji,ej->ei', J, cartesian_gradient)
    t1 = perf_counter()
    print(f"Spherical gradient transformation performed in {t1-t0:g} seconds")
    return sph_grad