import numpy as np
from time import perf_counter
from functools import cached_property
# Use relative import when imported as module, absolute when run directly
try:
  from .mesh import Hex2DMesh
except ImportError:
  from mesh import Hex2DMesh

class SphericalMesh(Hex2DMesh):
  def __init__(self, dimensions=None, r=None, phi=None, *args, deep: bool = False, **kwargs):
    super().__init__(None, None, None, *args, deep=deep, dimensions=dimensions, **kwargs)
    # Optional storage for spherical coordinates (r, phi) per point
    self.points_spherical: np.ndarray | None = None
    if r is not None and phi is not None:
      self.spherical_coor = (r, phi)
      self.cartesian_coor = (r, phi)
    return

  @property
  def cartesian_coor(self):
    return self.points
  
  @cartesian_coor.setter
  def cartesian_coor(self, value:tuple[np.ndarray|list, np.ndarray|list]):
    """
    Setter that accepts `(r, phi)` and builds cartesian coordinates.

    Parameters
    ----------
    value : tuple(ndarray, ndarray)
        A tuple `(r, phi)` where `r.shape == (ny,)` and `phi.shape == (nx,)` (no ghost).
    """
    try:
      r, phi = value
    except Exception as exc:
      raise TypeError("cartesian_coor must be a tuple of (r, phi)") from exc
    if self.dimensions is None:
      raise ValueError("mesh.dimensions must be set before assigning cartesian_coor")
    if isinstance(r, list):
      r = np.asarray(r, dtype=np.float64)
    if isinstance(phi, list):
      phi = np.asarray(phi, dtype=np.float64)
    self.set_cartesian_coor_vec(r, phi)
    return
  
  @property
  def spherical_coor(self) -> np.ndarray | None:
    """
    Getter returning the stored spherical coordinates array.

    Returns
    -------
    np.ndarray | None
        Array of shape `(ny*nx_ghost, 2)` with columns `[r, phi]`, or None if not set.
    """
    return self.points_spherical

  @spherical_coor.setter
  def spherical_coor(self, value:tuple[np.ndarray|list, np.ndarray|list]):
    """
    Setter that accepts `(r, phi)` and builds spherical coords.

    Parameters
    ----------
    value : tuple(ndarray, ndarray)
        A tuple `(r, phi)` where `r.shape == (ny,)` and `phi.shape == (nx,)` (no ghost).

    Notes
    -----
    - Uses `self.dimensions` to determine `(nx_ghost, ny)`. Ensure it is set.
    - Stores a 2-column array `[r, phi]` for each point in `self.points_spherical`.
    """
    try:
      r, phi = value
    except Exception as exc:
      raise TypeError("spherical_coor must be a tuple of (r, phi)") from exc
    if self.dimensions is None:
      raise ValueError("mesh.dimensions must be set before assigning spherical_coor")
    if isinstance(r, list):
      r = np.asarray(r, dtype=np.float64)
    if isinstance(phi, list):
      phi = np.asarray(phi, dtype=np.float64)
    self.set_spherical_coor_vec(r, phi)
    return

  def set_cartesian_coor(self,r:np.ndarray,phi:np.ndarray,dimensions=None):
    """
    Sets the cartesian coordinates of the mesh vertices based on given spherical coordinates.

    Parameters
    ----------
    - r:   1D array of radial coordinates of length ny (no ghost)
    - phi: 1D array of azimuthal coordinates of length nx (no ghost)
    - dimensions: optional tuple where dimensions[0] = nx_ghost (nx + 1), dimensions[1] = ny
    """
    t0 = perf_counter()
    if self.dimensions is None and dimensions is None:
      raise ValueError("Either mesh.dimensions or dimensions argument must be provided.")
    # If dimension is provided as argument, override mesh.dimensions
    if dimensions is not None:
      self.dimensions = dimensions
    if dimensions is None:
      dimensions = self.dimensions
    # check that dimensions match
    if dimensions[0]-1 != phi.shape[0]:
      s  = f"dimensions[0]-1 ({dimensions[0]-1}) != phi.shape[0] ({phi.shape[0]})\n"
      s += "Either set mesh.dimensions or pass dimensions using the keyword argument \"dimensions=\"."
      raise ValueError(s)
    if dimensions[1] != r.shape[0]:
      s  = f"dimensions[1] ({dimensions[1]}) != r.shape[0] ({r.shape[0]})\n"
      s += "Either set mesh.dimensions or pass dimensions using the keyword argument \"dimensions=\"."
      raise ValueError(s)
    # create coordinates
    self.points = np.zeros((dimensions[0]*dimensions[1], 3), dtype=np.float64)
    nx       = dimensions[0] - 1 # exclude ghost point
    nx_ghost = dimensions[0]     # include ghost point
    ny       = dimensions[1]
    for j in range(ny):
      for i in range(nx):
        nidx = i + j * nx_ghost
        self.points[nidx, 0] = r[j] * np.cos(phi[i])  # x
        self.points[nidx, 1] = r[j] * np.sin(phi[i])  # y
      # Close the periodic boundary by copying first column to last
      nidx = nx + j * nx_ghost
      self.points[nidx, 0] = self.points[j * nx_ghost, 0]
      self.points[nidx, 1] = self.points[j * nx_ghost, 1]
    t1 = perf_counter()
    print(f"{self.set_cartesian_coor.__name__} created coordinates for {self.number_of_points} points in {t1 - t0:g} seconds.")
    return

  def set_cartesian_coor_vec(self, r:np.ndarray, phi:np.ndarray, dimensions=None):
    """
    Vectorized version of create_coor using NumPy broadcasting.

    Parameters
    ----------
    - r:   1D array of radial coordinates of length ny (no ghost)
    - phi: 1D array of azimuthal coordinates of length nx (no ghost)
    - dimensions: optional tuple matching mesh.dimensions where
                  dimensions[0] = nx_ghost (nx + 1), dimensions[1] = ny

    Produces self.points shaped (nx_ghost * ny, 3) with the last azimuthal
    column closing the periodic boundary by duplicating the first column.
    """
    t0 = perf_counter()
    if self.dimensions is None and dimensions is None:
      raise ValueError("Either mesh.dimensions or dimensions argument must be provided.")
    # If dimension is provided as argument, override mesh.dimensions
    if dimensions is not None:
      self.dimensions = dimensions
    if dimensions is None:
      dimensions = self.dimensions
    # check that dimensions match
    if dimensions[0]-1 != phi.shape[0]:
      s  = f"dimensions[0]-1 ({dimensions[0]-1}) != phi.shape[0] ({phi.shape[0]})\n"
      s += "Either set mesh.dimensions or pass dimensions using the keyword argument \"dimensions=\"."
      raise ValueError(s)
    if dimensions[1] != r.shape[0]:
      s  = f"dimensions[1] ({dimensions[1]}) != r.shape[0] ({r.shape[0]})\n"
      s += "Either set mesh.dimensions or pass dimensions using the keyword argument \"dimensions=\"."
      raise ValueError(s)

    nx       = dimensions[0] - 1  # exclude ghost point
    nx_ghost = dimensions[0]      # include ghost point
    ny       = dimensions[1]

    # Base coordinates without ghost: shape (ny, nx)
    cos_phi = np.cos(phi)            # (nx,)
    sin_phi = np.sin(phi)            # (nx,)
    r_col   = r.reshape(ny, 1)       # (ny,1)

    x_base = r_col * cos_phi.reshape(1, nx)  # (ny, nx)
    y_base = r_col * sin_phi.reshape(1, nx)  # (ny, nx)

    # Append ghost column by duplicating the first column to close periodic boundary
    x = np.concatenate([x_base, x_base[:, :1]], axis=1)  # (ny, nx_ghost)
    y = np.concatenate([y_base, y_base[:, :1]], axis=1)  # (ny, nx_ghost)

    # Allocate and fill points: flatten row-major to match nidx = i + j*nx_ghost
    pts = np.zeros((nx_ghost * ny, 3), dtype=np.float64)
    pts[:, 0] = x.reshape(-1)
    pts[:, 1] = y.reshape(-1)
    self.points = pts
    t1 = perf_counter()
    print(f"{self.set_cartesian_coor_vec.__name__} created coordinates for {self.number_of_points} points in {t1 - t0:g} seconds.")
    return
  
  def set_spherical_coor_vec(self, r:np.ndarray, phi:np.ndarray):
    """
    Sets the spherical coordinates of the mesh vertices based on given r and phi arrays.

    Parameters
    ----------
    - r:   1D array of radial coordinates of length ny (no ghost)
    - phi: 1D array of azimuthal coordinates of length nx (no ghost)
    
    Produces self.points_spherical shaped (nx_ghost * ny, 2) with columns [r, phi],
    where the last azimuthal column closes the periodic boundary by duplicating the first column.
    
    Notes
    -----
    - Uses `self.dimensions` to determine `(nx_ghost, ny)`. Ensure it is set.
    - Stores a 2-column array `[r, phi]` for each point in `self.points_spherical`.
    """
    t0 = perf_counter()
    nx_ghost = self.dimensions[0]
    ny       = self.dimensions[1]
    nx       = nx_ghost - 1

    if r.shape[0] != ny:
      raise ValueError(f"r length ({r.shape[0]}) must equal dimensions[1] ({ny})")
    if phi.shape[0] != nx:
      raise ValueError(f"phi length ({phi.shape[0]}) must equal dimensions[0]-1 ({nx})")

    # Build spherical coordinates per point (including ghost column for periodic closure)
    phi_ghost = np.concatenate([phi, phi[:1]])                    # (nx_ghost,)
    r_grid    = np.repeat(r.reshape(ny, 1), nx_ghost, axis=1)     # (ny, nx_ghost)
    phi_grid  = np.tile(phi_ghost.reshape(1, nx_ghost), (ny, 1))  # (ny, nx_ghost)

    spherical = np.zeros((nx_ghost * ny, 2), dtype=np.float64)
    spherical[:, 0] = r_grid.reshape(-1)
    spherical[:, 1] = phi_grid.reshape(-1)
    self.points_spherical = spherical
    t1 = perf_counter()
    print(f"{self.set_spherical_coor_vec.__name__} created spherical coordinates for {self.number_of_points} points in {t1 - t0:g} seconds.")
    return
  
  def rotation_matrix(self, phi:np.ndarray):
    """
    Returns the rotation matrix containing the basis vectors in spherical coordinates.

    Returns
    -------
    np.ndarray
        Array of shape `(number_of_points, 2, 2)` where each 2x2 matrix is:
        [[cos(phi), -sin(phi)],
         [sin(phi),  cos(phi)]]
    """
    R = np.zeros((phi.shape[0], 2, 2), dtype=np.float64)
    R[:, 0, 0] = np.cos(phi)
    R[:, 0, 1] = -np.sin(phi)
    R[:, 1, 0] = np.sin(phi)
    R[:, 1, 1] = np.cos(phi)
    return R

  @cached_property
  def rotation_matrix_vertices(self):
    """
    Returns the rotation matrix at mesh vertices.
    
    Returns
    -------
    np.ndarray
        Array of shape `(number_of_points, 2, 2)` where each 2x2 matrix is:
        [[cos(phi), -sin(phi)],
        [sin(phi),  cos(phi)]]
        
    """
    if self.spherical_coor is None:
      raise ValueError("spherical_coor must be set before accessing rotation_matrix")
    phi = self.spherical_coor[:, 1]
    return self.rotation_matrix(phi)

  @cached_property
  def rotation_matrix_centroids(self):
    """
    Returns the rotation matrix at cell centroids.

    Returns
    -------
    np.ndarray
        Array of shape `(number_of_cells, 2, 2)` where each 2x2 matrix is:
        [[cos(phi), -sin(phi)],
         [sin(phi),  cos(phi)]]
    """
    if self.spherical_coor is None:
      raise ValueError("spherical_coor must be set before accessing rotation_matrix")
    phi = self.centroids_spherical[:,1]
    return self.rotation_matrix(phi)

  def vector_spherical_to_cartesian(self, u:np.ndarray):
    """
    Change of basis from spherical to cartesian for vector field u.

    Parameters
    ----------
    u : np.ndarray
        Input vector field of shape `(n_points|n_cells, n_components)` in spherical basis.

    Returns
    -------
    np.ndarray
        Output vector field of shape `(n_points|n_cells, n_components)` in cartesian basis. 
    """
    if self.is_cell_field(u):
      R = self.rotation_matrix_centroids
    elif self.is_point_field(u):
      R = self.rotation_matrix_vertices
    else:
      raise ValueError(f"Input field u must be either a point or cell field. Expected shape ({self.number_of_points}|{self.number_of_cells}, n_components), got {u.shape}.")
    v = np.zeros_like(u, dtype=u.dtype)
    v[:,0] = R[:,0,0] * u[:,0] + R[:,0,1] * u[:,1]
    v[:,1] = R[:,1,0] * u[:,0] + R[:,1,1] * u[:,1]
    return v
  
  def vector_cartesian_to_spherical(self, u:np.ndarray):
    """
    Change of basis from cartesian to spherical for vector field u.

    Parameters
    ----------
    u : np.ndarray
        Input vector field of shape `(n_points|n_cells, n_components)` in cartesian basis.

    Returns
    -------
    np.ndarray
        Output vector field of shape `(n_points|n_cells, n_components)` in spherical basis.
    """
    if self.is_cell_field(u):
      R = self.rotation_matrix_centroids
    elif self.is_point_field(u):
      R = self.rotation_matrix_vertices
    else:
      raise ValueError(f"Input field u must be either a point or cell field. Expected shape ({self.number_of_points}|{self.number_of_cells}, n_components), got {u.shape}.")
    v = np.zeros_like(u, dtype=u.dtype)
    # Note the transpose of R for inverse transformation
    v[:,0] = R[:,0,0] * u[:,0] + R[:,1,0] * u[:,1]
    v[:,1] = R[:,0,1] * u[:,0] + R[:,1,1] * u[:,1]
    return v
  
  @cached_property
  def centroids_spherical(self):
    """
    Spherical coordinates of cell centroids.
    
    Returns
    -------
    np.ndarray
        Array of shape `(number_of_cells, 2)` with columns `[r, phi]
    """
    if self.cartesian_coor is None:
      raise ValueError("cartesian_coor must be set before accessing centroids_spherical")
    centroids_cartesian = self.centroids
    centroids_spherical = np.zeros((self.number_of_cells, 2), dtype=np.float64)
    x   = centroids_cartesian[:,0]
    y   = centroids_cartesian[:,1]
    r   = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    centroids_spherical[:,0] = r
    centroids_spherical[:,1] = phi
    return centroids_spherical

  def Jacobian_matrix(self,spherical_coor:np.ndarray):
    """
    Computes the Jacobian matrix for the transformation from spherical to cartesian coordinates.

    Parameters
    ----------
    spherical_coor : np.ndarray
        Array of shape (n_points, 2) with columns [r, phi]. Can be either point or cell coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2, 2) where each 2x2 matrix is the Jacobian:
        [[cos(phi), -r*sin(phi)],
         [sin(phi),  r*cos(phi)]]
    """
    r   = spherical_coor[:,0]
    phi = spherical_coor[:,1]
    J = np.zeros((spherical_coor.shape[0], 2, 2), dtype=np.float64)
    J[:,0,0] = np.cos(phi)
    J[:,0,1] = -r * np.sin(phi)
    J[:,1,0] = np.sin(phi)
    J[:,1,1] = r * np.cos(phi)
    return J
  
  def spherical_gradient(self, cartesian_gradient:np.ndarray):
    """
    Transforms a gradient vector from cartesian to spherical coordinates.

    Parameters
    ----------
    cartesian_gradient : np.ndarray
        Input gradient vector field of shape `(n_points|n_cells, 2)` in cartesian coordinates.
    
    Returns
    -------
    np.ndarray
        Output gradient vector field of shape `(n_points|n_cells, 2)` in spherical coordinates.

    Notes
    -----
    The transformation is given by:
        spherical_gradient = J @ cartesian_gradient
    where J is the Jacobian matrix of the transformation from spherical to cartesian coordinates.
    """
    if self.is_cell_field(cartesian_gradient):
      spherical_coor = self.centroids_spherical
    elif self.is_point_field(cartesian_gradient):
      spherical_coor = self.spherical_coor
    else:
      raise ValueError(f"Input field cartesian_gradient must be either a point or cell field. Got shape {cartesian_gradient.shape}.")
    J = self.Jacobian_matrix(spherical_coor)
    grad_spherical = np.zeros_like(cartesian_gradient)
    grad_spherical[:,0] = J[:,0,0] * cartesian_gradient[:,0] + J[:,1,0] * cartesian_gradient[:,1]
    grad_spherical[:,1] = J[:,0,1] * cartesian_gradient[:,0] + J[:,1,1] * cartesian_gradient[:,1]
    return grad_spherical

  def cell_data_to_point_data(self):
    """
    Converts cell data to point data with proper handling of periodic boundaries.
    
    This method overrides the parent PyVista method to correctly average cell values
    at the periodic boundary (first and last azimuthal columns).
    
    Returns
    -------
    result : pyvista.StructuredGrid
        Mesh with cell data averaged to point data, with periodic boundaries handled.
    """
    # Call parent method to do the initial conversion
    result = super().cell_data_to_point_data()
    
    # Fix periodic boundary for each field
    nx_ghost = self.dimensions[0]
    ny = self.dimensions[1]
    
    # Process each point data field
    for field_name in result.point_data.keys():
      field = result.point_data[field_name]
      
      # Handle scalar and vector fields
      if field.ndim == 1:
        # Scalar field
        for j in range(ny):
          first_idx = j * nx_ghost
          last_idx = j * nx_ghost + (nx_ghost - 1)
          
          # Average the values at first and last columns
          avg_value = 0.5 * (field[first_idx] + field[last_idx])
          
          # Set both to the average
          field[first_idx] = avg_value
          field[last_idx] = avg_value
      else:
        # Vector/tensor field
        for j in range(ny):
          first_idx = j * nx_ghost
          last_idx = j * nx_ghost + (nx_ghost - 1)
          
          # Average the values at first and last columns
          avg_value = 0.5 * (field[first_idx,:] + field[last_idx,:])
          
          # Set both to the average
          field[first_idx,:] = avg_value
          field[last_idx,:] = avg_value
    
    return result

def test():
  import pyvista as pvs
  nx = 6
  nx_ghost = nx + 1
  ny = 3
  dx = 2 * np.pi / nx_ghost
  theta = np.linspace(0, nx*dx, nx) 
  r     = np.linspace(1, 2, ny)     

  coor = np.zeros((ny*nx_ghost, 3), dtype=np.float64)
  for j in range(ny):
    for i in range(nx):
      nidx = i + j * nx_ghost
      coor[nidx, 0] = r[j] * np.cos(theta[i])  # x
      coor[nidx, 1] = r[j] * np.sin(theta[i])  # y
    # Close the periodic boundary by copying first column to last
    nidx = nx + j * nx_ghost
    coor[nidx, 0] = coor[j * nx_ghost, 0]
    coor[nidx, 1] = coor[j * nx_ghost, 1]
  
  mesh = SphericalMesh(coor, dimensions=(nx_ghost, ny, 1))
  #mesh.dimensions = (nx_ghost, ny, 1)
  mesh.create_e2v()
  print(mesh.elidx)
  for iel in range(mesh.number_of_cells):
    eidx = mesh.elidx[iel,:]
    print(f"Element {iel} nodes: {eidx}, coordinates:")
    for n in eidx:
      print(f"  Node {n}: {mesh.points[n,:]}")

  plotter = pvs.Plotter()
  plotter.add_mesh(mesh, show_edges=True, color='lightgrey')
  label_coords = mesh.points + [0, 0, 0.01]
  plotter.add_point_labels(
    label_coords,
    [f'Point {i}' for i in range(mesh.number_of_points)],
    font_size=20,
    point_size=20,
  )
  plotter.camera_position = 'xy'
  plotter.show()

  return

def test2():
  import pyvista as pvs
  nx = 6
  nx_ghost = nx + 1
  ny = 3
  dx = 2 * np.pi / nx_ghost
  theta = np.linspace(0, nx*dx, nx)  # 4 elements in azimuthal direction
  r     = np.linspace(1, 2, ny)      # 2 elements in radial direction

  mesh = SphericalMesh(dimensions=(nx_ghost, ny, 1), r=r, phi=theta)
  print(mesh.dimensions)
  print(mesh.cells_dimensions)
  print(mesh.nodes_dimensions)
  print(mesh.spherical_coor)
  print(mesh.cartesian_coor)
  print(mesh.rotation_matrix_vertices)
  #mesh.create_coor(r, theta)#, dimensions=(nx_ghost, ny, 1))
  #mesh.spherical_coor = (r, theta)
  #R = mesh.rotation_matrix
  
  exit()
  mesh.create_e2v()
  print(mesh.elidx)
  for iel in range(mesh.number_of_cells):
    eidx = mesh.elidx[iel,:]
    print(f"Element {iel} nodes: {eidx}, coordinates:")
    for n in eidx:
      print(f"  Node {n}: {mesh.points[n,:]}")

  plotter = pvs.Plotter()
  plotter.add_mesh(mesh, show_edges=True, color='lightgrey')
  label_coords = mesh.points + [0, 0, 0.01]
  plotter.add_point_labels(
    label_coords,
    [f'Point {i}' for i in range(mesh.number_of_points)],
    font_size=20,
    point_size=20,
  )
  plotter.camera_position = 'xy'
  plotter.show()

  return

def stacked_inverse_3x3(A):
  AI = np.empty_like(A)
  for i in range(3):
    AI[...,:,i] = np.cross(A[...,i-2,:], A[...,i-1,:])
  x1x2 = AI[...,:,0]
  x0 = A[...,0,:]
  mydet = np.einsum('ij,ij->i',x1x2, x0)
  #print(mydet,np.linalg.det(A[0]))
  return  AI / mydet[..., None, None]


if __name__ == "__main__":
  test2()