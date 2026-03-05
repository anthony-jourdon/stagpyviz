from pathlib import Path
from functools import cached_property
from time import perf_counter
import pyvista as pvs
import numpy as np
from scipy.spatial import ConvexHull

try:
  from ..parsers import BinHeader64, BinHeader, read_stag_bin
  from ..elements.wedge_3d import Wedge3D
  from .spherical_3d import UnstructuredSphere
  from .shell import ShellMesh
except ImportError:
  from stagpyviz.parsers import BinHeader64, BinHeader, read_stag_bin
  from stagpyviz.elements.wedge_3d import Wedge3D
  from stagpyviz.mesh.spherical_3d import UnstructuredSphere
  from stagpyviz.mesh.shell import ShellMesh

class YinYangMesh(UnstructuredSphere):
  """
  A class to represent an unstructured mesh of a sphere reconstructed from a StagYY Yin-Yang grid.
  The mesh is discretized as a collection of wedge (prismatic) elements :math:`\\mathcal W_1`.

  To reconstruct the mesh from a raw binary file containing the Yin-Yang grid we process as follows:

  1.
    Extract the yin and yang grids

  2.
    Determine the indices of the points that are not in the overlapping region between the two grids

  3.
    Exploiting the fact that the grid is structured in the radial direction even after removing the overlapping points 
    and that the number of points per radial layer is always the same, 
    we reshape the coordinates of the points in the Yin and Yang grids into 2D arrays of shape ``(points_per_layer, n_radial_layers)``

  4.
    Generate a surface mesh of the outer shell of the sphere using the points of the last radial layer of both grids using the 
    class :py:class:`ShellMesh <stagpyviz.ShellMesh>`

  5.
    Generate the volume mesh by extruding the surface mesh radially and connecting the nodes of the surface mesh 
    to the nodes of the inner layers by constructing wedge elements.

  The mesh can be generated directly from the raw binary file or from a VTU file containing the mesh and the fields.

  :param pathlib.Path|str rawbin: The path to the raw binary file containing the Yin-Yang grid (output of StagYY)

  :Attributes:

  .. py:attribute:: yin

    Dictionary containing the coordinates of the points in the Yin grid. 
    The keys are ``"x"``, ``"y"``, ``"z"`` for Cartesian coordinates 
    and ``"R"``, ``"theta"``, ``"phi"`` for spherical coordinates.
    Contain the entire grid including the overlapping region.
    Arrays of shape ``(nx, ny, nz)``.

    :type: dict

  .. py:attribute:: yang

    Dictionary containing the coordinates of the points in the Yang grid. 
    The keys are ``"x"``, ``"y"``, ``"z"`` for Cartesian coordinates 
    and ``"R"``, ``"theta"``, ``"phi"`` for spherical coordinates.
    Contain the entire grid including the overlapping region.
    Arrays of shape ``(nx, ny, nz)``.

    :type: dict

  .. py:attribute:: good_indices

    Boolean array of shape ``(nx*ny*nz,)`` to flag the points that are not in the overlapping region between the Yin and Yang grids.

    :type: numpy.ndarray

  .. py:attribute:: points_per_layer
    
    Number of points per radial layer in the Yin (or Yang) grid after removing the overlapping points.

    :type: int

  .. py:attribute:: surface_mesh

    A surface mesh of the outer shell of the sphere generated from the points of the last radial layer of 
    both grids.

    :type: :py:class:`ShellMesh <stagpyviz.ShellMesh>`

  .. py:attribute:: elements

    An instance of the class :py:class:`Wedge3D <stagpyviz.Wedge3D>` 
    to represent the wedge elements of the mesh and perform operations on them.

    :type: :py:class:`Wedge3D <stagpyviz.Wedge3D>`

  .. py:attribute:: grid_dimensions

    Tuple of the dimensions of the Yin (or Yang) grid in the x, y and z directions.

    :type: tuple[int,int,int]

  .. py:attribute:: grid_npoints

    Total number of points in the Yin (or Yang) grid.

    :type: int

  .. py:attribute:: yin_radial_idx

    2D array of shape ``(points_per_layer, n_radial_layers)`` containing the indices of the points in the Yin grid reshaped radially.

    :type: numpy.ndarray

  .. py:attribute:: yang_radial_idx

    2D array of shape ``(points_per_layer, n_radial_layers)`` containing the indices of the points in the Yang grid reshaped radially.

    :type: numpy.ndarray

  .. py:attribute:: surface_idx

    1D array of shape ``(2*points_per_layer,)`` containing the indices of the points in the surface mesh 
    (last radial layer of both grids).

    :type: numpy.ndarray

  .. py:attribute:: surface_cells

    1D array of shape ``(number_of_surface_cells,)`` containing the indices of the cells in the surface mesh 
    (last radial layer of both grids).

    :type: numpy.ndarray

  :Methods:

  """
  def __init__(self, rawbin:Path|str, *args, deep:bool=False, **kwargs):
    if not isinstance(rawbin, (Path, str)):
      raise ValueError("First argument must be the path to the raw binary file (StragYY output)")
    # Read the binary file to extract information about the grid
    self.header, _ = read_stag_bin(rawbin)
    
    self.yin                  = None
    self.yang                 = None
    self._good_indices        = None
    self._points_per_layer    = None
    self._surface_mesh        = None
    self.elements:Wedge3D = Wedge3D()

    self.yin, self.yang = self.reconstruct_yinyang()
    self.reshape_YY_radially()
    self.generate_surface_mesh()
    self.generate_radial_indices()

    # Check if user provided a VTU file to load the mesh from
    if len(args) == 1 and isinstance(args[0], (Path, str)):
      if (args[0].endswith(".vtu")):
        t0 = perf_counter()
        mesh:pvs.UnstructuredGrid = pvs.read(args[0])
        super().__init__(mesh, deep=deep, **kwargs)
        t1 = perf_counter()
        print(f"VTU file loaded in {t1-t0:g} seconds")
    else:
      # Generate mesh from raw binary file
      t0 = perf_counter()
      self.construct_mesh()
      t1 = perf_counter()
      print(f"Yin-Yang mesh re-constructed in {t1-t0:g} seconds")
    self.nodes_per_el = self.elements.basis_per_el
    return
  
  @property
  def grid_dimensions(self) -> tuple[int,int,int]:
    n = (
      self.header['ntot'][0],
      self.header['ntot'][1],
      self.header['ntot'][2]
    )
    return n
  
  @property
  def grid_npoints(self) -> int:
    n = self.grid_dimensions
    return n[0]*n[1]*n[2]
  
  def reconstruct_yinyang(self) -> tuple[dict,dict]:
    yin  = {}
    yang = {}
    header = self.header
    npoints_grid = self.grid_npoints
    # construct a mesh grid for one grid
    X,Y,Z = np.meshgrid(
      header['x'],
      header['y'],
      header['z'],
      indexing='ij'
    )
    # Flatten the arrays
    X = np.reshape(X, (npoints_grid))
    Y = np.reshape(Y, (npoints_grid))
    Z = np.reshape(Z, (npoints_grid))
    # =================== #
    # StagYY coordinates: #
    # =================== #
    R   = Z + header["rcmb"]
    lat = 0.25*np.pi - X
    lon = Y - 0.75*np.pi
    # =================== #
    #  Cartesian coords   #
    # =================== #
    # Yin grid
    yin["x"],yin["y"],yin["z"] = self.spherical_to_cartesian(R,lat,lon)
    # Yang grid
    yang["x"] = -yin["x"]
    yang["y"] = yin["z"]
    yang["z"] = yin["y"]
    # =================== #
    #   Spherical coords  #
    # =================== #
    yin["R"],yin["theta"],yin["phi"] = self.cartesian_to_spherical(
      yin["x"],
      yin["y"],
      yin["z"]
    )
    yang["R"],yang["theta"],yang["phi"] = self.cartesian_to_spherical(
      yang["x"],
      yang["y"],
      yang["z"]
    )
    return yin, yang

  def generate_good_indices(self):
    # Ensure that the Yin and Yang grids have been reconstructed
    if self.yin is None or self.yang is None:
      self.yin, self.yang = self.reconstruct_yinyang()

    theta12  = np.arccos( 
      np.multiply( 
        np.sin(self.yin["theta"]), 
        np.sin(self.yin["phi"]) 
      ) 
    )
    # Boolean array to flag the overlapping region
    redFlags  = np.logical_or(
      np.logical_and( 
        (theta12 > 0.25*np.pi), 
        (self.yin["phi"] > 0.5*np.pi) 
      ), np.logical_and(
        (theta12 < 0.75*np.pi),
        (self.yin["phi"] < -0.5*np.pi)
      )
    )
    self._good_indices = np.ones(self.yin["x"].shape[0], dtype=bool)
    self._good_indices[redFlags] = False
    return 
  
  @property
  def good_indices(self) -> np.ndarray:
    if self._good_indices is None:
      self.generate_good_indices()
    return self._good_indices

  def generate_points_per_layer(self):
    if self._points_per_layer is not None:
      return
    gidx = self.good_indices
    n    = self.grid_dimensions
    for key in self.yin.keys():
      self.yin[key] = self.yin[key][gidx]
      self.yang[key] = self.yang[key][gidx]
    self._points_per_layer = np.int32(self.yin["x"].shape[0] / n[2])
    return
  
  @property
  def points_per_layer(self) -> np.int32:
    if self._points_per_layer is None:
      self.generate_points_per_layer()
    return self._points_per_layer
  
  def reshape_radially(self,field:np.ndarray) -> np.ndarray:
    """
    Reshape a field defined on the Yin or Yang grid into a 2D array of shape ``(points_per_layer, n_radial_layers)``.
    """
    ppl = self.points_per_layer
    n   = self.grid_dimensions
    return np.reshape(field, (ppl, n[2]))
  
  def reshape_YY_radially(self) -> None:
    ppl = self.points_per_layer
    n   = self.grid_dimensions
    for key in self.yin.keys():
      if self.yin[key].shape == (ppl, n[2]):
        continue
      self.yin[key]  = self.reshape_radially(self.yin[key])
    for key in self.yang.keys():
      if self.yang[key].shape == (ppl, n[2]):
        continue
      self.yang[key] = self.reshape_radially(self.yang[key])
    return

  def generate_radial_indices(self) -> np.ndarray:
    n   = self.grid_dimensions
    ppl = self.points_per_layer
    nidx = np.arange(0,ppl*n[2], dtype=np.int32)
    self.yin["idx"]  = np.reshape(nidx, (ppl, n[2]), order='F')
    self.yang["idx"] = self.yin["idx"] + ppl*n[2]
    return
  
  @property
  def yin_radial_idx(self) -> np.ndarray:
    if "idx" not in self.yin:
      self.generate_radial_indices()
    return self.yin["idx"]
  
  @property
  def yang_radial_idx(self) -> np.ndarray:
    if "idx" not in self.yang:
      self.generate_radial_indices()
    return self.yang["idx"]
  
  @property
  def surface_idx(self) -> np.ndarray:
    n        = self.grid_dimensions
    ppl      = self.points_per_layer
    yin_idx  = self.yin_radial_idx
    yang_idx = self.yang_radial_idx
    idx = np.zeros((2*ppl), dtype=np.int32)
    idx[0:ppl]     = yin_idx[:, n[2]-1]
    idx[ppl:2*ppl] = yang_idx[:, n[2]-1]
    return idx
  
  @property
  def surface_cells(self) -> np.ndarray:
    n         = self.grid_dimensions
    shell_nel = self.surface_mesh.number_of_cells
    return np.arange((n[2]-2)*shell_nel, (n[2]-1)*shell_nel, dtype=np.int64)

  @property
  def surface_mesh(self) -> ShellMesh:
    if self._surface_mesh is None:
      self.generate_surface_mesh()
    return self._surface_mesh
  
  def generate_surface_mesh(self) -> None:
    n   = self.grid_dimensions
    ppl = self.points_per_layer

    self.reshape_YY_radially()

    # Gather both grids
    X = np.zeros((n[2],2*ppl))
    Y = np.zeros((n[2],2*ppl))
    Z = np.zeros((n[2],2*ppl))
    # First half is Yin
    X[:,0:ppl] = self.yin["x"].T
    Y[:,0:ppl] = self.yin["y"].T
    Z[:,0:ppl] = self.yin["z"].T
    # Second half is Yang
    X[:,ppl:2*ppl] = self.yang["x"].T
    Y[:,ppl:2*ppl] = self.yang["y"].T
    Z[:,ppl:2*ppl] = self.yang["z"].T

    points = np.array([X[n[2]-1,:], Y[n[2]-1,:], Z[n[2]-1,:]]).T
    self._surface_mesh = ShellMesh(points)
    return 

  def construct_mesh(self):
    n   = self.grid_dimensions
    ppl = self.points_per_layer
    self.reshape_YY_radially()

    # Create a convex hull mesh of the surface (triangles)
    outer_shell = self.surface_mesh
    # pyvista connectivity is a flat array => number_of_cells x nodes_per_cell
    shell_elidx = np.reshape(outer_shell.cell_connectivity, (-1,3))
    shell_nel   = shell_elidx.shape[0]

    # Generate the volume mesh
    yin_idx  = self.yin_radial_idx
    yang_idx = self.yang_radial_idx
    # Wedge type elements => 6 nodes per element
    elidx = np.zeros( ((n[2]-1)*shell_nel, 6), dtype=np.int32 )
    upper_triangle = np.zeros( (2*ppl), dtype=np.int32 )
    lower_triangle = np.zeros( (2*ppl), dtype=np.int32 )
    # construct connectivity element to vertex indices
    for k in range(n[2]-1):
      upper_triangle[0:ppl]     = yin_idx[:,k+1]
      upper_triangle[ppl:2*ppl] = yang_idx[:,k+1]
      
      lower_triangle[0:ppl]     = yin_idx[:,k]
      lower_triangle[ppl:2*ppl] = yang_idx[:,k]
      for i in range(3):
        elidx[k*shell_nel:(k+1)*shell_nel, i]   = upper_triangle[shell_elidx[:,i]]
        elidx[k*shell_nel:(k+1)*shell_nel, i+3] = lower_triangle[shell_elidx[:,i]]

    # coordinates
    points = np.zeros((2*ppl*n[2],3), dtype=np.float64)
    points[0:ppl*n[2], 0] = self.yin["x"].reshape((ppl*n[2]),order='F')
    points[0:ppl*n[2], 1] = self.yin["y"].reshape((ppl*n[2]),order='F')
    points[0:ppl*n[2], 2] = self.yin["z"].reshape((ppl*n[2]),order='F')
    points[ppl*n[2]:2*ppl*n[2], 0] = self.yang["x"].reshape((ppl*n[2]),order='F')
    points[ppl*n[2]:2*ppl*n[2], 1] = self.yang["y"].reshape((ppl*n[2]),order='F')
    points[ppl*n[2]:2*ppl*n[2], 2] = self.yang["z"].reshape((ppl*n[2]),order='F')
    # create the UnstructuredGrid
    super().__init__({pvs.CellType.WEDGE: elidx}, points)
    return
  
  def reconstruct_velocity(self, velocity_raw:np.ndarray) -> None:
    """
    Reconstruct the velocity field from the raw velocity components defined on the Yin and Yang grids.
    First the raw coordinates of the points are transformed such that:

    .. math::
      \\begin{split}
        r &= z_{\\text{raw}} + r_{\\text{cmb}} \\\\
        \\theta &= \\frac{\\pi}{4} - x_{\\text{raw}} \\\\
        \\phi &= y_{\\text{raw}} - \\frac{3\\pi}{4}
      \\end{split}

    Then the velocity components are transformed from the Yin-Yang grid coordinates to Cartesian coordinates 
    using the following relations:

    .. math::
      \\begin{split}
        \\mathbf{v}_{\\text{yin}} &= 
        \\begin{bmatrix}
          v_0 \\sin(\\theta)\\cos(\\phi) - v_1\\sin(\\phi) + v_2\\cos(\\theta)\\cos(\\phi) \\\\
          v_0 \\sin(\\theta)\\sin(\\phi) + v_1\\cos(\\phi) + v_2\\cos(\\theta)\\sin(\\phi) \\\\
          -v_0 \\cos(\\theta) + v_2\\sin(\\theta)
        \\end{bmatrix} \\\\
        \\mathbf{v}_{\\text{yang}} &=
        \\begin{bmatrix}
          -v_{\\text{yin}_x} \\\\
          v_{\\text{yin}_z} \\\\
          v_{\\text{yin}_y}
        \\end{bmatrix}
      \\end{split}

    where :math:`v_0`, :math:`v_1` and :math:`v_2` are the raw velocity components defined on the Yin and Yang grids,
    :math:`\\mathbf{v}_{\\text{yin}}` and :math:`\\mathbf{v}_{\\text{yang}}` are the velocity vectors 
    in Cartesian coordinates on the Yin and Yang grids respectively.

    :param numpy.ndarray velocity_raw: 
      The raw velocity components defined on the Yin and Yang grids.
      The array should have shape ``(3, nx, ny, nz, 2)``, where the first dimension corresponds to the velocity components, 
      the next three dimensions correspond to the grid dimensions and the last dimension corresponds to the Yin and Yang grids (0 for Yin and 1 for Yang).

    .. warning::
      This is an in-place operation that modifies the input array ``velocity_raw``.

    """
    t0 = perf_counter()
    # construct a mesh grid for one grid
    header = self.header
    n = self.grid_dimensions
    # =================== #
    # StagYY coordinates: #
    # =================== #
    X,Y,Z = np.meshgrid(
      header['x'],
      header['y'],
      header['z'],
      indexing='ij'
    )
    R   = Z + header["rcmb"]
    lat = 0.25*np.pi - X
    lon = Y - 0.75*np.pi

    # Yin grid
    v0 = velocity_raw[0,0:n[0],0:n[1],:,0]
    v1 = velocity_raw[1,0:n[0],0:n[1],:,0]
    v2 = velocity_raw[2,0:n[0],0:n[1],:,0]

    vx_yin = v0*np.sin(lat)*np.cos(lon) - v1*np.sin(lon) + v2*np.cos(lat)*np.cos(lon)
    vy_yin = v0*np.sin(lat)*np.sin(lon) + v1*np.cos(lon) + v2*np.cos(lat)*np.sin(lon)
    vz_yin = -v0*np.cos(lat) + v2*np.sin(lat)

    # Yang grid
    v0 = velocity_raw[0,0:n[0],0:n[1],:,1]
    v1 = velocity_raw[1,0:n[0],0:n[1],:,1]
    v2 = velocity_raw[2,0:n[0],0:n[1],:,1]

    vx_yang = -v0*np.sin(lat)*np.cos(lon) + v1*np.sin(lon) - v2*np.cos(lat)*np.cos(lon)
    vy_yang = -v0*np.cos(lat) + v2*np.sin(lat)
    vz_yang = v0*np.sin(lat)*np.sin(lon) + v1*np.cos(lon) + v2*np.cos(lat)*np.sin(lon)
    
    velocity_raw[0,0:n[0],0:n[1],:,0] = vx_yin
    velocity_raw[1,0:n[0],0:n[1],:,0] = vy_yin
    velocity_raw[2,0:n[0],0:n[1],:,0] = vz_yin

    velocity_raw[0,0:n[0],0:n[1],:,1] = vx_yang
    velocity_raw[1,0:n[0],0:n[1],:,1] = vy_yang
    velocity_raw[2,0:n[0],0:n[1],:,1] = vz_yang
    t1 = perf_counter()
    print(f"Velocity field reconstructed in {t1-t0:g} seconds")
    return
  
  def add_field(self, name:str, values:np.ndarray) -> None:
    """
    Add a field defined on the Yin and Yang grids to the mesh. 
    The field can be either a scalar field or a vector field.
    
    :param str name: 
      The name of the field to be added to the mesh.
      This field will then be accessible in the mesh as ``self[name]``.
    :param numpy.ndarray values:
      The field values defined on the Yin and Yang grids.
      For a scalar field, the array should have shape ``(nx, ny, nz, 2)``, 
      where the first three dimensions correspond to the grid dimensions 
      and the last dimension corresponds to the Yin and Yang grids (0 for Yin and 1 for Yang).
      For a vector field, the array should have shape ``(3, nx, ny, nz, 2)``, 
      where the first dimension corresponds to the vector components, 
      the next three dimensions correspond to the grid dimensions and
      the last dimension corresponds to the Yin and Yang grids (0 for Yin and 1 for Yang).

    """
    t0 = perf_counter()
    n            = self.grid_dimensions
    ppl          = self.points_per_layer
    npoints_grid = self.grid_npoints
    gidx         = self.good_indices
    dim = {0: 'x', 1:'y', 2:'z'}
    yin_fields  = {}
    yang_fields = {}
    if values.ndim == 5: # vector field
      for d in dim:
        # Yin grid
        yin_fields[name+dim[d]] = np.reshape(values[d,0:n[0],0:n[1],:,0],(npoints_grid))
        yin_fields[name+dim[d]] = yin_fields[name+dim[d]][gidx]
        yin_fields[name+dim[d]] = np.reshape(yin_fields[name+dim[d]],(ppl, n[2]))
        # Yang grid
        yang_fields[name+dim[d]] = np.reshape(values[d,0:n[0],0:n[1],:,1],(npoints_grid))
        yang_fields[name+dim[d]] = yang_fields[name+dim[d]][gidx]
        yang_fields[name+dim[d]] = np.reshape(yang_fields[name+dim[d]],(ppl, n[2]))
    else: # scalar field
      # Yin grid
      yin_fields[name] = np.reshape(values[0:n[0],0:n[1],:,0],(npoints_grid))
      yin_fields[name] = yin_fields[name][gidx]
      yin_fields[name] = np.reshape(yin_fields[name],(ppl, n[2]))
      # Yang grid
      yang_fields[name] = np.reshape(values[0:n[0],0:n[1],:,1],(npoints_grid))
      yang_fields[name] = yang_fields[name][gidx]
      yang_fields[name] = np.reshape(yang_fields[name],(ppl, n[2]))

    # Gather fields into the mesh
    if values.ndim == 5: # vector field
      self[name] = np.zeros( (2*ppl*n[2], 3) )
      for d in dim:
        # Yin grid
        self[name][0:ppl*n[2],d] = np.reshape(yin_fields[name+dim[d]],(ppl*n[2]),order='F')
        # Yang grid
        self[name][ppl*n[2]:2*ppl*n[2],d] = np.reshape(yang_fields[name+dim[d]],(ppl*n[2]),order='F')
    else: # scalar field
      self[name] = np.zeros( (2*ppl*n[2]) )
      # Yin grid
      self[name][0:ppl*n[2]] = np.reshape(yin_fields[name],(ppl*n[2]),order='F')
      # Yang grid
      self[name][ppl*n[2]:2*ppl*n[2]] = np.reshape(yang_fields[name],(ppl*n[2]),order='F')

    t1 = perf_counter()
    print(f"Added field {name} to the mesh in {t1-t0:g} seconds")
    return
  
  def add_fields(self, fields:dict) -> None:
    """
    Add multiple fields defined on the Yin and Yang grids to the mesh.

    :param dict fields:
      A dictionary containing the fields to be added to the mesh.
      The keys of the dictionary are the names of the fields and 
      the values are the field values defined on the Yin and Yang grids.
      The field values should be in the same format as described in 
      the :py:meth:`add_field <stagpyviz.YinYangMesh.add_field>` method.

    """
    for key in fields.keys():
      self.add_field(key, fields[key])
    return
  
  def compute_gradient(self, field:np.ndarray) -> np.ndarray:
    """
    Compute the Cartesian gradient of a scalar field defined on the mesh 
    using the shape functions of the wedge elements.

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
    elidx = self.cell_connectivity.reshape((self.number_of_cells, self.nodes_per_el))
    elcoords = self.points[ elidx, : ]  # (number_of_cells, nodes_per_el, 3)
    GNi = self.elements.GNi_centroid()
    J = self.elements.evaluate_Jacobian(GNi, elcoords)  # (number_of_cells, 3, 3)
    detJ = self.elements.evaluate_detJ(J)
    invJ = self.elements.evaluate_invJ(J, detJ)
    dNdx = self.elements.evaluate_dNidx(invJ, GNi)  # (number_of_cells, nodes_per_el, 3)
    # Gather field values at element nodes
    field_el = field[ elidx ]  # (number_of_cells, nodes_per_el)
    # Compute gradient at element centroids
    grad_field = np.einsum('eki,ek->ei', dNdx, field_el)
    t1 = perf_counter()
    print(f"Gradient computation performed in {t1-t0:g} seconds")
    return grad_field

def test():
  import os
  rawbin_file = os.path.join(os.environ["POSTPROC"],'Stagyy','PJB6_YS1_Rh32_vp00039') 
  rawT_file   = os.path.join(os.environ["POSTPROC"],'Stagyy','PJB6_YS1_Rh32_t00039')
  mesh_file   = os.path.join(os.environ["SOFTS"],"StagYY","stagpyviz","yinyang_wedge.vtu")
  mesh_with_fields = os.path.join(os.environ["SOFTS"],"StagYY","stagpyviz","yinyang_wedge_with_fields.vtu")

  mesh:YinYangMesh = YinYangMesh(rawbin_file)
  
  with open(rawbin_file,'rb') as f:
    bh = BinHeader(f)
    bh.read_header()
    header = bh.header
    flds = bh.read_fields()
  del bh
  fields = {}
  fields["velocity"] = flds[0:3,:,:,:,:]
  mesh.reconstruct_velocity(fields["velocity"])
  fields["pressure"] = flds[3,:,:,:,:]
  
  
  with open(rawT_file,'rb') as f:
    bh = BinHeader(f)
    bh.read_header()
    headerT = bh.header
    fldsT = bh.read_fields()
  del bh
  fields["temperature"] = fldsT[0,:,:,:,:]
  print(fields["velocity"].shape)
  print(fields["pressure"].shape)
  print(fields["temperature"].shape)
  
  mesh.add_fields(fields)
  # get the surface mesh and velocity field on it
  surface_mesh = mesh.surface_mesh
  surface_mesh["velocity"] = mesh["velocity"][mesh.surface_idx,:]

  plotter = pvs.Plotter()
  plotter.add_mesh(surface_mesh, scalars="velocity", cmap="viridis", show_scalar_bar=True)
  plotter.show()
  #mesh["velocity_r"] = mesh.vector_cartesian_to_spherical(mesh["velocity"])
  
  #mesh.save(mesh_with_fields)
  return

if __name__ == "__main__":
  test()