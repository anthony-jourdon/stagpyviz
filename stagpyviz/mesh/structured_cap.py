from pathlib import Path
import pyvista as pvs
import numpy as np

from time import perf_counter

try:
  from ..parsers import read_stag_bin
  from .structured import FEStructuredMesh
  from .spherical_3d import SphericalGrid
except ImportError:
  from stagpyviz.parsers import read_stag_bin
  from stagpyviz.mesh import FEStructuredMesh
  from stagpyviz.mesh import SphericalGrid

class StructuredCapMesh(FEStructuredMesh, SphericalGrid):
  """
  Class representing a structured finite element mesh in 3D, with a spherical cap geometry.
  Uses :math:`\\mathcal{Q}_1` elements (:py:class:`Q1_3D`).

  The mesh can either be created from a StaggYY binary file or loaded from a VTS file.
  If no StaggYY binary file is provided, the mesh can still be created from the VTS file, 
  but some functionalities may be unavailable.

  :param args: 
    Additional positional arguments to pass to the parent class. 
    Can be used to provide a VTS file to load the mesh from.
  :param str|Path rawbin: 
    Path to the StaggYY binary file to read the mesh from. 
  :param deep: 
    If ``True``, a deep copy of the mesh will be created. If False, a shallow copy will be created. 
    Default is ``False``.
  :param kwargs: 
    Additional keyword arguments to pass to the parent class.
  """
  def __init__(self, *args, rawbin:str|Path|None=None, deep:bool=False, **kwargs):
    self.header:dict|None = None
    if rawbin is None:
      # check if user provided the header information in the kwargs
      self.header = kwargs.pop("header", None)
      if self.header is None:
        print(
          "WARNING: No StaggYY binary file provided, or no header information provided in the kwargs. "
          "The mesh will be created from the VTS file only, but some functionalities may be unavailable."
        )
    else:
      self.header, _ = read_stag_bin(rawbin)
    
    # Check if user provided a VTS file to load the mesh from
    if len(args) == 1 and isinstance(args[0], (Path, str)):
      if (args[0].endswith(".vts")):
        t0 = perf_counter()
        mesh:pvs.StructuredGrid = pvs.read(args[0])
        super().__init__(mesh, deep=deep, **kwargs)
        t1 = perf_counter()
        print(f"VTS file loaded in {t1-t0:g} seconds")
    elif rawbin is not None:
      # Create the mesh from the header information
      t0 = perf_counter()
      self.create_mesh()
      t1 = perf_counter()
      print(f"Mesh created in {t1-t0:g} seconds")
    else:
      # attempt to create the mesh from given arguments, if any
      t0 = perf_counter()
      super().__init__(*args, deep=deep, **kwargs)
      t1 = perf_counter()
      print(f"Mesh created in {t1-t0:g} seconds")
    
    self.create_elements2vertices()
    return

  def create_mesh(self):
    if self.header is None:
      raise ValueError("No StaggYY binary file provided. Cannot create mesh from nothing.")
    # 1D arrays of coordinates (Stag system)
    x = np.asarray(self.header['x'])
    y = np.asarray(self.header['y'])
    z = np.asarray(self.header['z'])
    # Grid of coordinates in 3D
    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
    # convert to spherical coordinates
    R   = Z + self.header["rcmb"]
    lat = 0.25*np.pi - X
    lon = Y - 0.75*np.pi
    # convert to Cartesian coordinates
    X,Y,Z = self.spherical_to_cartesian(R, lat, lon)
    # Create the structured grid
    super().__init__(X, Y, Z)
    return

  def add_field(self, name:str, values:np.ndarray) -> None:
    """
    Add a field to the mesh.
    
    :param str name: Name of the field to add.
    :param numpy.ndarray values: 
      Field values to add. Should be a 3D array of shape ``(ni, nj, nk)``, 
      where ``ni``, ``nj``, and ``nk`` are the dimensions of the nodes in the mesh.
      The array is expected to be in Fortran order.
    """
    t0 = perf_counter()
    n = self.dimensions
    if values.ndim == 5: # vector field
      bs = values.shape[0]
      self[name] = np.zeros((n[0]*n[1]*n[2], bs), dtype=np.float64)
      for d in range(bs):
        self[name][:, d] = np.reshape(values[d, 0:n[0], 0:n[1], :, 0], (n[0]*n[1]*n[2]), order='F')
    else:
      self[name] = np.reshape(values[0:n[0], 0:n[1], :, 0], (n[0]*n[1]*n[2]), order='F')
    t1 = perf_counter()
    print(f"Field '{name}' added in {t1-t0:g} seconds")
    return

  def reconstruct_velocity(self, velocity_raw:np.ndarray) -> None:
    """
    .. warning::
      This method cannot be used if no binary file was provided to create the mesh, 
      as the header information is required to reconstruct the velocity field.

    Reconstruct the velocity field from the raw velocity components defined on the grid.
    First the raw coordinates of the points are transformed such that:

    .. math::
      \\begin{split}
        r &= z_{\\text{raw}} + r_{\\text{cmb}} \\\\
        \\theta &= \\frac{\\pi}{4} - x_{\\text{raw}} \\\\
        \\phi &= y_{\\text{raw}} - \\frac{3\\pi}{4}
      \\end{split}

    Then the velocity components are transformed from the raw grid coordinates to Cartesian coordinates:

    .. math::
      \\mathbf{v} = 
      \\begin{bmatrix}
        v_{0} \\sin(\\theta) \\cos(\\phi) - v_{1} \\sin(\\phi) + v_{2} \\cos(\\theta) \\cos(\\phi) \\\\
        v_{0} \\sin(\\theta) \\sin(\\phi) + v_{1} \\cos(\\phi) + v_{2} \\cos(\\theta) \\sin(\\phi) \\\\
        -v_{0} \\cos(\\theta) + v_{2} \\sin(\\theta)
      \\end{bmatrix}

    :param numpy.ndarray velocity_raw:
      Raw velocity components defined on the grid. 
      Should be a 4D array of shape ``(3, ni, nj, nk, 1)``, 
      where ``ni``, ``nj``, and ``nk`` are the dimensions of the nodes in the mesh.
    :return: None. The input array representing the velocity field is modified in place.
    :rtype: None
    """
    t0 = perf_counter()
    if self.header is None:
      raise ValueError("No StaggYY binary file provided. Cannot reconstruct velocity from nothing.")
    header = self.header
    x = np.asarray(header['x'])
    y = np.asarray(header['y'])
    z = np.asarray(header['z'])
    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
    #R   = Z + header["rcmb"]
    lat = 0.25*np.pi - X
    lon = Y - 0.75*np.pi
    n = self.dimensions

    v0 = velocity_raw[0,0:n[0],0:n[1],:,0]
    v1 = velocity_raw[1,0:n[0],0:n[1],:,0]
    v2 = velocity_raw[2,0:n[0],0:n[1],:,0]

    vx = v0*np.sin(lat)*np.cos(lon) - v1*np.sin(lon) + v2*np.cos(lat)*np.cos(lon)
    vy = v0*np.sin(lat)*np.sin(lon) + v1*np.cos(lon) + v2*np.cos(lat)*np.sin(lon)
    vz = -v0*np.cos(lat) + v2*np.sin(lat)
    velocity_raw[0,0:n[0],0:n[1],:,0] = vx
    velocity_raw[1,0:n[0],0:n[1],:,0] = vy
    velocity_raw[2,0:n[0],0:n[1],:,0] = vz
    t1 = perf_counter()
    print(f"Velocity reconstructed in {t1-t0:g} seconds")
    return

  def extract_subset(
      self, 
      voi:list[int]|tuple[int]|np.ndarray, 
      rate:tuple[int]|list[int]|np.ndarray=(1, 1, 1), 
      boundary:bool=False
    ):
    mesh = super().extract_subset(voi, rate=rate, boundary=boundary)
    # create a new StructuredCapMesh object from the extracted subset
    result = StructuredCapMesh(mesh.x, mesh.y, mesh.z, deep=True)
    # copy the fields from the original mesh to the new mesh
    for key, value in mesh.point_data.items():
      result.point_data[key] = value
    for key, value in mesh.cell_data.items():
      result.cell_data[key] = value
    return result