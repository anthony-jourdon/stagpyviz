import numpy as np
from pyvista import StructuredGrid

try:
  from .femesh import FEMesh
  from ..elements import Q1_3D
except ImportError:
  from stagpyviz.mesh import FEMesh
  from stagpyviz.elements import Q1_3D

class FEStructuredMesh(FEMesh, StructuredGrid):
  """
  Class representing a structured finite element mesh in 3D.
  Uses :math:`\\mathcal{Q}_1` elements (:py:class:`Q1_3D`).
  """
  def __init__(self, *args, deep:bool=False, **kwargs):
    self.elements = Q1_3D()
    super().__init__(*args, deep=deep, **kwargs)
    self._node_grid:tuple[np.ndarray, np.ndarray, np.ndarray]|None    = None
    self._element_grid:tuple[np.ndarray, np.ndarray, np.ndarray]|None = None
    return

  @property
  def node_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid of node indices in 3D.
    Tuple of three 3D arrays of shape ``(nk, nj, ni)``, 
    containing the indices of the nodes in the mesh, 
    where ``nk``, ``nj``, and ``ni`` are the dimensions of the nodes in the mesh 
    in the k, j, and i directions, respectively.
    """
    if self._node_grid is None:
      n = self.dimensions
      i = np.arange(n[0], dtype=np.int32)
      j = np.arange(n[1], dtype=np.int32)
      k = np.arange(n[2], dtype=np.int32)
      self._node_grid = np.meshgrid(k, j, i, indexing='ij')
    return self._node_grid

  @property
  def element_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid of element indices in 3D.
    Tuple of three 3D arrays of shape ``(nek, nej, nei)``, 
    containing the indices of the elements in the mesh, 
    where ``nek``, ``nej``, and ``nei`` are the dimensions of the elements in the mesh 
    in the k, j, and i directions, respectively.
    """
    if self._element_grid is None:
      n = np.asarray(self.dimensions, dtype=np.int32)
      nel = n - 1
      iel = np.arange(nel[0], dtype=np.int32)
      jel = np.arange(nel[1], dtype=np.int32)
      kel = np.arange(nel[2], dtype=np.int32)
      self._element_grid = np.meshgrid(kel, jel, iel, indexing='ij')
    return self._element_grid

  def create_elements2vertices(self):
    """
    Create the elements-to-vertices mapping for the structured mesh.
    The mapping is stored in :py:attr:`elidx <stagpyviz.FEMesh.elidx>` attribute.
    """
    n   = np.asarray(self.dimensions, dtype=np.int32)
    nel = n - 1
    K,J,I = self.element_grid
    ie2v = I + J*n[0] + K*n[0]*n[1]
    ie2v = ie2v.flatten()
    elidx_v = np.zeros((nel[0]*nel[1]*nel[2], 8), dtype=np.int32)
    elidx_v[:,0] = ie2v
    elidx_v[:,1] = ie2v + 1
    elidx_v[:,2] = ie2v + n[0]
    elidx_v[:,3] = ie2v + n[0] + 1
    elidx_v[:,4] = ie2v + n[0]*n[1]
    elidx_v[:,5] = ie2v + n[0]*n[1] + 1
    elidx_v[:,6] = ie2v + n[0]*n[1] + n[0]
    elidx_v[:,7] = ie2v + n[0]*n[1] + n[0] + 1
    self._elidx = elidx_v
    return

  def get_layer_indices(self, axis:str, index:int, n:list|tuple|np.ndarray, grid:np.ndarray) -> np.ndarray:
    axis_dict = {'i': 2, 'j': 1, 'k': 0}
    # check that the required axis is valid
    if axis not in axis_dict:
      raise ValueError(f"Invalid axis '{axis}'. Choose from 'i', 'j', or 'k'.")

    # get the position of the required layer
    direction = axis_dict[axis]
    # check that the index is within bounds
    if index < 0 or index >= n[ direction ]:
      raise ValueError(f"Index {index} is out of bounds for axis '{axis}' with size {n[direction]}.")

    # get the structured grid indices
    K,J,I = grid

    # create a list of slices for each dimension
    slices = [slice(None)] * 3
    # set the slice for the specified axis to the given index
    slices[ direction ] = index

    # extract the layer indices
    indices = [K, J, I]
    layers  = 3*[None]
    for d in range(3):
      if d == direction:
        layers[d] = index
      else:
        layers[d] = indices[d][tuple(slices)].flatten()
    # generate the indices for the specified layer
    nidx = layers[2] + layers[1]*n[0] + layers[0]*n[0]*n[1]
    return nidx

  def get_layer_nodes(self, axis:str, index:int) -> np.ndarray:
    """
    Returns the indices of the nodes in a specified layer of the structured mesh.
    The layer is specified by the axis (``'i'``, ``'j'``, or ``'k'``) and the index of the layer along that axis.

    :param str axis: 
      The axis along which to select the layer (``'i'``, ``'j'``, or ``'k'``).
    :param int index:
      The index of the layer along the specified axis. 
      Must be in the range ``[0, n[axis]-1]``, where ``n[axis]`` is the number of 
      nodes along that axis.
    :return:
      A 1D numpy array containing the indices of the nodes in the specified layer.
    :rtype: numpy.ndarray
    """
    n    = np.asarray(self.dimensions, dtype=np.int32)    
    grid = self.node_grid
    return self.get_layer_indices(axis, index, n, grid)

  def get_layer_elements(self, axis:str, index:int) -> np.ndarray:
    """
    Returns the indices of the elements in a specified layer of the structured mesh.
    The layer is specified by the axis (``'i'``, ``'j'``, or ``'k'``) and the index of the layer along that axis.

    :param str axis: 
      The axis along which to select the layer (``'i'``, ``'j'``, or ``'k'``).
    :param int index:
      The index of the layer along the specified axis. 
      Must be in the range ``[0, n[axis]-1]``, where ``n[axis]`` is the number of 
      elements along that axis.
    :return:
      A 1D numpy array containing the indices of the elements in the specified layer.
    :rtype: numpy.ndarray
    """
    n    = np.asarray(self.dimensions, dtype=np.int32)
    nel  = n - 1
    grid = self.element_grid
    return self.get_layer_indices(axis, index, nel, grid)

  def extract_subset(
      self, 
      voi:list[int]|tuple[int]|np.ndarray, 
      rate:tuple[int]|list[int]|np.ndarray=(1, 1, 1), 
      boundary:bool=False
    ):
    """
    Extract a subset of the structured mesh. Overloads
    :py:meth:`extract_subset <pyvista.StructuredGrid.extract_subset>` 
    to return a new instance of 
    :py:class:`FEStructuredMesh` instead of a 
    :py:class:`pyvista.StructuredGrid`.
    See :py:meth:`extract_subset <pyvista.StructuredGrid.extract_subset>` 
    for more details.

    :param list[int]|tuple[int]|numpy.ndarray voi: 
      Length 6 iterable of ``int``
      s: ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
      These bounds specify the volume of interest in i-j-k min/max
      indices.
    :param list[int]|tuple[int]|numpy.ndarray rate: 
      Length 3 iterable of ``int``
      s: ``(xrate, yrate, zrate)``.
      Defaults to ``(1, 1, 1)``. 
      These values specify the sampling rate in each direction.
    :param bool boundary: 
      Control whether to enforce that the "boundary" of the grid
      is output in the subsampling process. (This only has
      effect when the rate in any direction is not equal to
      1). When this is on, the subsampling will always include
      the boundary of the grid even if the sample rate is
      not an even multiple of the grid dimensions.
    :return: A new :py:class:`FEStructuredMesh` mesh containing the extracted subset.
    :rtype: FEStructuredMesh
    """
    mesh = super().extract_subset(voi, rate=rate, boundary=boundary)
    result = FEStructuredMesh(mesh.x, mesh.y, mesh.z, deep=True)
    # Copy any point/cell data from original mesh
    for key, value in mesh.point_data.items():
      result.point_data[key] = value
    for key, value in mesh.cell_data.items():
      result.cell_data[key] = value
    return result