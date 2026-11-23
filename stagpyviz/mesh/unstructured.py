import numpy as np
from pyvista import UnstructuredGrid, CellType

try:
  from .femesh import FEMesh
  from ..elements import P1_2D, P1_2D_R3, Wedge3D, Q1_3D, Q1_2D
except ImportError:
  from stagpyviz.mesh import FEMesh

class FEUnstructuredMesh(FEMesh, UnstructuredGrid):
  """
  Class representing an unstructured finite element mesh in 2D or 3D.
  """
  def __init__(self, *args, deep:bool=False, **kwargs):
    dim = kwargs.pop("dim", 3)
    super().__init__(*args, deep=deep, **kwargs)

    match self.celltypes[0]:
      case CellType.HEXAHEDRON:
        if dim == 2:
          self.elements = Q1_2D()
        elif dim == 3:
          self.elements = Q1_3D()
      case CellType.TRIANGLE:
        if dim == 2:
          self.elements = P1_2D()
        elif dim == 3:
          self.elements = P1_2D_R3()
        else:
          raise ValueError(f"Unsupported dimension {dim} for triangle elements.")
      case CellType.WEDGE:
        self.elements = Wedge3D()
      case _:
        raise ValueError(f"Unsupported cell type {self.celltypes[0]} for FEUnstructuredMesh. Supported types are HEXAHEDRON, TRIANGLE, and WEDGE.")
    self.nodes_per_el = self.elements.basis_per_el

    return

  @property
  def elidx(self) -> np.ndarray:
    """
    Returns the elements-to-vertices mapping for the unstructured mesh.
    The mapping is a 2D numpy array of shape ``(number_of_cells, nodes_per_el)``, 
    where ``number_of_cells`` is the number of cells in the mesh and ``nodes_per_el`` 
    is the number of nodes per element.
    """
    return self.cell_connectivity.reshape((self.number_of_cells, self.nodes_per_el), order='C')