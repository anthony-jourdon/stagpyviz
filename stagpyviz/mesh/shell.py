from pathlib import Path
from functools import cached_property
from time import perf_counter
import pyvista as pvs
import numpy as np
from scipy.spatial import ConvexHull

try:
  from ..elements.p1_2d import P1_2D_R3
except ImportError:
  from stagpyviz.elements.p1_2d import P1_2D_R3

class ShellMesh(pvs.UnstructuredGrid):
  def __init__(self, *args, deep:bool=False, **kwargs):
    self.elements:P1_2D_R3 = P1_2D_R3()
    self._centroids = None

    # Check if user provided a VTU file to load the mesh from
    if len(args) == 1 and isinstance(args[0], (Path, str)):
      if (args[0].endswith(".vtu")):
        t0 = perf_counter()
        mesh:pvs.UnstructuredGrid = pvs.read(args[0])
        elidx = mesh.cell_connectivity.reshape((mesh.number_of_cells, 3))
        oriented_elidx = self._orient_triangles(elidx, mesh.points).reshape((mesh.number_of_cells*3))
        super().__init__({pvs.CellType.TRIANGLE: oriented_elidx}, mesh.points)
        # copy point and cell data from original mesh
        for name in mesh.point_data:
          self.point_data[name] = mesh.point_data[name]
        for name in mesh.cell_data:
          self.cell_data[name] = mesh.cell_data[name]
        del mesh
        t1 = perf_counter()
        print(f"VTU file loaded in {t1-t0:g} seconds")
    elif len(args) == 1 and isinstance(args[0], pvs.UnstructuredGrid):
      super().__init__(args[0], deep=deep, **kwargs)
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
      t0 = perf_counter()
      points = args[0]
      if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points array must be of shape (n_points, 3), found {points.shape}")
      # Create a convex hull to define the surface mesh
      hull = ConvexHull(points)
      oriented_elidx = self._orient_triangles(hull.simplices, points).reshape((hull.simplices.shape[0]*3))
      super().__init__({pvs.CellType.TRIANGLE: oriented_elidx}, points)
      t1 = perf_counter()
      print(f"Shell mesh created from points in {t1-t0:g} seconds")
    else:
      raise ValueError("ShellMesh constructor requires a VTU file path, an UnstructuredGrid, or a points array.")
    return
  
  @property
  def centroids(self) -> np.ndarray:
    if self._centroids is None:
      elidx = self.cell_connectivity.reshape((self.number_of_cells, self.elements.basis_per_el))
      elcoords = self.points[elidx, :]
      Ni = self.elements.Ni_centroid()
      centroids = np.einsum('k,eki->ei', Ni, elcoords)
      self._centroids = centroids
    return self._centroids

  def compute_normals(self, point_normals:bool=True, cell_normals:bool=True):
    if point_normals:
      self.point_data["normals"] = np.zeros((self.number_of_points, 3), dtype=np.float64)
      R = np.sqrt(np.sum(self.points**2, axis=1))
      self.point_data["normals"] = self.points / R[:, np.newaxis]
    if cell_normals:
      self.cell_data["normals"] = np.zeros((self.number_of_cells, 3), dtype=np.float64)
      R = np.sqrt(np.sum(self.centroids**2, axis=1))
      self.cell_data["normals"] = self.centroids / R[:, np.newaxis]
    return
  
  @property
  def points_normal(self) -> np.ndarray:
    if "normals" not in self.point_data:
      self.compute_normals(point_normals=True, cell_normals=False)
    return self.point_data["normals"]
  
  @property
  def cells_normal(self) -> np.ndarray:
    if "normals" not in self.cell_data:
      self.compute_normals(point_normals=False, cell_normals=True)
    return self.cell_data["normals"]
  
  @staticmethod
  def _orient_triangles(elidx:np.ndarray, points:np.ndarray) -> np.ndarray:
    t0 = perf_counter()
    element = P1_2D_R3()
    facets = np.copy(elidx)
    elcoords = points[facets, :]  # (n_cells, 3, 3)
    J = element.evaluate_Jacobian(element.GNi_centroid(), elcoords)
    normal = element.normal_vector_nonu(J)  # (n_cells, 3)
    centroids = np.einsum('k,eki->ei', element.Ni_centroid(), elcoords)
    dot = np.einsum('ei,ei->e', normal, centroids)  # Find elements where dot < 0
    bad = dot < 0.0
    print(f"Found {np.sum(bad)} elements with negative orientation")
    if np.any(bad):
      # Swap columns 1 and 2 for elements with negative orientation
      temp = facets[bad, 1].copy()
      facets[bad, 1] = facets[bad, 2]
      facets[bad, 2] = temp
    t1 = perf_counter()
    print(f"Oriented {np.sum(bad)} elements in {t1-t0:g} seconds")
    return facets

def test():
  import os
  basedir = os.path.join(os.environ["M3D_DIR"],"Stagyy","Pandora","llsvp-surface")
  fname = "step00010_surface.vtu"

  mesh:ShellMesh = ShellMesh(os.path.join(basedir, fname))

  # check that we do not have flipped elements anymore
  elidx = mesh.cell_connectivity.reshape((mesh.number_of_cells, 3))
  
  import matplotlib.pyplot as plt
  import matplotlib.tri as mtri


  
  #plotter = pvs.Plotter()
  #plotter.add_mesh(mesh, scalars="temperature", cmap="RdYlBu_r")
  #plotter.show()

  return

if __name__ == "__main__":
  test()