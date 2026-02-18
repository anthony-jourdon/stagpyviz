from pathlib import Path
from time import perf_counter
import pyvista as pvs
import numpy as np
from scipy.spatial import ConvexHull

try:
  from ..elements.p1_2d import P1_2D_R3
  from .spherical_3d import UnstructuredSphere
except ImportError:
  from stagpyviz.elements.p1_2d import P1_2D_R3
  from stagpyviz.mesh.spherical_3d import UnstructuredSphere

class ShellMesh(UnstructuredSphere):
  def __init__(self, *args, deep:bool=False, **kwargs):
    self.elements:P1_2D_R3 = P1_2D_R3()

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
      self.cell_data["neighbors"] = hull.neighbors
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
      centroids = self.elements.evaluate_element_centroid(elcoords)
      self._centroids = centroids
    return self._centroids

  def compute_normals(self, point_normals:bool=True, cell_normals:bool=True):
    if point_normals:
      x = self.points
      self.point_data["normals"] = np.zeros((self.number_of_points, 3), dtype=np.float64)
      R = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
      self.point_data["normals"] = np.divide(x.T, R).T
    if cell_normals:
      x = self.centroids
      self.cell_data["normals"] = np.zeros((self.number_of_cells, 3), dtype=np.float64)
      R = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
      self.cell_data["normals"] = np.divide(x.T, R).T
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
  
  @property
  def neighbors(self) -> np.ndarray:
    if "neighbors" not in self.cell_data:
      raise ValueError("Neighbors information not available in cell data.")
    return self.cell_data["neighbors"]

  def locate_points(self, points:np.ndarray, max_it:int=1000, tol:float=1e-12) -> tuple[np.ndarray, np.ndarray]:
    print("WARNING: ShellMesh.locate_points may be inaccurate for points near element boundaries.")
    npoints = points.shape[0]
    elidx    = self.cell_connectivity.reshape((self.number_of_cells, self.elements.basis_per_el))
    elcoords = self.points[elidx, :]
    # array to store the index of the containing element for each point
    econtaining  = np.ones(npoints, dtype=np.int64) * -1
    local_coords = np.zeros((npoints, 3), dtype=np.float64) # barycentric coordinates of points in containing elements
    # local coordinates derivatives (constant for P1)
    GNi = self.elements.GNi_centroid()
    # Surface Jacobian matrix
    J = self.elements.evaluate_Jacobian(GNi, elcoords)
    # Jacobian determinant (2*area of the element)
    detJ = self.elements.evaluate_detJ(J)
    # Eclidian derivatives of shape functions, note the scaling by detJ 
    dNdx = self.elements.evaluate_dNidx(J, GNi) * detJ[:, np.newaxis, np.newaxis] 
    # First vertex of each element
    v0 = elcoords[:,0,:]
    
    for p in range(npoints):
      point = points[p]
      # initial guess
      # distance between point and element centroids
      d2 = np.sum((self.centroids - point)**2, axis=1)
      # index of closest element centroid
      start_e = np.argmin(d2)
      # locate point in mesh starting from initial guess
      it = 0
      e  = start_e
      visited = set()
      while it < max_it:
        # Compute barycentric coordinates of point in current element
        # Project point onto plane of the element
        point_proj = point - np.dot(point - v0[e], self.cells_normal[e]) * self.cells_normal[e]
        d = point_proj - v0[e] # vector from first vertex of element to point
        lam = np.empty(3)
        lam[0] = 1.0 + np.dot(dNdx[e, 0], d)
        lam[1] =       np.dot(dNdx[e, 1], d)
        lam[2] =       np.dot(dNdx[e, 2], d)
        if np.all(lam >= -tol):
          # Point is inside the element
          lam /= np.sum(lam) # ensure partition of unity
          econtaining[p] = e
          local_coords[p,:] = lam
          break
        if e in visited:
          if np.min(lam) >= -1e-3:
            lam /= np.sum(lam) # ensure partition of unity
            econtaining[p] = e
            local_coords[p,:] = lam
            break
          else:
            print(f"Warning: Point {p} ({point}) location is cycling through elements, stopping search.")
            break
        visited.add(e)
        # Point is outside the element, find the facet with the most negative barycentric coordinate
        k = np.argmin(lam)        # most negative barycentric coord
        e = self.neighbors[e, k]  # move across opposite edge
        if e < 0: # outside mesh
          break
        it += 1
      if it == max_it:
        print(f"Warning: Point {p} ({point}) location did not converge after {max_it} iterations.")
    return econtaining, local_coords

  @staticmethod
  def _orient_triangles(elidx:np.ndarray, points:np.ndarray) -> np.ndarray:
    t0 = perf_counter()
    element = P1_2D_R3()
    facets = np.copy(elidx)
    elcoords = points[facets, :]  # (n_cells, 3, 3)
    J = element.evaluate_Jacobian(element.GNi_centroid(), elcoords)
    normal = element.normal_vector_nonu(J)  # (n_cells, 3)
    centroids = element.evaluate_element_centroid(elcoords)
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

  #plotter = pvs.Plotter()
  #plotter.add_mesh(mesh, scalars="temperature", cmap="RdYlBu_r")
  #plotter.show()

  return

if __name__ == "__main__":
  test()