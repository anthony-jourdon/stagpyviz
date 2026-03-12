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
  """
  Class representing the surface mesh of a spherical shell, which can be loaded from a VTU file or created from a point cloud.
  Inherits from :py:class:`UnstructuredSphere` and `pyvista.UnstructuredGrid`_.
  
  The mesh is represented as a collection of triangular facets defining :math:`\\mathcal P_1` elements in :math:`\\mathbb R^3`
  using the class :py:class:`P1_2D_R3`.

  The constructor can be called in three ways:

  1. 
    With a single argument that is a path to a VTU file containing an unstructured grid. 
    The mesh will be loaded from the file, and point and cell data will be copied if available.

  2. 
    With a single argument that is a :py:class:`pyvista.UnstructuredGrid` object. 
    The mesh will be created from the provided unstructured grid, and point and cell data will be copied if available.

  3. 
    With a single argument that is a :py:class:`numpy.ndarray` of shape ``(number_of_points, 3)`` containing the coordinates of points. 
    The mesh will be created as the convex hull of the provided points, and a ``"neighbors"`` array representing the 
    connectivity of the facets will be added to the cell data.

  :Attributes:

  .. py:attribute:: elements
    
    The isoparametric element class representing triangular facets in 3D space.

    :type: :py:class:`P1_2D_R3 <stagpyviz.P1_2D_R3>`

  .. py:attribute:: points_normal

    An array of shape ``(number_of_points, 3)`` containing the normal vectors at each point.
    The normals are evaluated as the normalized position vectors of the points.

    :type: numpy.ndarray

  .. py:attribute:: cells_normal

    An array of shape ``(number_of_cells, 3)`` containing the normal vectors at each cell (facet).
    The normals are evaluated as the normalized position vectors of the cell centroids.

    :type: numpy.ndarray

  .. py:attribute:: neighbors

    An array of shape ``(number_of_cells, 3)`` containing the indices of neighboring cells across each facet.
    This attribute is only available if the mesh was created from a points array or if the VTU file from which the mesh was loaded contained this information,
    and is stored in the cell data under the key ``"neighbors"``.

    :type: numpy.ndarray

  .. py:attribute:: cells_area

    An array of shape ``(number_of_cells,)`` containing the area of each triangular facet.
    The area is computed using the determinant of the Jacobian of the transformation from the reference element to the physical element such that:
    :math:`A_e = \\frac{1}{2} |\\det(J_e)|`.

    :type: numpy.ndarray


  """
  def __init__(self, *args, deep:bool=False, **kwargs):
    self.elements:P1_2D_R3 = P1_2D_R3()
    self._cells_area = None

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

  @property
  def cells_area(self) -> np.ndarray:
    if self._cells_area is None:
      # Compute the area of each triangle
      elidx = self.cell_connectivity.reshape((self.number_of_cells, self.elements.basis_per_el))
      el_coords = self.points[elidx, :]
      self._cells_area = self.elements.compute_area(el_coords)
    return self._cells_area

  def integrate_per_cell(self, field:np.ndarray) -> np.ndarray:
    """
    Compute the numerical integral of a field over each cell of the mesh using a 1 point quadrature rule.
    The field can be either a point field (defined at mesh points) or a cell field (defined at mesh cells). 
    If the field is a point field, it will be interpolated to cell centroids using the shape functions 
    of the elements evaluated at the cell centroids before integration. 
    The integral over each cell is computed such that:

    .. math:: 
      I_e = \\int_{\\Omega_e} \\phi_e \\, dS \\approx \\phi_e A_e
    
    where :math:`\\phi_e` is the value of the field at the cell centroid and :math:`A_e` is the area of the cell.

    :param field: A 1D array containing the values of the field to integrate, either at points or at cells.
    :type field: numpy.ndarray
    :return: A 1D array of shape ``(number_of_cells,)`` containing the integral of the field over each cell.
    :rtype: numpy.ndarray
    """
    elidx = self.cell_connectivity.reshape((self.number_of_cells, self.elements.basis_per_el))
    if not self.is_cell_field(field):
      if self.is_point_field(field):
        # first interpolate point field to cell centroids
        Ni_centroid = self.elements.Ni_centroid() # shape functions at cell centroids
        _field = np.einsum('k,ek->e', Ni_centroid, field[elidx]) # interpolate field at cell centroids
      else:
        raise ValueError(f"Field must be either a point field ({self.number_of_points}) or a cell field ({self.number_of_cells}), found {field.shape}.")
    else:
      _field = field
    integral = _field * self.cells_area
    return integral

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
  basedir = os.path.join("/data","jourdon","llsvp-pandora-vtu","surface-yy","LLSVP-Run3")
  fname = "step00050-surface.vtu"

  mesh:ShellMesh = ShellMesh(os.path.join(basedir, fname))
  # Compute the area of each triangle using the Jacobian determinant
  A1 = mesh.cells_area
  # Compute the area of each triangle using the cross product of edge vectors
  A2 = np.zeros(mesh.number_of_cells, dtype=np.float64)
  elidx = mesh.cell_connectivity.reshape((mesh.number_of_cells, mesh.elements.basis_per_el))
  elcoords = mesh.points[elidx, :]
  v0 = elcoords[:,0,:]
  v1 = elcoords[:,1,:]
  v2 = elcoords[:,2,:]
  A2 = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
  print(f"Max relative difference in area: {np.max(np.abs(A1-A2)/A2):g}")


  #plotter = pvs.Plotter()
  #plotter.add_mesh(mesh, scalars="temperature", cmap="RdYlBu_r")
  #plotter.show()

  return

if __name__ == "__main__":
  test()