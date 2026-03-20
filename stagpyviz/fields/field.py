try:
  from ..parsers import read_stag_bin
  from ..mesh.yinyang import YinYangMesh
  from ..utils.io_utils import IOutils
  from ..scaling.scaling import Scaling, scaling_factors
except ImportError:
  from stagpyviz.parsers import read_stag_bin
  from stagpyviz.mesh import YinYangMesh
  from stagpyviz.utils import IOutils
  from stagpyviz.scaling import Scaling, scaling_factors

import os
import numpy as np

class Field:
  def __init__(self, name:str, scaling:Scaling|None=None, **kwargs):
    self.name:str             = name
    self.scaling:Scaling|None = scaling
    return

class StagField(Field):
  """
  Generic class to represent a field that can be added to the mesh. 
  By default, this class assumes that the field is stored in a binary file output by StagYY.
  Different types of fields can be created by inheriting from this class and implementing their own 
  methods to retrieve the data and add them to the mesh.

  :param str name: Name of the field, should correspond the the name registered in the IOutils class filelist attribute.
  :param stagpyviz.IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh reconstructed as an unstructured grid
  :param stagpyviz.Scaling|None scaling: 
    Scaling factor for the field, if the field is non-dimensional 
    and needs to be scaled to dimensional units, default: None.
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, scaling)
    self.io_utils:IOutils       = io_utils
    self.mesh:YinYangMesh       = mesh
    self.values:np.ndarray|None = None
    return
  
  def get_data(self) -> np.ndarray|None:
    """
    Extract the field data from the binary file output by StagYY 
    corresponding to the field name and the current step using the function
    :py:func:`read_stag_bin <stagpyviz.read_stag_bin>`.
    If the file cannot be read or does not exist, the function returns None and the field will be ignored.

    :return: A numpy array containing the field data, or None if the file could not be read.
    :rtype: np.ndarray|None
    """
    io_utils = self.io_utils
    fname = f"{io_utils.model}_{io_utils.filelist[self.name]}{str(io_utils.step).zfill(5)}"
    # check file existence
    full_fname = os.path.join(io_utils.model_dir,fname)
    if not os.path.exists(full_fname): 
      print(f"\t\tFile {full_fname} not found, ignoring field {self.name}.")
      return None
    header, data = read_stag_bin(full_fname)
    self.io_utils.time = header["time"]
    return data

  def get_values(self) -> np.ndarray|None:
    """
    Return the field values as a numpy array.
    If :py:meth:`get_data <stagpyviz.StagField.get_data>` returns None, 
    the function returns None and the field will be ignored.
     
    :return: 
      Array of the shape ``(nx, ny, nz, nblock)`` where 
      ``nx``, ``ny``, ``nz`` are the dimensions of the field in each direction and 
      ``nblock`` is the number of blocks in the mesh (2 for Yin-Yang mesh).
    :rtype: np.ndarray|None
    """
    if self.values is not None:
      return self.values
    data = self.get_data()
    if data is None:
      return None
    self.values = data[0,:,:,:,:]
    return self.values

  def add_to_mesh(self) -> None:
    """
    Add the field to the mesh using its name and the 
    :py:meth:`add_field <stagpyviz.YinYangMesh.add_field>` method of the mesh.
    If the field is non-dimensional and a scaling factor is provided, 
    the values are scaled to dimensional units before being added to the mesh.
    """
    if self.values is None:
      values = self.get_values()
    if values is None:
      print(f"Cannot add field '{self.name}' to mesh because its values could not be retrieved.")
      return
    self.mesh.add_field(self.name, values)
    if self.scaling is not None:
      self.mesh[self.name] = self.scaling.dim(self.mesh[self.name])
    return

  def reset(self) -> None:
    """
    Remove the field from the mesh and reset its values to None.
    """
    if self.name in self.mesh.point_data:
      self.mesh.point_data.pop(self.name)
    if self.name in self.mesh.cell_data:
      self.mesh.cell_data.pop(self.name)
    self.values = None
    return

class DerivedField(StagField):
  """
  Class to add a field to the mesh deriving from a :py:class:`StagField <stagpyviz.StagField>` 
  that has already been added to the mesh. 

  :param str name: Name of the field.
  :param stagpyviz.IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh.
  :param stagpyviz.Scaling|None scaling: 
    Scaling factor for the field, if the field is non-dimensional 
    and needs to be scaled to dimensional units, default: None.
  :param str prefix: Prefix to be added to the output file names, default: "".
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, io_utils, mesh, scaling)
    return
  
  def add_to_mesh(self) -> None:
    """
    Add the field to the mesh using its name.
    If the field is non-dimensional and a scaling factor is provided, 
    the values are scaled to dimensional units before being added to the mesh.
    """
    if self.values is None:
      values = self.get_values()
    if values is None:
      print(f"Cannot add field '{self.name}' to mesh because its values could not be retrieved.")
      return
    self.mesh[self.name] = values
    if self.scaling is not None:
      self.mesh[self.name] = self.scaling.dim(self.mesh[self.name])
    return

class Velocity(StagField):
  """
  Class to extract the velocity field from the StagYY binary output.

  :param str name: Name of the field.
  :param stagpyviz.IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh.
  :param stagpyviz.Scaling|None scaling: 
    Scaling factor for the field, if the field is non-dimensional 
    and needs to be scaled to dimensional units, default: None.
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, io_utils, mesh, scaling)
    return
  
  def get_values(self) -> np.ndarray|None:
    """
    Return the velocity field values as a numpy array.
    If it cannot be retrieve the function returns None and the field will be ignored.

    :return: 
      Array of the shape ``(3, nx, ny, nz, nblock)``, 
      where the first dimension corresponds to the 
      three velocity components in Cartesian coordinates.
    :rtype: np.ndarray|None
    """
    if self.values is not None:
      return self.values
    data = self.get_data()
    if data is None:
      return None
    values = data[0:3,:,:,:,:]
    self.mesh.reconstruct_velocity(values)
    self.values = values
    return self.values

class Pressure(StagField):
  """
  Class to extract the pressure field from the StagYY binary output.

  :param str name: Name of the field.
  :param stagpyviz.IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh.
  :param stagpyviz.Scaling|None scaling: 
    Scaling factor for the field, if the field is non-dimensional 
    and needs to be scaled to dimensional units, default: None.
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, io_utils, mesh, scaling)
    return
  
  def get_values(self) -> np.ndarray|None:
    """
    Return the pressure field values as a numpy array.
    If it cannot be retrieve the function returns None and the field will be ignored.

    :return: 
      Array of the shape ``(nx, ny, nz, nblock)``
    :rtype: np.ndarray|None
    """
    if self.values is not None:
      return self.values
    data = self.get_data()
    if data is None:
      return None
    self.values = data[3,:,:,:,:]
    return self.values

class SphericalField(DerivedField):
  """
  Class to add to the mesh a spherical field transformed from a Cartesian field 
  that has already been added to the mesh.

  .. note::
    Spherical fields are derived from Cartesian fields that have already been added to the mesh.
    Therefore, they are automatically scaled if the Cartesian field is scaled.

  :param str name: Name of the field.
  :param stagpyviz.IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh.
  :param stagpyviz.StagField cartesian_field: Cartesian field to be transformed into a spherical field.
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, cartesian_field:StagField):
    # this field should not be scaled because it is derived from another field 
    # that is already scaled if scaling was required, 
    # so we pass None for the scaling argument
    super().__init__(name, io_utils, mesh, None)
    self.field_x:StagField = cartesian_field
    return
  
  def get_data(self) -> np.ndarray|None:
    """
    Get the Cartesian field values from the mesh and return them as a numpy array.
    If the Cartesian field values are not already in the mesh, they are added to the mesh
    using the :py:meth:`add_to_mesh <stagpyviz.StagField.add_to_mesh>` method of the Cartesian field.
    If the Cartesian field values cannot be retrieved, the function returns None and the spherical field will be ignored.

    :return: 
      Array containing the Cartesian field values of the shape
      ``(mesh.number_of_points, components)`` for point fields or
      ``(mesh.number_of_cells, components)`` for cell fields, 
      where ``components`` is the number of components of the Cartesian field (e.g. 3 for velocity).
    :rtype: np.ndarray|None
    """
    if self.field_x.name in self.mesh.cell_data or self.field_x.name in self.mesh.point_data:
      return self.mesh[self.field_x.name]
    else:
      self.field_x.add_to_mesh()
      if self.field_x.name in self.mesh.cell_data or self.field_x.name in self.mesh.point_data:
        return self.mesh[self.field_x.name]
      else:
        print(f"Cannot compute spherical field because Cartesian field '{self.field_x.name}' values could not be retrieved.")
        return None

  def get_values(self) -> np.ndarray|None:
    """
    Transform the field from Cartesian to spherical coordinates using the 
    :py:meth:`vector_cartesian_to_spherical <stagpyviz.UnstructuredSphere.vector_cartesian_to_spherical>` 
    method of the mesh.

    :return:
      Array containing the spherical field values of the shape
      ``(mesh.number_of_points, components)`` for point fields or
      ``(mesh.number_of_cells, components)`` for cell fields, 
      where ``components`` is the number of components of the Cartesian field (e.g. 3 for velocity).
    :rtype: np.ndarray|None
    """
    if self.values is not None:
      return self.values
    cartesian_vec = self.get_data()
    if cartesian_vec is None:
      return None
    self.values = self.mesh.vector_cartesian_to_spherical(cartesian_vec)
    return self.values

class CartesianGradient(DerivedField):
  """
  Class to compute and add to the mesh the gradient of a Cartesian field.

  .. note::
    The gradient of a Cartesian field is derived from the original Cartesian field 
    that has already been added to the mesh. 
    Therefore, it is automatically scaled if the original Cartesian field is scaled.
  
  :param str name: Name of the field.
  :param stagpyviz.IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh.
  :param stagpyviz.StagField field: Cartesian field for which the gradient is computed.
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, field:StagField):
    # gradients should not be scaled because they are derived from the original field
    # that is already scaled if scaling was required, 
    # so we pass None for the scaling argument
    super().__init__(name, io_utils, mesh, None)
    self.mesh  = mesh
    self.field = field
    return
  
  def get_data(self) -> np.ndarray|None:
    """
    Get the values of the field to differentiate from the mesh.

    .. note::
      The gradient can only be computed for a point field.

    :return:
      Array containing the field values of the shape
      ``(mesh.number_of_points)``.
    :rtype: np.ndarray|None
    """
    if self.field.name not in self.mesh.point_data:
      phi = self.field.get_values()
      if phi is None:
        print(f"Cannot compute gradient for field '{self.field.name}' because its values could not be retrieved.")
        return None
    else:
      phi = self.mesh.point_data[self.field.name]
    return phi
  
  def get_values(self) -> np.ndarray|None:
    """
    Compute the gradient of the field using the 
    :py:meth:`compute_gradient <stagpyviz.YinYangMesh.compute_gradient>` method of the mesh.

    :return:
      Array containing the gradient values of the shape
      ``(mesh.number_of_cells, components)``
    :rtype: np.ndarray|None
    """
    if self.values is not None:
      return self.values
    field = self.get_data()
    if field is None:
      return None
    if not self.mesh.is_point_field(field):
      raise ValueError(f"Cannot compute gradient for field of shape {field.shape} that is not a point field ({self.mesh.number_of_points}).")
    self.values = self.mesh.compute_gradient(field)
    return self.values
  
class SphericalVectorGradient(DerivedField):
  """
  Class to compute and add to the mesh the gradient of a vector field in spherical coordinates.
  It proceeds in three steps:

  1.
    It transforms the vector field from Cartesian to spherical coordinates using the 
    :py:meth:`vector_cartesian_to_spherical <stagpyviz.UnstructuredSphere.vector_cartesian_to_spherical>` method of the mesh.
  
  2.
    It computes the gradient of the spherical vector field in Cartesian coordinates using the 
    :py:meth:`compute_gradient <stagpyviz.YinYangMesh.compute_gradient>` method of the mesh.
  
  3.
    It transforms the gradient of the vector field from Cartesian to spherical coordinates using the 
    :py:meth:`vector_cartesian_to_spherical <stagpyviz.UnstructuredSphere.vector_cartesian_to_spherical>` method of the mesh.
  
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, field:StagField):
    # gradients should not be scaled because they are derived from the original field
    # that is already scaled if scaling was required, 
    # so we pass None for the scaling argument
    super().__init__(name, io_utils, mesh, None)
    self.field = field
    return
  
  def get_data(self) -> np.ndarray|None:
    if self.field.name not in self.mesh.point_data:
      try:
        self.field.add_to_mesh()
      except Exception as e:
        print(f"Cannot compute gradient for field '{self.field.name}' because its values could not be retrieved.")
        print(f"Error: {e}")
        return None
      if self.field.name not in self.mesh.point_data:
        print(f"Cannot compute gradient for field '{self.field.name}' because its values could not be retrieved.")
        return None
    return self.mesh.point_data[self.field.name]
  
  def get_values(self) -> np.ndarray|None:
    if self.values is not None:
      return self.values
    field = self.get_data()
    if field is None:
      return None
    if not self.mesh.is_point_field(field):
      raise ValueError(f"Cannot compute gradient for field of shape {field.shape} that is not a point field ({self.mesh.number_of_points}).")
    # First we convert the vector field from Cartesian to spherical coordinates
    field_spherical = self.mesh.vector_cartesian_to_spherical(field)
    # Then we compute the gradient of the spherical vector field in cartesian coordinates
    grad_x_fs = self.mesh.compute_gradient(field_spherical)
    bs = grad_x_fs.shape[-1]
    # Finally we convert the gradient of the vector field from Cartesian to spherical coordinates
    self.values = np.zeros_like(grad_x_fs)
    for b in range(bs):
      self.values[...,b] = self.mesh.vector_cartesian_to_spherical(grad_x_fs[...,b])
    return self.values

def fields_instances(io_utils:IOutils, mesh:YinYangMesh, scalings:dict[str, Scaling]={}) -> dict[str, StagField]:
  """
  Function to create and return a dictionary of field instances 
  for the fields that can be added to the mesh.

  :param stagpyviz.IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh.
  :param dict[str, stagpyviz.Scaling] scalings: 
    Dictionary of scaling factors for the fields, where the keys are the field 
    names and the values are the corresponding :py:class:`Scaling <stagpyviz.Scaling>` instances. 
    Default: empty dictionary, which means that all fields will be added to the mesh 
    without scaling.
  :return: 
    Dictionary of field instances, where the keys are the field names and the 
    values are the corresponding field classes instances.
  :rtype: dict[str, StagField]

  Current implementation:

  .. code-block:: python

    field_classes = {}
    field_classes["composition"] = StagField("composition", io_utils, mesh)
    field_classes["divergence"]  = StagField("divergence", io_utils, mesh, scalings.get("strain_rate", None))
    field_classes["e2"]          = StagField("e2", io_utils, mesh, scalings.get("strain_rate", None))
    field_classes["nrc"]         = StagField("nrc", io_utils, mesh) # don't know what is this field
    field_classes["primordial"]  = StagField("primordial", io_utils, mesh)
    field_classes["proterozoic"] = StagField("proterozoic", io_utils, mesh)
    field_classes["stress"]      = StagField("stress", io_utils, mesh, scalings.get("pressure", None))
    field_classes["temperature"] = StagField("temperature", io_utils, mesh, scalings.get("temperature", None))
    field_classes["viscosity"]   = StagField("viscosity", io_utils, mesh, scalings.get("viscosity", None))
    field_classes["vorticity"]   = StagField("vorticity", io_utils, mesh, scalings.get("strain_rate", None)) # check units of this field

    field_classes["pressure"]    = Pressure("pressure", io_utils, mesh, scalings.get("pressure", None))
    field_classes["velocity"]    = Velocity("velocity", io_utils, mesh, scalings.get("velocity", None))
    field_classes["velocity_r"]  = SphericalField("velocity_r", io_utils, mesh, field_classes["velocity"])
    
    field_classes["grad_T"]      = CartesianGradient("grad_T", io_utils, mesh, field_classes["temperature"])
    field_classes["grad_T_r"]    = SphericalField("grad_T_r", io_utils, mesh, field_classes["grad_T"])
    field_classes["grad_P"]      = CartesianGradient("grad_P", io_utils, mesh, field_classes["pressure"])
    field_classes["grad_P_r"]    = SphericalField("grad_P_r", io_utils, mesh, field_classes["grad_P"])
    field_classes["grad_v"]      = CartesianGradient("grad_v", io_utils, mesh, field_classes["velocity"])
    field_classes["grad_v_r"]    = SphericalVectorGradient("grad_v_r", io_utils, mesh, field_classes["velocity"])

  """
  field_classes = {}
  field_classes["composition"] = StagField("composition", io_utils, mesh)
  field_classes["divergence"]  = StagField("divergence", io_utils, mesh, scalings.get("strain_rate", None))
  field_classes["e2"]          = StagField("e2", io_utils, mesh, scalings.get("strain_rate", None))
  field_classes["nrc"]         = StagField("nrc", io_utils, mesh) # don't know what is this field
  field_classes["primordial"]  = StagField("primordial", io_utils, mesh)
  field_classes["proterozoic"] = StagField("proterozoic", io_utils, mesh)
  field_classes["stress"]      = StagField("stress", io_utils, mesh, scalings.get("pressure", None))
  field_classes["temperature"] = StagField("temperature", io_utils, mesh, scalings.get("temperature", None))
  field_classes["viscosity"]   = StagField("viscosity", io_utils, mesh, scalings.get("viscosity", None))
  field_classes["vorticity"]   = StagField("vorticity", io_utils, mesh, scalings.get("strain_rate", None)) # check units of this field

  field_classes["pressure"]    = Pressure("pressure", io_utils, mesh, scalings.get("pressure", None))
  field_classes["velocity"]    = Velocity("velocity", io_utils, mesh, scalings.get("velocity", None))
  field_classes["velocity_r"]  = SphericalField("velocity_r", io_utils, mesh, field_classes["velocity"])
  
  field_classes["grad_T"]      = CartesianGradient("grad_T", io_utils, mesh, field_classes["temperature"])
  field_classes["grad_T_r"]    = SphericalField("grad_T_r", io_utils, mesh, field_classes["grad_T"])
  field_classes["grad_P"]      = CartesianGradient("grad_P", io_utils, mesh, field_classes["pressure"])
  field_classes["grad_P_r"]    = SphericalField("grad_P_r", io_utils, mesh, field_classes["grad_P"])
  field_classes["grad_v"]      = CartesianGradient("grad_v", io_utils, mesh, field_classes["velocity"])
  field_classes["grad_v_r"]    = SphericalVectorGradient("grad_v_r", io_utils, mesh, field_classes["velocity"])

  return field_classes
  
  
  
  
  
  
  

def test():
  

  return

if __name__ == "__main__":
  test()