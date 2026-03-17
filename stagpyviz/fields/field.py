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
  By default, this class assumes that the field is stored in a binray file output by StagYY.
  Different types of fields can be created by inheriting from this class and implementing their own 
  methods to retrieve the data and add them to the mesh.

  :param str name: Name of the field, should correspond the the name registered in the IOutils class filelist attribute.
  :param IOutils io_utils: Path and file management utilities
  :param stagpyviz.YinYangMesh mesh: Volume mesh reconstructed as an unstructured grid
  """
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, scaling)
    self.io_utils:IOutils       = io_utils
    self.mesh:YinYangMesh       = mesh
    self.values:np.ndarray|None = None
    return
  
  def get_data(self) -> np.ndarray|None:
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
    if self.values is not None:
      return self.values
    data = self.get_data()
    if data is None:
      return None
    self.values = data[0,:,:,:,:]
    return self.values

  def add_to_mesh(self) -> None:
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
    if self.name in self.mesh.point_data:
      self.mesh.point_data.pop(self.name)
    if self.name in self.mesh.cell_data:
      self.mesh.cell_data.pop(self.name)
    self.values = None
    return

class DerivedField(StagField):
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, io_utils, mesh, scaling)
    return
  
  def add_to_mesh(self) -> None:
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
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, io_utils, mesh, scaling)
    return
  
  def get_values(self) -> np.ndarray|None:
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
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, scaling:Scaling|None=None):
    super().__init__(name, io_utils, mesh, scaling)
    return
  
  def get_values(self) -> np.ndarray|None:
    if self.values is not None:
      return self.values
    data = self.get_data()
    if data is None:
      return None
    self.values = data[3,:,:,:,:]
    return self.values

class SphericalField(DerivedField):
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, cartesian_field:StagField):
    # this field should not be scaled because it is derived from another field 
    # that is already scaled if scaling was required, 
    # so we pass None for the scaling argument
    super().__init__(name, io_utils, mesh, None)
    self.field_x:StagField = cartesian_field
    return
  
  def get_data(self) -> np.ndarray|None:
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
    if self.values is not None:
      return self.values
    cartesian_vec = self.get_data()
    if cartesian_vec is None:
      return None
    self.values = self.mesh.vector_cartesian_to_spherical(cartesian_vec)
    return self.values

class CartesianGradient(DerivedField):
  def __init__(self, name:str, io_utils:IOutils, mesh:YinYangMesh, field:StagField):
    # gradients should not be scaled because they are derived from the original field
    # that is already scaled if scaling was required, 
    # so we pass None for the scaling argument
    super().__init__(name, io_utils, mesh, None)
    self.mesh  = mesh
    self.field = field
    return
  
  def get_data(self) -> np.ndarray|None:
    if self.field.name not in self.mesh.point_data:
      phi = self.field.get_values()
      if phi is None:
        print(f"Cannot compute gradient for field '{self.field.name}' because its values could not be retrieved.")
        return None
    else:
      phi = self.mesh.point_data[self.field.name]
    return phi
  
  def get_values(self) -> np.ndarray|None:
    if self.values is not None:
      return self.values
    field = self.get_data()
    if field is None:
      return None
    if not self.mesh.is_point_field(field):
      raise ValueError(f"Cannot compute gradient for field of shape {field.shape} that is not a point field ({self.mesh.number_of_points}).")
    self.values = self.mesh.compute_gradient(field)
    return self.values

def fields_instances(io_utils:IOutils, mesh:YinYangMesh, scalings:dict[str, Scaling]={}) -> dict[str, StagField]:
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

  return field_classes
  
  
  
  
  
  
  

def test():
  

  return

if __name__ == "__main__":
  test()