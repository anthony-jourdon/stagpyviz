import os
import argparse
import yaml
import numpy as np
import pyvista as pvs
import stagpyviz as spv
from time import perf_counter


class IOutils:
  def __init__(
      self,
      model_name:str, 
      model_dir:str, 
      basedir:str,
      prefix:str,
      pvd:str,
      output_dir:str,
      volume_mesh_name:str,
      step_start:int=0,
      step_end:int=1,
      dstep:int=1,
      is_surface:bool=True
    ):
    self.model:str   = model_name
    self.mdir:str    = model_dir
    self.basedir:str = basedir
    self.prefix:str  = prefix
    # path to model directory
    self.model_dir = os.path.join(basedir,model_dir)

    self.mesh_name:str   = volume_mesh_name
    self.pvd:str         = pvd
    self.is_surface:bool = is_surface
    # output directory
    self.output_dir:str = output_dir
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
      print(f"Created output directory: {self.output_dir}")

    self.step:int = 0
    self.steps_idx:np.ndarray = np.arange(step_start, step_end+1, dstep)
    self.time:float|None = None
    self.timeseries:list = []

    self.filelist = {
      "composition": "c",
      "divergence": "div",
      "e2" : "ed",
      "viscosity": "eta",
      "nrc": "nrc", # ??
      "primordial": "prm",
      "prot": "prot", # ??
      "stress" : "str",
      "temperature": "t",
      "tracer": "tra",
      "vorticity": "vor",
      "velocity": "vp",
    }
    return

class Field:
  def __init__(self, name:str, io_utils:IOutils):
    self.name     = name
    self.io_utils = io_utils
    self.values   = None
    return
  
  def get_data(self) -> np.ndarray|None:
    io_utils = self.io_utils
    fname = f"{io_utils.model}_{io_utils.filelist[self.name]}{str(io_utils.step).zfill(5)}"
    # check file existence
    full_fname = os.path.join(io_utils.model_dir,fname)
    if not os.path.exists(full_fname): 
      print(f"\t\tFile {full_fname} not found, ignoring field {self.name}.")
      return None
    header, data = spv.read_stag_bin(full_fname)
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
  
  def reset(self) -> None:
    self.values = None
    return

class Velocity(Field):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh):
    super().__init__(name, io_utils)
    self.mesh = mesh
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
  
class Pressure(Field):
  def __init__(self, name:str, io_utils:IOutils):
    super().__init__(name, io_utils)
    return
  
  def get_values(self) -> np.ndarray|None:
    if self.values is not None:
      return self.values
    data = self.get_data()
    if data is None:
      return None
    self.values = data[3,:,:,:,:]
    return self.values

class SphericalField(Field):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh, name_cartesian:str):
    super().__init__(name, io_utils)
    self.mesh = mesh
    self.name_x = name_cartesian
    return
  
  def get_data(self) -> np.ndarray|None:
    if self.name_x not in self.mesh.point_data and self.name_x not in self.mesh.cell_data:
      print(f"Cartesian field '{self.name_x}' not found, cannot compute spherical field '{self.name}'.")
      return None
    cartesian_field = self.mesh[self.name_x]
    return cartesian_field

  def get_values(self) -> np.ndarray|None:
    if self.values is not None:
      return self.values
    cartesian_vec = self.get_data()
    if cartesian_vec is None:
      return None
    self.values = self.mesh.vector_cartesian_to_spherical(cartesian_vec)
    return self.values
  
class CartesianGradient(Field):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh, field:Field):
    super().__init__(name, io_utils)
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
    bs = field.shape[1] if field.ndim == 2 else 1
    if bs == 1:
      values = self.mesh.compute_gradient(field)
    else:
      values = np.zeros((self.mesh.number_of_cells, 3, bs), dtype=np.float64)
      for b in range(bs):
        values[:,:,b] = self.mesh.compute_gradient(field[:,b])
    self.values = values
    return self.values

class SphericalGradient(Field):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh, cartesian_gradient:CartesianGradient):
    super().__init__(name, io_utils)
    self.mesh = mesh
    self.grad_x = cartesian_gradient
    return
  
  def get_data(self):
    if self.grad_x.name in self.mesh.cell_data or self.grad_x.name in self.mesh.point_data:
      return self.mesh[self.grad_x.name]
    else:
      grad_x = self.grad_x.get_values()
      if grad_x is None:
        print(f"Cannot compute spherical gradient because Cartesian gradient values could not be retrieved.")
        return None
      return grad_x

  def get_values(self) -> np.ndarray|None:
    if self.values is not None:
      return self.values
    cartesian_gradient = self.get_data()
    if cartesian_gradient is None:
      return None
    bs = cartesian_gradient.shape[1] if cartesian_gradient.ndim == 2 else 1
    if bs == 1:
      values = self.mesh.spherical_gradient(cartesian_gradient)
    else:
      values = np.zeros((self.mesh.number_of_cells, 3, bs), dtype=np.float64)
      for b in range(bs):
        values[:,:,b] = self.mesh.spherical_gradient(cartesian_gradient[:,b])
    self.values = values
    return self.values

def write_step(mesh:spv.YinYangMesh, io_utils:IOutils, fields_names:list[str]) -> None:
  step = io_utils.step
  is_surface = io_utils.is_surface
  outfname = f"step{str(step).zfill(5)}{io_utils.prefix}.vtu"
  if is_surface:
    surface_mesh:pvs.StructuredGrid = mesh.surface_mesh
    #surface_mesh.clear_data()
    for field in fields_names:
      surface_mesh[field] = mesh[field][mesh.surface_idx]
    # save the surface mesh
    print(f"\tWriting surface mesh: {outfname}...")
    t0 = perf_counter()
    surface_mesh.save(os.path.join(io_utils.output_dir,outfname))
    t1 = perf_counter()
    print(f"\tSurface mesh written in {t1-t0:g} seconds.")
  else:
    print(f"\tWriting volume mesh: {outfname}")
    t0 = perf_counter()
    mesh.save(os.path.join(io_utils.output_dir,outfname))
    t1 = perf_counter()
    print(f"\tVolume mesh written in {t1-t0:g} seconds.")
  return

def mesh_add_fields(mesh:spv.YinYangMesh, io_utils:IOutils, fields_names:list[str]) -> None:
  fields = {}
  for field in fields_names:
    print(f"\tProcessing field: {field}")
    fname = f"{io_utils.model}_{io_utils.filelist[field]}{str(io_utils.step).zfill(5)}"
    # check file existence
    full_fname = os.path.join(io_utils.model_dir,fname)
    if not os.path.exists(full_fname): 
      print(f"\t\tFile {full_fname} not found, ignoring field {field}.")
      continue
    header, field_data = spv.read_stag_bin(full_fname)
    time = header["time"]
    if field == "velocity":
      fields["velocity"] = field_data[0:3,:,:,:,:]
      fields["pressure"] = field_data[3,:,:,:,:]
      mesh.reconstruct_velocity(fields["velocity"])
    else:
      fields[field] = field_data[0,:,:,:,:]
  # once all fields have been collected we can assign them to the surface mesh
  print("\tAdding fields to mesh...")
  mesh.add_fields(fields)
  io_utils.timeseries.append( (str(time), str(io_utils.step).zfill(5)) )
  return

def process_model_steps(io_utils:IOutils, fields_names:list[str], reset:bool=True) -> None:
  fname = f"{io_utils.model}_{io_utils.filelist['velocity']}{str(io_utils.step).zfill(5)}"
  print("Generating volume mesh...")
  mesh = spv.YinYangMesh(os.path.join(io_utils.model_dir,fname))
  
  for step in io_utils.steps_idx:
    io_utils.step = step
    print(f"Processing step: {step}")
    mesh_add_fields(mesh, io_utils, fields_names, reset)
    write_step(mesh, io_utils, fields_names)
  return

def generate_pvd_only(io_utils:IOutils) -> None:
  return

def get_info():
  return

def main():
  return

def test():
  
  return

if __name__ == "__main__":
  test()