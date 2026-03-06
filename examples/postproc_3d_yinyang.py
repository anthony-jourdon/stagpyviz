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
      pvd:str,
      output_dir:str,
      output_fields:list[str],
      **kwargs
    ):
    self.model:str               = model_name
    self.mdir:str                = model_dir
    self.basedir:str             = basedir
    self.output_fields:list[str] = output_fields
    self.pvd:str                 = pvd
    self.prefix:str = kwargs.get("prefix", "")
    # path to model directory
    self.model_dir = os.path.join(basedir,model_dir)
    self.time:float|None = None
    self.timeseries:dict[str,list] = {
      "time": [], # store time as string
      "step": [], # store step number as string
    }
    # output directory
    self.output_dir:str = output_dir
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
      print(f"Created output directory: {self.output_dir}")

    self.is_surface:bool = kwargs.get("is_surface", True)

    # stepping parameters for time series processing
    self.step:int|None  = kwargs.get("step", None)
    self.step_start:int = kwargs.get("step_start", 0)
    self.step_end:int   = kwargs.get("step_end", None)
    dstep:int           = kwargs.get("dstep", 1)
    self.steps_idx:np.ndarray|None = np.arange(self.step_start, self.step_end+1, dstep) if self.step_end is not None else None

    self.regions:list[str] = kwargs.get("regions", ["composition"])
    self.reset_fields:bool = kwargs.get("reset_fields", False)

    self.filelist = {
      "composition": "c",
      "divergence": "div",
      "e2" : "ed",
      "viscosity": "eta",
      "nrc": "nrc", # ??
      "primordial": "prm",
      "proterozoic": "prot", # ??
      "stress" : "str",
      "temperature": "t",
      "tracer": "tra",
      "vorticity": "vor",
      "velocity": "vp",
      "pressure": "vp", # pressure is stored in the 4th component of the velocity file
    }
    return
  
  def __str__(self) -> str:
    s  = f"Model: {self.model}\n"
    s += f"Model directory: {self.model_dir}\n"
    s += f"Output directory: {self.output_dir}\n"
    s += f"Output fields: {self.output_fields}\n"
    s += f"PVD file: {self.pvd}\n"
    s += f"Regions: {self.regions}\n"
    s += f"Reset: {self.reset_fields}\n"
    s += f"Prefix: {self.prefix}\n"
    s += f"Step: {self.step}\n"
    s += f"Step start: {self.step_start}\n"
    s += f"Step end: {self.step_end}\n"
    s += f"Delta step: {self.steps_idx[1] - self.steps_idx[0] if self.steps_idx is not None else 1}\n"
    s += f"Surface: {self.is_surface}\n"
    return s

class Field:
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh):
    self.name     = name
    self.io_utils = io_utils
    self.mesh     = mesh
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

  def add_to_mesh(self) -> None:
    if self.values is None:
      values = self.get_values()
    if values is None:
      print(f"Cannot add field '{self.name}' to mesh because its values could not be retrieved.")
      return
    self.mesh.add_field(self.name, values)
    return

  def reset(self) -> None:
    if self.name in self.mesh.point_data:
      self.mesh.point_data.pop(self.name)
    if self.name in self.mesh.cell_data:
      self.mesh.cell_data.pop(self.name)
    self.values = None
    return

class DerivedField(Field):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh):
    super().__init__(name, io_utils, mesh)
    return
  
  def add_to_mesh(self) -> None:
    if self.values is None:
      values = self.get_values()
    if values is None:
      print(f"Cannot add field '{self.name}' to mesh because its values could not be retrieved.")
      return
    self.mesh[self.name] = values
    return

class Velocity(Field):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh):
    super().__init__(name, io_utils, mesh)
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
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh):
    super().__init__(name, io_utils, mesh)
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
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh, cartesian_field:Field):
    super().__init__(name, io_utils, mesh)
    self.field_x = cartesian_field
    return
  
  def get_data(self) -> np.ndarray|None:
    if self.field_x.name in self.mesh.cell_data or self.field_x.name in self.mesh.point_data:
      return self.mesh[self.field_x.name]
    else:
      self.field_x.add_to_mesh()
      return self.mesh[self.field_x.name]

  def get_values(self) -> np.ndarray|None:
    if self.values is not None:
      return self.values
    cartesian_vec = self.get_data()
    if cartesian_vec is None:
      return None
    self.values = self.mesh.vector_cartesian_to_spherical(cartesian_vec)
    return self.values

class CartesianGradient(DerivedField):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh, field:Field):
    super().__init__(name, io_utils, mesh)
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

class SphericalGradient(DerivedField):
  def __init__(self, name:str, io_utils:IOutils, mesh:spv.YinYangMesh, cartesian_gradient:CartesianGradient):
    super().__init__(name, io_utils, mesh)
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
    if cartesian_gradient.ndim == 2: # vector field
      values = self.mesh.spherical_gradient(cartesian_gradient)
    elif cartesian_gradient.ndim == 3: # tensor field
      bs = cartesian_gradient.shape[2]
      values = np.zeros((self.mesh.number_of_cells, 3, bs), dtype=np.float64)
      for b in range(bs):
        values[:,:,b] = self.mesh.spherical_gradient(cartesian_gradient[:,b])
    self.values = values
    return self.values

def class_instances(mesh:spv.YinYangMesh, io_utils:IOutils) -> dict[str,Field]:
  """
  Dictionnary containing all the available field instances that can be added to the mesh.
  Add here any field that can be computed from the available data.
  
  :param mesh: Volume mesh reconstructed as an unstructured grid
  :type mesh: spv.YinYangMesh
  :param io_utils: Path and file management utilities
  :type io_utils: IOutils
  :return: Dictionnary of field instances that can be added to the mesh, with field names as keys
  :rtype: dict[str, Field]
  """
  avail_fields = {}
  avail_fields["velocity"]    = Velocity("velocity", io_utils, mesh)
  avail_fields["velocity_r"]  = SphericalField("velocity_r", io_utils, mesh, avail_fields["velocity"])
  avail_fields["temperature"] = Field("temperature", io_utils, mesh)
  avail_fields["grad_T"]      = CartesianGradient("grad_T", io_utils, mesh, avail_fields["temperature"])
  avail_fields["grad_T_r"]    = SphericalGradient("grad_T_r", io_utils, mesh, avail_fields["grad_T"])
  avail_fields["divergence"]  = Field("divergence", io_utils, mesh)
  avail_fields["e2"]          = Field("e2", io_utils, mesh)
  avail_fields["stress"]      = Field("stress", io_utils, mesh)
  avail_fields["pressure"]    = Pressure("pressure", io_utils, mesh)
  avail_fields["nrc"]         = Field("nrc", io_utils, mesh) # don't know what is this field
  avail_fields["composition"] = Field("composition", io_utils, mesh)
  avail_fields["primordial"]  = Field("primordial", io_utils, mesh)
  avail_fields["proterozoic"] = Field("proterozoic", io_utils, mesh)
  avail_fields["vorticity"]   = Field("vorticity", io_utils, mesh)
  avail_fields["viscosity"]   = Field("viscosity", io_utils, mesh)
  return avail_fields

def add_fields_to_mesh(mesh:spv.YinYangMesh, fields_to_add:list[str], class_fields:dict[str,Field], regions:list[str]=["composition"]) -> None:
  for field in fields_to_add:
    if field == "regions":
      continue # regions require special processing
    if field not in class_fields:
      print(f"Field \"{field}\" is not available and cannot be added to the mesh.")
      continue
    print(f"\tProcessing field: \"{field}\"")
    class_fields[field].add_to_mesh()
  
  # Rgions need a special treatment because depending on the version they may have been output as separated fields 
  if "regions" in fields_to_add:
    #mesh["regions"] = np.zeros(mesh.number_of_points, dtype=np.int32)
    mesh["regions"] = mesh.create_point_field(bs=1, dtype=np.int32)
    if len(regions) > 1: # only merge if there are more than 1 region
      # merge the region fields into a single field
      for region in regions:
        mesh["regions"] += (mesh[region] > 0.1) * (regions.index(region) + 1)
        mesh.point_data.pop(region)
    else:
      mesh["regions"] = mesh[regions[0]]
      
  # Now remove any extra field that was added to mesh but is not in the list of fields to output
  for field in mesh.point_data:
    if field not in fields_to_add:
      mesh.point_data.pop(field)
  for field in mesh.cell_data:
    if field not in fields_to_add:
      mesh.cell_data.pop(field)
  return

def write_mesh(mesh:spv.YinYangMesh|spv.ShellMesh, io_utils:IOutils, outfname:str) -> None:
  t0 = perf_counter()
  print(f"\tWriting mesh: {outfname}")
  mesh.save(os.path.join(io_utils.output_dir,outfname))
  t1 = perf_counter()
  print(f"\tMesh written in {t1-t0:g} seconds.")
  return

def write_step(mesh:spv.YinYangMesh, io_utils:IOutils, class_fields:dict[str,Field]) -> None:
  is_surface = io_utils.is_surface
  regions_list = io_utils.regions
  fields_to_add = io_utils.output_fields
  
  # Add fields to the mesh
  add_fields_to_mesh(mesh, fields_to_add, class_fields, regions_list)

  outfname = f"step{str(io_utils.step).zfill(5)}{io_utils.prefix}.vtu"
  if is_surface:
    if os.path.exists(os.path.join(io_utils.output_dir,outfname)) and not io_utils.reset_fields:
      print(f"\tFound existing output file {outfname}, append required fields.")
      surface_mesh:spv.ShellMesh = spv.ShellMesh(os.path.join(io_utils.output_dir,outfname))
    else:
      surface_mesh:spv.ShellMesh = mesh.surface_mesh
    for field in fields_to_add:
      if field in mesh.point_data:
        surface_mesh.point_data[field] = mesh.point_data[field][mesh.surface_idx]
      if field in mesh.cell_data:
        surface_mesh.cell_data[field] = mesh.cell_data[field][mesh.surface_cells]
    write_mesh(surface_mesh, io_utils, outfname)
  else:
    write_mesh(mesh, io_utils, outfname)
  return

def clean_up(class_instances:dict[str,Field]) -> None:
  for field in class_instances:
    class_instances[field].reset()
  return

def process_full_directory(mesh:spv.YinYangMesh, io_utils:IOutils, class_fields:dict[str,Field]) -> None:
  """
  Process the entire directory of output files.
  This function assumes that the steps are sequential and there are no missing steps in the directory.
  If this condition is not met, use EXAMPLE
  
  :param mesh: Description
  :type mesh: spv.YinYangMesh
  :param io_utils: Description
  :type io_utils: IOutils
  :param class_fields: Description
  :type class_fields: dict[str, Field]
  """
  found = True
  while found:
    fname = f"{io_utils.model}_{io_utils.filelist[io_utils.output_fields[0]]}{str(io_utils.step).zfill(5)}"
    if not os.path.exists(os.path.join(io_utils.model_dir,fname)):
      print(f"File {fname} not found, stopping time series processing.")
      found = False
    else:
      write_step(mesh, io_utils, class_fields)
      clean_up(class_fields)
      io_utils.timeseries["time"].append(str(io_utils.time))
      io_utils.timeseries["step"].append(str(io_utils.step).zfill(5))
      io_utils.step += 1
  return

def process_step_range(mesh:spv.YinYangMesh, io_utils:IOutils, class_fields:dict[str,Field]) -> None:
  """
  Process a specified range of steps, defined by io_utils.step_start and io_utils.step_end.
  This function is more robust to missing steps in the directory, but requires the user to specify the step range and dstep.
  
  :param mesh: Description
  :type mesh: spv.YinYangMesh
  :param io_utils: Description
  :type io_utils: IOutils
  :param class_fields: Description
  :type class_fields: dict[str, Field]
  """
  for step in io_utils.steps_idx:
    fname = f"{io_utils.model}_{io_utils.filelist[io_utils.output_fields[0]]}{str(step).zfill(5)}"
    if not os.path.exists(os.path.join(io_utils.model_dir,fname)):
      print(f"File {fname} not found, skipping step {step}.")
      continue
    else:
      io_utils.step = step
      write_step(mesh, io_utils, class_fields)
      clean_up(class_fields)
      io_utils.timeseries["time"].append(str(io_utils.time))
      io_utils.timeseries["step"].append(str(step).zfill(5))
  return

def process_model(iou:IOutils) -> None:
  if iou.step is None:
    iou.step = iou.step_start
    if iou.step_end is None:
      use_case = "full_directory"
    else:
      use_case = "step_range"
  else:
    use_case = "single_step"

  # Generate the mesh
  fname = f"{iou.model}_{iou.filelist[iou.output_fields[0]]}{str(iou.step).zfill(5)}"
  print(f"Generating volume mesh using file {fname}...")
  mesh = spv.YinYangMesh(os.path.join(iou.model_dir,fname))
  # create the available class instances for the fields that can be added to the mesh
  class_fields = class_instances(mesh, iou)

  match use_case:
    case "single_step":
      print(f"Processing single step: {iou.step}")
      write_step(mesh, iou, class_fields)
      clean_up(class_fields)
      iou.timeseries["time"].append(str(iou.time))
      iou.timeseries["step"].append(str(iou.step).zfill(5))
    case "full_directory":
      print(f"Processing full directory starting from step {iou.step} until no more files are found.")
      process_full_directory(mesh, iou, class_fields)
    case "step_range":
      print(f"Processing steps from {iou.step_start} to {iou.step_end} with dstep {iou.steps_idx[1]-iou.steps_idx[0]}.")
      process_step_range(mesh, iou, class_fields)

  print(f"Writing pvd file {iou.pvd}")
  spv.timeseries_write(os.path.join(iou.output_dir,iou.pvd), iou.timeseries, prefix=iou.prefix, extension="vtu", erase=iou.reset_fields)
  return

def main():
  description = "Postprocessing parser for StagYY 3D YinYang models"

  parser = argparse.ArgumentParser(
    prog="postproc_3d_yinyang.py", 
    description=description, 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    '-f',
    '--file',
    type=str,
    help='Path to the yaml input file',
    dest='yaml_file',
    required=True
  )
  args = parser.parse_args()
  if not os.path.exists(args.yaml_file):
    help_str = f"YAML file {args.yaml_file} not found.\n"
    raise FileNotFoundError(help_str)

  with open(args.yaml_file) as f:
    user:dict = yaml.load(f, Loader=yaml.FullLoader)
  #####################
  # Steps information #
  #####################
  if "steps" not in user:
    user["steps"] = {
      "step": None,
      "step_start": 0,
      "step_end": None,
      "dstep": 1,
    }
    help_str = f"steps options not provided in yaml file, defaulting to:\n{user['steps']}\n"
    help_str += f"Consider adding a steps section to your yaml file to specify the steps to process\n"
    print(help_str)

  #####################
  # Paths information #
  #####################
  if "paths" not in user:
    help_str  = f"paths section not found in yaml file.\n"
    help_str += f"Use the paths section to specify the paths to the model directory and output directory using the following keys:\n"
    help_str += f"  model: path to the model directory\n"
    help_str += f"  directory: path to the output directory\n"
    help_str += f"  base:\n"
    help_str += f"    model: absolute path to the base model directory\n"
    help_str += f"    output: absolute path to the base output directory\n"
    raise ValueError(help_str)
  
  paths:dict = user["paths"]
  if "model" not in paths:
    help_str = f"model key not found in paths section of yaml file.\n"
    help_str += f"Use the model key to specify the name of the model.\n"
    raise ValueError(help_str)
  if "directory" not in paths:
    help_str = f"directory key not found in paths section of yaml file.\n"
    help_str += f"Use the directory key to specify the name of the model directory where the output files are located.\n"
    raise ValueError(help_str)
  if "base" not in paths:
    help_str = f"base key not found in paths section of yaml file.\n"
    help_str += f"Use the base key to specify the absolute paths to the base model and output directories using the following keys:\n"
    help_str += f"  model: absolute path to the base model directory\n"
    help_str += f"  output: absolute path to the base output directory\n"
    raise ValueError(help_str)
  if "model" not in paths["base"]:
    help_str = f"model key not found in base section of paths section of yaml file.\n"
    help_str += f"Use the model key to specify the absolute path to the base model directory.\n"
    raise ValueError(help_str)
  if "output" not in paths["base"]:
    help_str = f"output key not found in base section of paths section of yaml file.\n"
    help_str += f"Use the output key to specify the absolute path to the base output directory.\n"
    raise ValueError(help_str)

  ######################
  # Fields information #
  ######################
  if "fields" not in user:
    help_str  = f"fields section not found in yaml file.\n"
    help_str += f"Use the fields section to specify the fields to be added to the output mesh using the following key:\n"
    help_str += f"  process: list of fields to be added to the output mesh.\n"
    help_str += f"  regions: list of region fields to be merged into a single regions field in the output mesh.\n"
    help_str += f"  surface: boolean indicating whether to output surface fields (True) or volume fields (False), default: False.\n"
    raise ValueError(help_str)
  
  fields:dict = user["fields"]
  if "process" not in fields:
    help_str = f"process key not found in fields section of yaml file.\n"
    help_str += f"Use the process key to specify the list of fields to be added to the output mesh.\n"
    raise ValueError(help_str)
  
  prefix = paths.get("prefix", "")
  is_surface = fields.get("surface", False)
  if is_surface:
    prefix += "-surface"
  default_pvd = f"timeseries_{paths['model']}{prefix}.pvd"
  pvd = paths.get("pvd", default_pvd)

  output_fields:list[str] = fields["process"]
  regions_list  = fields.get("regions", ["composition"])
  for region in regions_list:
    if region in fields["process"]:
      output_fields.append("regions")
      break

  step       = user["steps"].get("step", None)
  reset      = user["steps"].get("reset", False)
  output_dir = os.path.join(paths["base"]["output"],paths["directory"])
  start_step = user["steps"].get("start", 0)
  if step is None:
    # check if a pvd file already exists
    if os.path.exists(os.path.join(output_dir,pvd)) and not reset:
      print(f"Found exisiting pvd file {pvd}, appending new time steps.")
      timeseries = spv.timeseries_process(os.path.join(output_dir,pvd))
      start_step = int(timeseries["step"][-1]) + 1
      print(f"Updated start step to {start_step} based on existing pvd file.")

  io_utils = IOutils(
    model_name=paths["model"],
    model_dir=paths["directory"],
    basedir=paths["base"]["model"],
    pvd=pvd,
    output_dir=output_dir,
    output_fields=output_fields,
    regions=regions_list,
    step=step,
    step_start=start_step,
    step_end=user["steps"].get("end", None),
    dstep=user["steps"].get("delta", 1),
    is_surface=is_surface,
    reset_fields=reset,
    prefix=prefix,
  )
  process_model(io_utils)
  exit(0)

def test():
  model = "llsvp"
  mdir  = "LLSVP-Run4"
  pvd_fname = f"timeseries_{model}-surface.pvd"
  basedir = os.path.join("/data","ens","ncoltice","CLAUDIO")
  output_dir = os.path.join("/data","jourdon","llsvp-pandora-vtu","surface-yy",mdir)
  output_fields = ["velocity", "velocity_r", "e2", "temperature", "divergence", "composition", "primordial", "proterozoic"]
  reset = False
  step  = None
  start_step = 0
  end_step = None
  # will be given by user in real use case
  regions = ["primordial", "composition", "proterozoic"]
  for region in regions:
    if region in output_fields:
      output_fields.append("regions")
      break
  
  if step is None:
    # check if a pvd file already exists
    if os.path.exists(os.path.join(output_dir,pvd_fname)) and not reset:
      print(f"Found exisiting pvd file {pvd_fname}, appending new time steps.")
      timeseries = spv.timeseries_process(os.path.join(output_dir,pvd_fname))
      start_step = int(timeseries["step"][-1]) + 1
      print(f"Updated start step to {start_step} based on existing pvd file.")

  io_utils = IOutils(
    model_name=model,
    model_dir=mdir,
    basedir=basedir,
    pvd=pvd_fname,
    output_dir=output_dir,
    output_fields=output_fields,
    regions=regions,
    step=step,
    step_start=start_step,
    step_end=end_step,
    is_surface=True,
    reset_fields=reset,
  )
  process_model(io_utils)
  return

def test2():
  model = "llsvp"
  mdir  = "LLSVP-Run4"
  pvd_fname = f"timeseries_{model}_surface.pvd"
  basedir = os.path.join("/data","ens","ncoltice","CLAUDIO")
  output_dir = os.path.join("/data","jourdon","llsvp-pandora-vtu","surface-yy",mdir)
  output_fields = ["velocity", "velocity_r", "e2", "temperature", "divergence", "composition", "primordial", "proterozoic"]
  reset = False
  start_step = 0
  end_step = 0
  # will be given by user in real use case
  regions = ["primordial", "composition", "proterozoic"]
  for region in regions:
    if region in output_fields:
      output_fields.append("regions")
      break

  # check if a pvd file already exists
  #if os.path.exists(os.path.join(output_dir,pvd_fname)) and not reset:
  #  print(f"Found exisiting pvd file {pvd_fname}, appending new time steps.")
  #  timeseries = spv.timeseries_process(os.path.join(output_dir,pvd_fname))
  #  print(timeseries)
  #  start_step = int(timeseries["step"][-1]) + 1
  #  print(f"Updated start step to {start_step} based on existing pvd file.")

  io_utils = IOutils(
    model_name=model,
    model_dir=mdir,
    basedir=basedir,
    pvd=pvd_fname,
    output_dir=output_dir,
    output_fields=output_fields,
    regions=regions,
    step_start=start_step,
    step_end=end_step,
    is_surface=True,
    reset_fields=reset,
  )
  
  io_utils.step = io_utils.steps_idx[0]
  # Generate the mesh
  fname = f"{model}_{io_utils.filelist[io_utils.output_fields[0]]}{str(io_utils.step).zfill(5)}"
  print("Generating volume mesh...")
  mesh = spv.YinYangMesh(os.path.join(io_utils.model_dir,fname))
  # create the available class instances for the fields that can be added to the mesh
  class_fields = class_instances(mesh, io_utils)
  
  #found = True
  #io_utils.step = start_step
  #while found:
  #  fname = f"{model}_{io_utils.filelist[io_utils.output_fields[0]]}{str(io_utils.step).zfill(5)}"
  #  if not os.path.exists(os.path.join(io_utils.model_dir,fname)):
  #    print(f"File {fname} not found, stopping time series processing.")
  #    found = False
  #  else:
  #    write_step(mesh, io_utils, class_fields)
  #    clean_up(class_fields)
  #    io_utils.timeseries["time"].append(str(io_utils.time))
  #    io_utils.timeseries["step"].append(str(io_utils.step).zfill(5))
  #    io_utils.step += 1

  for step in io_utils.steps_idx:
    io_utils.step = step
    print(f"Processing step {step}...")
    write_step(mesh, io_utils, class_fields)
    clean_up(class_fields)
    io_utils.timeseries["time"].append(str(io_utils.time))
    io_utils.timeseries["step"].append(str(step).zfill(5))
  
  # write the pvd file
  print(f"Writing pvd file {pvd_fname}")
  spv.timeseries_write(os.path.join(io_utils.output_dir,io_utils.pvd), io_utils.timeseries, prefix=io_utils.prefix, extension="vtu")
  return

if __name__ == "__main__":
  main()