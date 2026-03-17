import os
import argparse
import yaml
import numpy as np
import pyvista as pvs
import stagpyviz as spv
from time import perf_counter

def add_fields_to_mesh(mesh:spv.YinYangMesh, fields_to_add:list[str], class_fields:dict[str,spv.StagField], regions:list[str]=["composition"]) -> None:
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

def write_mesh(mesh:spv.YinYangMesh|spv.ShellMesh, io_utils:spv.IOutils, outfname:str) -> None:
  t0 = perf_counter()
  print(f"\tWriting mesh: {outfname}")
  mesh.save(os.path.join(io_utils.output_dir,outfname))
  t1 = perf_counter()
  print(f"\tMesh written in {t1-t0:g} seconds.")
  return

def write_step(mesh:spv.YinYangMesh, io_utils:spv.IOutils, class_fields:dict[str,spv.StagField]) -> None:
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

def clean_up(class_instances:dict[str,spv.StagField]) -> None:
  for field in class_instances:
    class_instances[field].reset()
  return

def process_full_directory(mesh:spv.YinYangMesh, io_utils:spv.IOutils, class_fields:dict[str,spv.StagField]) -> None:
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

def process_step_range(mesh:spv.YinYangMesh, io_utils:spv.IOutils, class_fields:dict[str,spv.StagField]) -> None:
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

def process_model(iou:spv.IOutils, scaling_factors:dict[str, spv.Scaling]) -> None:
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
  mesh = spv.YinYangMesh(os.path.join(iou.model_dir,fname),scaling=scaling_factors.get("length", None))
  # create the available class instances for the fields that can be added to the mesh
  #class_fields = class_instances(mesh, iou)
  class_fields = spv.fields_instances(iou, mesh, scaling_factors)

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

  scaling_factors = {}
  if "scaling" in user:
    scaling:dict = user["scaling"]
    if scaling.get("scale", False):
      Ra = scaling.get("Ra", 1e7)
      scaling.pop("Ra")
      scaling.pop("scale")
      scaling_kwargs = {"Ra": Ra}
      for key in scaling:
        for subkey in scaling[key]:
          scaling_kwargs[key+"_"+subkey] = scaling[key][subkey]
      scaling_factors = spv.scaling_factors(**scaling_kwargs)

  io_utils = spv.IOutils(
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
  process_model(io_utils,scaling_factors)
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

  io_utils = spv.IOutils(
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

  io_utils = spv.IOutils(
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
  class_fields = spv.fields_instances(io_utils, mesh)
  
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