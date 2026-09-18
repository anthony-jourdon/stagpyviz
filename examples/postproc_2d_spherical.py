import os
import argparse
import yaml
import numpy as np
import pyvista as pvs
import stagpyviz as spv
from time import perf_counter

def create_mesh(fname:str) -> spv.SphericalMesh:
  header, _ = spv.read_stag_bin(fname)
  mesh = spv.SphericalMesh(
    dimensions=(
      header['ntot'][1] + 1,
      header['ntot'][2] + 1,
      1
    ), 
    r=header["rpoints"] + header["rcmb"],
    phi=header["y"]
  )
  return mesh

def add_fields_to_mesh(mesh:spv.SphericalMesh, iou:spv.IOutils):
  field_list   = iou.output_fields
  avail_fields = iou.volume_fields
  for field in field_list:
    if field not in avail_fields:
      print(f"Field {field} not available. Skipping.")
      continue
    fname = iou.get_field_filename(field, iou.step)
    if fname is None:
      print(f"Field {field} not found for step {iou.step}. Skipping.")
      continue
    header, field_values = spv.read_stag_bin(fname)
    iou.time = header["time"]
    mesh.add_field(field, field_values)  
  return

def write_step(mesh:spv.SphericalMesh, iou:spv.IOutils):
  add_fields_to_mesh(mesh, iou)
  output_fname = os.path.join(iou.output_dir, f"step{iou.step:05d}{iou.prefix}.vts")
  print(f"Writing mesh to {output_fname}...", flush=True)
  mesh.save(output_fname)
  timeseries = {
    "time": [str(iou.time)],
    "step": [str(iou.step).zfill(5)]
  }
  spv.timeseries_write(os.path.join(iou.output_dir,iou.pvd), timeseries, prefix=iou.prefix, extension="vts")
  return

def process_step_range(mesh:spv.SphericalMesh, iou:spv.IOutils):
  for step in iou.steps_idx:
    print(f"Processing step {step}...", flush=True)
    iou.step = step
    write_step(mesh, iou)
  return

def process_full_directory(mesh:spv.SphericalMesh, iou:spv.IOutils):
  found = True
  while found:
    fname = iou.get_field_filename(iou.output_fields[0], iou.step)
    if not os.path.exists(os.path.join(iou.model_dir, fname)):
      found = False
      break
    print(f"Processing step {iou.step}...", flush=True)
    write_step(mesh, iou)
    iou.step += 1
  return

def process_model(iou:spv.IOutils):
  if iou.step is None:
    iou.step = iou.step_start
    if iou.step_end is None:
      use_case = "full_directory"
    else:
      use_case = "step_range"
  else:
    use_case = "single_step"

  # Generate the mesh
  for field in iou.output_fields:
    if field in iou.filelist:
      fname = iou.get_field_filename(field, iou.step)
      if fname is not None:
        break
  if fname is None:
    raise FileNotFoundError(f"None of the files for the requested fields {iou.output_fields} were found for step {iou.step}. Cannot generate mesh.")
  print(f"Generating mesh using file {fname}...", flush=True)
  mesh = create_mesh(fname)

  match use_case:
    case "single_step":
      print(f"Processing single step {iou.step}...", flush=True)
      write_step(mesh, iou)
    case "step_range":
      print(f"Processing steps from {iou.step_start} to {iou.step_end}...", flush=True)
      process_step_range(mesh, iou)
    case "full_directory":
      print(f"Processing all steps in directory {iou.model_dir}...", flush=True)
      process_full_directory(mesh, iou)

  return

def test():
  paths = {
    "model": "Ra7_pl_crust_HR",
    "directory": "Ra7_pl_crust_HR",
    "base": {
      "model": "/data/jourdon/models_2d",
      "output": "/data/jourdon/models_2d/test_postproc"
    }
  }
  output_dir  = os.path.join(paths["base"]["output"], paths["directory"])
  prefix      = paths.get("prefix", "")
  default_pvd = f"timeseries_{paths['model']}{prefix}.pvd"
  pvd = paths.get("pvd", default_pvd)

  fields = {
    "process": [
      "viscosity",
      "temperature",
      "velocity",
      "velocity_r",
      "pressure",
      "stress",
      "e2",
      "density",
      "heatflux",
      "composition",
    ]
  }


  output_fields:list[str] = fields["process"]
  regions_list  = fields.get("regions", None)
  if regions_list is not None:
    for region in regions_list:
      if region in fields["process"]:
        output_fields.append("regions")
        break

  step = 170
  end  = 172
  iou = spv.IOutils(
    model_name=paths["model"],
    model_dir=paths["directory"],
    basedir=paths["base"]["model"],
    pvd=pvd,
    output_dir=output_dir,
    output_fields=output_fields,
    regions=regions_list,
    step_start=step,
    step_end=end,
    prefix=prefix,
  )

  process_model(iou)

  return

def main():
  description = "Postprocessing parser for StagYY 2D Spherical models"
  
  parser = argparse.ArgumentParser(
    prog="postproc_2d_spherical.py", 
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
  regions_list  = fields.get("regions", None)
  if regions_list is not None:
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
  if reset:
    os.remove(os.path.join(output_dir,pvd))

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
  process_model(io_utils)
  exit(0)
  return

if __name__ == "__main__":
  main()