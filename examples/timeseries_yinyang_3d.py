import os
import numpy as np
import pyvista as pvs
import stagpyviz as spv

class IOutils:
  def __init__(
      self,
      model_name:str, 
      model_dir:str, 
      basedir:str,
      pvd:str,
      output_dir:str,
      volume_mesh_name:str
    ):
    self.model = model_name
    self.mdir  = model_dir
    self.basedir = basedir
    # path to model directory
    self.model_dir = os.path.join(basedir,model_dir)

    self.mesh_name = volume_mesh_name
    self.pvd = pvd
    # output directory
    self.output_dir = output_dir
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
      print(f"Created output directory: {self.output_dir}")

    self.step = 0
    self.timeseries = []

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

def get_volume_mesh(io_utils:IOutils) -> spv.YinYangMesh:
  step      = io_utils.step
  model     = io_utils.model
  model_dir = io_utils.model_dir
  mesh_name = io_utils.mesh_name

  fname = f"{model}_{io_utils.filelist['velocity']}{str(step).zfill(5)}"
  mesh_name = os.path.join("/data","jourdon","llsvp-pandora-vtu",mesh_name)
  #if os.path.exists(mesh_name):
  #  print("Loading existing volume mesh...")
  #  mesh = spv.YinYangMesh(os.path.join(model_dir,fname), mesh_name)
  #else:
  print("Generating volume mesh...")
  mesh = spv.YinYangMesh(os.path.join(model_dir,fname))
  if not os.path.exists(mesh_name):
    print("Writing volume mesh...")
    mesh.save(mesh_name)
  return mesh

def process_timestep(mesh:spv.YinYangMesh, io_utils:IOutils, fields_names:list[str], reset:bool=False):
  step = io_utils.step
  print(f"Processing step: {step}")
  outfname = f"step{str(step).zfill(5)}_surface.vtu"
  if os.path.exists(os.path.join(io_utils.output_dir,outfname)) and not reset:
    print(f"\tFound existing output file {outfname}, append only missing fields.")
    #surface_mesh:pvs.StructuredGrid = pvs.read(os.path.join(io_utils.output_dir,outfname))
    surface_mesh:spv.ShellMesh = spv.ShellMesh(os.path.join(io_utils.output_dir,outfname))
  else:
    surface_mesh:spv.ShellMesh = mesh.surface_mesh
    surface_mesh.clear_data()
  
  fields = {}
  for field in fields_names:
    print(f"\tProcessing field: {field}")
    if field in surface_mesh.point_data or field in surface_mesh.cell_data: 
      print(f"\t\tField {field} already present in surface mesh, ignoring.")
      continue
    fname = f"{io_utils.model}_{io_utils.filelist[field]}{str(step).zfill(5)}"
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
  for field in fields:
    surface_mesh[field] = mesh[field][mesh.surface_idx]
  # save the surface mesh
  print(f"\tWriting surface mesh: {outfname}")
  surface_mesh.save(os.path.join(io_utils.output_dir,outfname))
  io_utils.timeseries.append( (str(time), str(step).zfill(5)) )
  return

def process_timestep_for_pvd(mesh:spv.YinYangMesh, io_utils:IOutils, fields_names:list[str]):
  step = io_utils.step
  print(f"Processing step: {step}")
  for field in fields_names:
    fname = f"{io_utils.model}_{io_utils.filelist[field]}{str(step).zfill(5)}"
    # check file existence
    full_fname = os.path.join(io_utils.model_dir,fname)
    if not os.path.exists(full_fname): 
      print(f"\t\tFile {full_fname} not found, ignoring field {field}.")
      continue
    header, _ = spv.read_stag_bin(full_fname)
    time = header["time"]
  io_utils.timeseries.append( (str(time), str(step).zfill(5)) )
  return

def main_surface():
  model = "llsvp"
  mdir  = "LLSVP-Run6"
  mesh_name = "volume_mesh.vtu"
  pvd_fname = f"timeseries_{model}_surface.pvd"
  basedir = os.path.join("/data","ens","ncoltice","CLAUDIO")
  output_dir = os.path.join("/data","jourdon","llsvp-pandora-vtu","surface-yy",mdir)
  #model = "PJB7_YS5_Rh55_Ts06"
  #mdir  = "PJB6_YS1.5"
  #mesh_name = "PJB7_YS5_Rh55_Ts06.vtu"
  #pvd_fname = f"timeseries_{model}_surface.pvd"
  #basedir = os.path.join("/data","ens","cmallard")
  #output_dir = os.path.join("/data","jourdon","LagRBF_input",mdir)
  
  io_utils = IOutils(
    model_name=model,
    model_dir=mdir,
    basedir=basedir,
    pvd=pvd_fname,
    output_dir=output_dir,
    volume_mesh_name=mesh_name
  )
  
  frame = np.arange(0,222)
  fields_names = [
    "temperature", 
    "composition", 
    "viscosity", 
    "stress", 
    "e2", 
    "velocity", 
    "divergence", 
    "vorticity"
  ]
  # the very first thing we do is to create the mesh, if an already existing volume mesh is availabe
  # we can load it directly
  io_utils.step = frame[0]
  mesh:spv.YinYangMesh = get_volume_mesh(io_utils)

  for step in frame:
    io_utils.step = step
    process_timestep(mesh, io_utils, fields_names, reset=False)
    #process_timestep_for_pvd(mesh, io_utils, ["temperature"])
  
  pvd_file = os.path.join(io_utils.output_dir,pvd_fname)
  #spv.write_timeseries_pvd(pvd_file, io_utils.timeseries, "surface", "vtu")
  if os.path.exists(pvd_file):
    spv.append_timeseries_pvd(pvd_file, io_utils.timeseries, "surface", "vtu")
  else:
    spv.write_timeseries_pvd(pvd_file, io_utils.timeseries, "surface", "vtu")
  return

if __name__ == "__main__":
  main_surface()
