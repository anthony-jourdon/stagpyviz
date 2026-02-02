from dataclasses import fields
import os
import numpy as np
import pyvista as pvs
import stagpyviz as spv

"""
Ra7_pl_crust_HR_T00000 
Ra7_pl_crust_HR_c00000
Ra7_pl_crust_HR_cs00000 -> topo
Ra7_pl_crust_HR_ed00000 -> e2
Ra7_pl_crust_HR_eta00000
Ra7_pl_crust_HR_hf00000 -> heat flux
Ra7_pl_crust_HR_hfs00000 -> ?
Ra7_pl_crust_HR_rho00000
Ra7_pl_crust_HR_str00000 -> stress
Ra7_pl_crust_HR_tra00000 -> tracer
Ra7_pl_crust_HR_vp00000
"""

def fields_name() -> dict:
  fnames = {
    "temperature": "T",
    "composition": "c",
    "viscosity": "eta",
    "density": "rho",
    "velocity": "vp",
    "stress": "str",
    "e2": "ed"
  }
  return fnames

def get_filename(mname:str, field:str, step:int) -> str:
  name = f"{mname}_{field}" + str(step).zfill(5)
  return name

def create_mesh(header:dict) -> spv.SphericalMesh:
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

def get_velocity_pressure(mesh:spv.SphericalMesh, fields:np.ndarray):
  v_spherical = np.zeros((mesh.number_of_cells,3), dtype=np.float64)
  v_spherical[:,0] = np.reshape(fields[2,0,:-1,:,0], (mesh.number_of_cells), order="F")
  v_spherical[:,1] = np.reshape(fields[1,0,:-1,:,0], (mesh.number_of_cells), order="F")
  pressure         = np.reshape(fields[3,0,:-1,:,0], (mesh.number_of_cells), order="F")

  mesh["velocity_r"] = v_spherical
  mesh["pressure"]   = pressure
  v_cartesian = mesh.vector_spherical_to_cartesian(v_spherical)
  mesh["velocity_x"] = v_cartesian
  return

def write_timeseries_pvd(fname:str,time_series:list,prefix:str,extension:str="vts"):
  with open(fname,'w') as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for time_entry in time_series:
      f.write('    <DataSet timestep="'+time_entry[0]+'" file="step'+time_entry[1]+'_'+prefix+'.'+extension+'"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')
  return

def append_timeseries_pvd(fname:str,time_series:list,prefix:str,extension:str="vts"):
  with open(fname,'r+') as f:
    content = f.readlines()
    f.seek(0)
    # write up to Collection
    for line in content:
      if '  </Collection>' in line:
        break
      f.write(line)
      
    # append new entries
    for time_entry in time_series:
      f.write('    <DataSet timestep="'+time_entry[0]+'" file="step'+time_entry[1]+'_'+prefix+'.'+extension+'"/>\n')
    # write closing tags
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')

def test():
  model_name = "Ra7_50m15wb_ccm_noH"
  #model_name = "Ra7_pl_cont_LR_suite"
  mdir = os.path.join(os.environ["SCRATCH"], model_name)
  print(f"Model directory: {mdir}")
  odir = os.path.join(os.environ["SCRATCH"], model_name, "vts_output")
  if not os.path.exists(odir):
    os.makedirs(odir)
    print(f"Created output directory: {odir}")

  fnames = fields_name()
  flist  = ["temperature", "composition", "viscosity", "density", "stress", "e2", "velocity"]
  frames = np.arange(0,301) #443

  # first step we create the mesh, then we just update the fields
  istep = 0
  time_series = []
  for step in frames:
    print(f"Processing step: {step}")
    if istep == 0:
      fname = get_filename(model_name, fnames[flist[0]], step)
      fname = os.path.join(mdir, fname)
      with open(fname,'rb') as f:
        bh64 = spv.BinHeader64(f)
        bh64.read_header()
      header = bh64.header
      mesh = create_mesh(header)
    mesh.mesh_cell2point = None  # reset any previous cell to point mapping
    for field in flist:
      fname = get_filename(model_name, fnames[field], step)
      fname = os.path.join(mdir, fname)
      with open(fname,'rb') as f:
        bh64 = spv.BinHeader64(f)
        bh64.read_header()
        time = bh64.header["time"]
        field_data = bh64.read_fields()
      if field == "velocity":
        get_velocity_pressure(mesh, field_data)
      else:
        mesh[field] = np.reshape(field_data[0, 0, :, :, 0], (mesh.number_of_cells), order="F")
    for field in flist:
      if field != "composition":
        if field == "velocity":
          mesh.replace_cell_field_by_point_field("velocity_r")
          mesh.replace_cell_field_by_point_field("velocity_x")
          mesh.replace_cell_field_by_point_field("pressure")
        else:
          mesh.replace_cell_field_by_point_field(field)
    mesh.save(os.path.join(odir, f"step{str(step).zfill(5)}_cellfields.vts"))
    time_series.append( (str(time), str(step).zfill(5)) )
    istep += 1

  # write the pvd file
  pvd_fname = os.path.join(odir, "timeseries_cellfields.pvd")
  if os.path.exists(pvd_fname):
    append_timeseries_pvd(pvd_fname, time_series, "cellfields", "vts")
  else:
    write_timeseries_pvd(pvd_fname, time_series, "cellfields", "vts")

  return

if __name__ == "__main__":
  test()
