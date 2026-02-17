import re
import os
from pathlib import Path

def timeseries_process(pvdfname:str|Path, start_line:int|None=None) -> dict:
  if not isinstance(pvdfname, (str, Path)):
    raise ValueError("pvdfname must be a string or a Path object.")
  if not os.path.exists(pvdfname):
    raise FileNotFoundError(f"File {pvdfname} does not exist.")
  timeseries = {
    "time":     [], # store time as string
    "step_dir": [], # store step directory
    "step":     [], # store step number as string
    "fname":    [], # store file name
    "line":     []  # store line number in the .pvd file
  }
  with open(pvdfname,'r') as fp:
    if start_line is None:
      # skip the 3 first lines
      fp.readline()
      fp.readline()
      fp.readline()
      l = 3
    else:
      l = 0
      for _ in range(start_line):
        fp.readline()
        l += 1
    for line in fp:
      res = line.rsplit(sep='"')[1::2]
      if not res: continue
      time_str = res[0]
      file_str = res[1]
      timeseries["time"].append(time_str)
      timeseries["fname"].append(file_str)
      timeseries["line"].append(l)
      matchObj = re.match(r'(?:^(step\d+)/)?.*0(\d+)(.*)',file_str)
      if matchObj:
        if matchObj.group(1):
          timeseries["step_dir"].append(matchObj.group(1))
        else:
          timeseries["step_dir"].append('')
        timeseries["step"].append(matchObj.group(2))
      l += 1
  return timeseries
  
def timeseries_compare(pvdfname1:str|Path, pvdfname2:str|Path) -> tuple[dict, dict|None, dict|None]:
    """
    Compare two timeseries from .pvd files. 
    If they have different number of lines, 
    it will process the one with more lines starting from the line where they differ, 
    and return the new timeseries along with the previous timeseries and the full timeseries. 
    If they have the same number of lines, it will simply return the timeseries for the first file and None for the others.
    
    :param pvdfname1: PVD file name for the first timeseries
    :type pvdfname1: str|Path
    :param pvdfname2: PVD file name for the second timeseries
    :type pvdfname2: str|Path
    :return: Description
    :rtype: tuple[dict, dict | None, dict | None]
    """
    timeseries1 = timeseries_process(pvdfname1)
    timeseries2 = timeseries_process(pvdfname2)
    nlines1 = len(timeseries1['line'])
    nlines2 = len(timeseries2['line'])
    if nlines1 != nlines2:
      if nlines1 < nlines2:
        restart_line = timeseries2['line'][nlines1]
        return timeseries_process(pvdfname2, restart_line), timeseries1, timeseries2
      else:
        restart_line = timeseries1['line'][nlines2]
        return timeseries_process(pvdfname1, restart_line), timeseries2, timeseries1
    else:
      print("Timeseries have the same number of lines.")
      return timeseries1, None, None

def timeseries_write_step(time:str, step:str, extension:str, prefix:str|None=None) -> str:
    if prefix is None:
      s = f'  <DataSet timestep="{time}" file="step{step}.{extension}"/>\n'
    else:
      s = f'  <DataSet timestep="{time}" file="step{step}{prefix}.{extension}"/>\n'
    return s

def timeseries_write_new(timeseries:dict, prefix:str|None=None, extension:str="vts") -> str:
    nsteps = len(timeseries["time"])
    s = "<?xml version=\"1.0\"?>\n"
    s += "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    s += "<Collection>\n"
    for step in range(nsteps):
      time = timeseries["time"][step]
      step_num = timeseries["step"][step]
      s += timeseries_write_step(time, step_num, extension, prefix)
    s += "</Collection>\n"
    s += "</VTKFile>"
    return s
  
def timeseries_append(content:str, timeseries:dict, prefix:str|None=None, extension:str="vts") -> str:
    nsteps = len(timeseries["time"])
    s = ""
    for line in content:
      if '</Collection>' in line:
        break
      s += line
    for step in range(nsteps):
      time = timeseries["time"][step]
      step_num = timeseries["step"][step]
      s += timeseries_write_step(time, step_num, extension, prefix)
    s += '  </Collection>\n'
    s += '</VTKFile>'
    return s
  
def timeseries_write(fname:str, timeseries:dict, prefix:str|None=None, extension:str="vts", erase:bool=False) -> None:
    if os.path.exists(fname) and not erase:
      with open(fname,'r') as f:
        content = f.readlines()
      s = timeseries_append(content, timeseries, prefix, extension)
    else:
      s = timeseries_write_new(timeseries, prefix, extension)
    with open(fname,'w') as f:
      f.write(s)
    return


def write_timeseries_pvd(fname:str,time_series:list,prefix:str|None=None,extension:str="vts") -> None:
  if not fname.endswith('.pvd'):
    fname += '.pvd'
  with open(fname,'w') as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for time_entry in time_series:
      if prefix is None:
        f.write(f'    <DataSet timestep="{time_entry[0]}" file="step{time_entry[1]}.{extension}"/>\n')
      else:
        f.write(f'    <DataSet timestep="{time_entry[0]}" file="step{time_entry[1]}_{prefix}.{extension}"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')
  return

def append_timeseries_pvd(fname:str,time_series:list,prefix:str|None=None,extension:str="vts") -> None:
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
      if prefix is None:
        f.write(f'    <DataSet timestep="{time_entry[0]}" file="step{time_entry[1]}.{extension}"/>\n')
      else:
        f.write(f'    <DataSet timestep="{time_entry[0]}" file="step{time_entry[1]}_{prefix}.{extension}"/>\n')
    # write closing tags
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')
  return
