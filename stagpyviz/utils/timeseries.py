
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
