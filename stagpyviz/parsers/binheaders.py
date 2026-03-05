import struct
import ctypes
import io
import logging
import numpy as np
from itertools import product

# Create a logger for this module
logger = logging.getLogger(__name__)

class BinHeader:
  """
  Class to read and extract information from the **32 bits** binary files produced by StagYY.

  :param io.BufferedReader file: A binary file object opened in 'rb' mode.

  :Attributes:

  .. py:attribute:: f
    
    The binary file object from which the header and fields will be read.

    :type: io.BufferedReader

  .. py:attribute:: int_str

    The format string for reading 32-bit signed integers.

    :type: str

  .. py:attribute:: float_str

    The format string for reading 32-bit single precision floats.

    :type: str

  .. py:attribute:: sizeof_int

    The size in bytes of a 32-bit signed integer.

    :type: int

  .. py:attribute:: sizeof_float

    The size in bytes of a 32-bit single precision float.

    :type: int

  .. py:attribute:: header

    A dictionary containing the extracted header information from the binary file.

    :type: dict

  :Methods:

  """
  def __init__(self, file:io.BufferedReader):
    self.f:io.BufferedReader = file
    self.int_str:str         = 'i'  # 32 bits signed integer
    self.float_str:str       = 'f'  # 32 bits single precision float
    self.sizeof_int:int      = ctypes.sizeof(ctypes.c_int32)
    self.sizeof_float:int    = ctypes.sizeof(ctypes.c_float)

    self.header = None
    return
  
  def read_int(self, n:int=1) -> int | list[int]:
    """
    Reads ``n`` 32-bit signed integers from the binary file 
    and returns them as an integer or a list of integers.

    :param int n: The number of integers to read. Default is 1.
    :returns: An integer if ``n`` is 1, otherwise a list of integers.
    :rtype: int | list[int]
    """
    fmt:str          = f'{n}{self.int_str}'
    bytes_read:bytes = self.f.read(n * self.sizeof_int)
    values:tuple     = struct.unpack(fmt, bytes_read)
    if n == 1: return values[0]
    else:      return list(values)

  def read_float(self, n:int=1) -> float | list[float]:
    """
    Reads ``n`` 32-bit single precision floats from the binary file 
    and returns them as a float or a list of floats.

    :param int n: The number of floats to read. Default is 1.
    :returns: A float if ``n`` is 1, otherwise a list of floats.
    :rtype: float | list[float]
    """
    fmt:str          = f'{n}{self.float_str}'
    bytes_read:bytes = self.f.read(n * self.sizeof_float)
    values:tuple     = struct.unpack(fmt, bytes_read)
    if n == 1: return values[0]
    else:      return list(values)
  
  def interpret_magic(self, magic:int) -> tuple[int,int]:
    """
    Interprets the "magic" number read from the binary file header to 
    determine what to expect from the raw binary file content.

    :param int magic: The magic number read from the binary file header.
    :returns: 
      A tuple ``(nval, magic)`` containing the number of variables per grid point 
      and the remaining magic number after extracting the number of variables.
    :rtype: tuple[int, int]
    """
    nval:int = 1
    if magic > 100:
      nval = magic // 100
    magic = magic % 100
    return nval, magic
  
  def read_header(self) -> dict:
    """
    Read the header information from the binary file and store it in the 
    ``header`` attribute as a dictionary.
    The header contains the following information:

    - ``nval``: Number of variables per grid point. 4 for velocity-pressure fields, 1 for scalar fields.

    - ``xyp``: 1 if the file contains velocity-pressure fields, 0 otherwise.

    - ``ntot``: Total number of grid points in each dimension ``(x, y, z, block)``.

    - ``aspect``: Aspect ratio of the grid, 2 entries.

    - ``npar``: Number grid points per parallel subdomains in each dimension ``(x, y, z, block)``.

    - ``rpoints``: Radial coordinates of grid points in the z direction. Shape is ``(ntot[2]+1,)``.

    - ``rcells``: Radial coordinates of cell centers in the z direction. Shape is ``(ntot[2],)``.

    - ``rcmb``: Radius of the core-mantle boundary.

    - ``istep``: Simulation step number.

    - ``time``: Simulation time.

    - ``erupta_total``: 2 entries if the magic number is greater or equal to 12, otherwise 1 entry.

    - ``intruda``: Only if the magic number is greater or equal to 12. Shape ``(2,)``.

    - ``TTGmass``: Only if the magic number is greater or equal to 12. Shape ``(3,)``.

    - ``Tbot``: Only if the magic number is greater or equal to 6.

    - ``Tcore``: Only if the magic number is greater or equal to 10.

    - ``water``: Only if the magic number is greater or equal to 11.

    - ``x``, ``y``, ``z``: 
      Coordinates of grid points in each dimension, 
      only if the magic number is greater or equal to 3.

    - ``scale``: Scaling factor for the field values, only if ``nval`` is greater or equal to 4.

    - ``ncpu``: Number of parallel subdomains in each dimension, calculated as ``ntot[i] / npar[i]``.

    - ``npi``: 
      Total number of grid points per parallel subdomain, calculated as
      ``(ncpu[0]+xyp) * (ncpu[1]+xyp) * ncpu[2] * ncpu[3] * nval``.

    :return: The header information as a dictionary.
    :rtype: dict
    """
    header = {}
    magic = self.read_int(1)
    header["nval"], magic = self.interpret_magic(magic)
    if magic >= 9 and header["nval"] == 4: 
      header["xyp"] = 1
    else: 
      header["xyp"] = 0
    header["ntot"]   = self.read_int(4)
    header["aspect"] = self.read_float(2)
    header["npar"]   = self.read_int(4)
    nz2    = 2*header["ntot"][2]+1
    zg     = self.read_float(nz2)
    zg     = np.resize(zg, (header["ntot"][2] + 1, 2)) 
    header["rpoints"] = zg[:,0]
    header["rcells"]  = zg[0:header["ntot"][2],1]
    header["rcmb"]    = self.read_float(1)
    header["istep"]   = self.read_int(1)
    header["time"]    = self.read_float(1)
    header["erupta_total"] = self.read_float(1)
    if magic >= 12:
      header["erupta_total"] = [header["erupta_total"]]
      header["erupta_total"].append(self.read_float(1))
    header["intruda"] = self.read_float(2) if magic >= 12 else 0.0
    header["TTGmass"] = self.read_float(3) if magic >= 12 else 0.0
    header["Tbot"] = self.read_float(1) if magic >= 6 else 0.0
    header["Tcore"] = self.read_float(1) if magic >= 10 else 0.0
    header["water"] = self.read_float(1) if magic >= 11 else 0.0
    if magic >= 3:
      header["x"] = self.read_float(header["ntot"][0])
      header["y"] = self.read_float(header["ntot"][1])
      header["z"] = self.read_float(header["ntot"][2])
    header["scale"] = self.read_float(1) if header["nval"] >= 4 else 1.0

    header["ncpu"] = [ int(nt / npa) for nt, npa in zip(header["ntot"], header["npar"]) ]
    header["npi"]  = (
      (header["ncpu"][0]+header["xyp"]) * 
      (header["ncpu"][1]+header["xyp"]) * 
      header["ncpu"][2] * 
      header["ncpu"][3] * 
      header["nval"]
    )

    self.header = header

    logger.debug(f'Magic number: {magic}')
    logger.debug(f"nval = {header['nval']}")
    logger.debug(f"xyp = {header['xyp']}")
    logger.debug(f"ntot: {header['ntot']}")
    logger.debug(f"aspect: {header['aspect']}")
    logger.debug(f"npar: {header['npar']}")
    logger.debug(f"nz2: {nz2}, zg (first 10 values): {zg[:10]}")
    logger.debug(f"rcmb: {header['rcmb']}")
    logger.debug(f"istep: {header['istep']}")
    logger.debug(f"time: {header['time']}")
    logger.debug(f"eruption total: {header['erupta_total']}")
    logger.debug(f"intruda: {header['intruda']}")
    logger.debug(f"TTGmass: {header['TTGmass']}")
    logger.debug(f"Tbot: {header['Tbot']}")
    logger.debug(f"Tcore: {header['Tcore']}")
    logger.debug(f"water: {header['water']}")
    logger.debug(f"ncpu: {header['ncpu']}")
    logger.debug(f"npi: {header['npi']}")
    #logger.debug(f"x: {header['x'] if magic >=3 else 'N/A'}")
    #logger.debug(f"y: {header['y'] if magic >=3 else 'N/A'}")
    #logger.debug(f"z: {header['z'] if magic >=3 else 'N/A'}")
    logger.debug(f"scale: {header['scale']}")
    return 
  
  def read_fields(self) -> np.ndarray:
    """
    Read the field data from the binary file. Very inspired from this 
    `stagpy <https://github.com/StagPython/StagPy/blob/master/src/stagpy/stagyyparsers.py>`_
    function.

    :return: 
      A 5D numpy array containing the field data, indexed as
      ``(component, x, y, z, block)``.
    :rtype: np.ndarray
    """
    if self.header is None:
      raise RuntimeError("Header must be read before reading fields.")
    header = self.header
    flds = np.zeros(
      (
        header["nval"],
        header["ntot"][0] + header["xyp"],
        header["ntot"][1] + header["xyp"],
        header["ntot"][2],
        header["ntot"][3],
      )
    )

    # loop over parallel subdomains
    for icpu in product(
      range(header["npar"][3]),
      range(header["npar"][2]),
      range(header["npar"][1]),
      range(header["npar"][0]),
    ):
      # read the data for one CPU
      data_cpu:np.ndarray = np.asarray(self.read_float(header["npi"])) * header["scale"]
            
      # icpu is (icpu block, icpu z, icpu y, icpu x)
      # data from file is transposed to obtained a field
      # array indexed with (x, y, z, block), as in StagYY
      flds[
        :,
        icpu[3] * header["ncpu"][0] : (icpu[3] + 1) * header["ncpu"][0] + header["xyp"],  # x
        icpu[2] * header["ncpu"][1] : (icpu[2] + 1) * header["ncpu"][1] + header["xyp"],  # y
        icpu[1] * header["ncpu"][2] : (icpu[1] + 1) * header["ncpu"][2],  # z
        icpu[0] * header["ncpu"][3] : (icpu[0] + 1) * header["ncpu"][3],  # block
      ] = np.transpose(
        data_cpu.reshape(
          (
            header["ncpu"][3],
            header["ncpu"][2],
            header["ncpu"][1] + header["xyp"],
            header["ncpu"][0] + header["xyp"],
            header["nval"],
          )
        )
      )
      """
      if hdr.sfield:
        # for surface fields, variables are written along z direction
        flds = np.swapaxes(flds, 0, 3)
      """
    return flds

def test():
  import os
  logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  )
  fname = os.path.join(os.environ["POSTPROC"],"StagYY/PJB6_YS1_Rh32_vp00039")
  with open(fname,'rb') as f:
    bh64 = BinHeader(f)
    bh64.read_header()
  
  return

if __name__ == "__main__":
  test()