import numpy as np
try:
  from .binheaders import BinHeader
  from .binheaders64 import BinHeader64
except ImportError:
  from binheaders import BinHeader
  from binheaders64 import BinHeader64

def read_stag_bin(fname:str) -> tuple[dict, np.ndarray]:
  """
  Read a StagYY binary file and return the header and field data.
  Automatically detects whether the file is in 32-bit or 64-bit format and uses the appropriate reader.

  :param str fname: The path to the StagYY binary file.
  :return: A tuple containing the header (as a dictionary) and the field data (as a NumPy array).
  :rtype: tuple[dict, np.ndarray]
  """
  try:
    with open(fname,'rb') as f:
      bh64 = BinHeader64(f)
      bh64.read_header()
      header = bh64.header
      field_data = bh64.read_fields()
  except:
    with open(fname,'rb') as f:
      bh = BinHeader(f)
      bh.read_header()
      header = bh.header
      field_data = bh.read_fields()
  return header, field_data