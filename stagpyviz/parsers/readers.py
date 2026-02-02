import numpy as np
try:
  from .binheaders import BinHeader
  from .binheaders64 import BinHeader64
except ImportError:
  from binheaders import BinHeader
  from binheaders64 import BinHeader64

def read_stag_bin(fname:str) -> tuple[dict, np.ndarray]:
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