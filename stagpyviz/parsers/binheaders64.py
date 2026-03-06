import ctypes
import io
import logging

# Use relative import when imported as module, absolute when run directly
try:
    from .binheaders import BinHeader
except ImportError:
    from binheaders import BinHeader

class BinHeader64(BinHeader):
  """
  Class to read the binary header of a StagYY output file, for **64-bit** files. 
  It inherits from :py:class:`BinHeader <stagpyviz.BinHeader>`, 
  and overrides the necessary attributes and methods to handle 64-bit data. 
  """
  def __init__(self,file:io.BufferedReader):
    super().__init__(file)
    self.int_str:str      = 'q'  # 64 bits signed integer
    self.float_str:str    = 'd' # 64 bits double precision float
    self.sizeof_int:int   = ctypes.sizeof(ctypes.c_int64)
    self.sizeof_float:int = ctypes.sizeof(ctypes.c_double)
    return
  
  def interpret_magic(self, magic) -> tuple[int,int]:
    if magic < 8000:
      raise ValueError(f"Magic number = {magic} is invalid for 64-bit header")
    #magic = magic % 8000
    magic -= 8000
    return super().interpret_magic(magic)
    
def test():
  import os
  logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  )
  fname = os.path.join(os.environ["POSTPROC"],"StagYY/Ra7_pl_nocrust_HR_T00004")
  with open(fname,'rb') as f:
    bh64 = BinHeader64(f)
    bh64.read_header()
    fields = bh64.read_fields()
  return

if __name__ == "__main__":
  test()