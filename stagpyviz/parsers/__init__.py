"""Headers submodule for stagpyviz.

This submodule contains binary header readers for StagYY output files.
"""

# Import classes from binheaders module
from .binheaders import BinHeader

# Import classes from binheaders64 module
from .binheaders64 import BinHeader64

# Import function to read StagYY binary files
from .readers import read_stag_bin

# Define what gets exported with "from stagpyviz.headers import *"
__all__ = [
    'BinHeader',
    'BinHeader64',
    'read_stag_bin',
]