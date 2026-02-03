"""stagpyviz - A Python package for visualizing StagYY output files.

This package provides tools for reading and visualizing StagYY geodynamic
simulation output files.
"""
# Import the submodules
from . import parsers
from . import mesh
from . import elements
from . import utils

# Import specific classes for convenience at package level
from .parsers import BinHeader, BinHeader64, read_stag_bin
from .mesh import SphericalMesh, Hex2DMesh, YinYangMesh
from .elements import Element, Element2D, Q1_2D
from .utils.timeseries import write_timeseries_pvd, append_timeseries_pvd

# Define version
__version__ = '0.1.0'

# Define what gets exported with "from stagpyviz import *"
__all__ = [
  'parsers',
  'BinHeader',
  'BinHeader64',
  'read_stag_bin',
  'mesh',
  'SphericalMesh',
  'Hex2DMesh',
  'YinYangMesh',
  'elements',
  'Element',
  'Element2D',
  'Q1_2D',
  'utils',
  'write_timeseries_pvd',
  'append_timeseries_pvd',
]