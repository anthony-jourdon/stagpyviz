"""stagpyviz - A Python package for visualizing StagYY output files.

This package provides tools for reading and visualizing StagYY geodynamic
simulation output files.
"""
import pint

units = pint.UnitRegistry()

# Import the submodules
from . import parsers
from . import mesh
from . import elements
from . import utils
from . import scaling
from . import fields

# Import specific classes for convenience at package level
from .parsers import BinHeader, BinHeader64, read_stag_bin
from .mesh import SphericalMesh, Hex2DMesh, UnstructuredSphere, ShellMesh, YinYangMesh
from .elements import Element, Element2D, Element3D, SurfaceElement, Q1_2D, P1_2D, Wedge3D, P1_2D_R3
from .utils.timeseries import write_timeseries_pvd, append_timeseries_pvd, timeseries_compare, timeseries_write_new, timeseries_append, timeseries_write, timeseries_process, timeseries_write_step
from .scaling import Scaling, scaling_factors
from .fields import Field, StagField, DerivedField, Velocity, Pressure, SphericalField, CartesianGradient, SphericalVectorGradient, fields_instances
from .utils.io_utils import IOutils

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
  'UnstructuredSphere',
  'ShellMesh',
  'YinYangMesh',
  'elements',
  'Element',
  'Element2D',
  'Element3D',
  'SurfaceElement',
  'Q1_2D',
  'P1_2D',
  'Wedge3D',
  'P1_2D_R3',
  'utils',
  'write_timeseries_pvd',
  'append_timeseries_pvd',
  'timeseries_compare',
  'timeseries_write_new',
  'timeseries_append',
  'timeseries_write',
  'timeseries_process',
  'timeseries_write_step',
  'scaling',
  'Scaling',
  'scaling_factors',
  'fields',
  'Field',
  'StagField',
  'DerivedField',
  'Velocity',
  'Pressure',
  'SphericalField',
  'CartesianGradient',
  'SphericalVectorGradient',
  'fields_instances',
  'IOutils'
]