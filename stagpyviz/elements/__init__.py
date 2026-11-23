"""
Element submodule for stagpyviz.
"""

# Import classes from binheaders module
from .elements import Element, Element2D, Element3D, SurfaceElement

# Import classes from binheaders64 module
from .q1_2d import Q1_2D
from .p1_2d import P1_2D, P1_2D_R3
from .wedge_3d import Wedge3D
from .q1_3d import Q1_3D

# Define what gets exported with "from stagpyviz.headers import *"
__all__ = [
    'Element',
    'Element2D',
    'Element3D',
    'SurfaceElement',
    'Q1_2D',
    'Wedge3D',
    'P1_2D',
    'P1_2D_R3',
    'Q1_3D',
]