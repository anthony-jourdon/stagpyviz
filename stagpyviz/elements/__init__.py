"""
Element submodule for stagpyviz.
"""

# Import classes from binheaders module
from .elements import Element, Element2D, Element3D

# Import classes from binheaders64 module
from .q1_2d import Q1_2D
from .wedge_3d import Wedge3D

# Define what gets exported with "from stagpyviz.headers import *"
__all__ = [
    'Element',
    'Element2D',
    'Element3D',
    'Q1_2D',
    'Wedge3D',
]