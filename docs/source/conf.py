
import pathlib
import os
import sys

# project root directory to the sys.path
project_root = pathlib.Path(__file__).parents[2].resolve().as_posix()
module_dir   = os.path.join(project_root,'stagpyviz')

sys.path.insert(0, project_root)
sys.path.insert(0, module_dir)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'stagpyviz'
copyright = '2026, Anthony Jourdon'
author = 'Anthony Jourdon'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx.ext.duration',
  'sphinx.ext.intersphinx',
  'sphinx.ext.doctest',
  'sphinx.ext.autodoc',
  "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = []
include_patterns = [
  '**',
  '../stagpyviz/**'
]

intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
  "matplotlib": ("https://matplotlib.org/stable/", None),
  "sympy": ("https://docs.sympy.org/latest/", None),
  "pyvista": ("https://docs.pyvista.org/", None),
  "scipy": ("https://docs.scipy.org/doc/scipy/", None),
  "pint": ("https://pint.readthedocs.io/en/stable/", None),
}

rst_prolog = """
.. _pyvista.UnstructuredGrid: https://docs.pyvista.org/api/core/_autosummary/pyvista.unstructuredgrid
.. _pint: https://pint.readthedocs.io/en/stable/
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

html_theme_options = {
  'prev_next_buttons_location': 'both',
  #'style_external_links': True,
}

html_favicon = 'figures/favicon.png'

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "mixed"