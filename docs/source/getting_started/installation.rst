Installation
------------

To install the package, go to the `repository <https://github.com/anthony-jourdon/stagpyviz>`_ and download the sources.

Option 1: using pip (recommended)
.................................
Once the sources are downloaded, go to the package directory and run:

.. code-block:: bash
  
  pip install .

This will automatically install the package and all its dependencies.

Option 2: adding to PYTHONPATH
..............................
After downloading the sources, you can add the package to your ``PYTHONPATH`` environment variable. 
If a virtual environment is used, the package will be available only in this environment.

.. code-block:: bash

  export PYTHONPATH=path/to/stagpyviz:$PYTHONPATH

This method requires that all dependencies are already installed in the environment.

Requirements
............
The package has been developed with python 3.13 therefore, for compatibility, users are encouraged to use python 3.13 and above.
Other used packages:

- `numpy <https://numpy.org/>`_
- `pyvista <https://pyvista.org/>`_
- `scipy <https://scipy.org/>`_
- `pint <https://pint.readthedocs.io/en/stable/>`_