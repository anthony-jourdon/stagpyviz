Scaling parameters
==================

The script ``scaling_nondim.py`` located in the ``examples`` directory 
allows the user to print the scaling factors, dimension and adimension
values.

The script uses terminal prompts to ask the user for the field, values and units.
Simply call the script and follow the prompts to get the 
scaling information for your model.

Available modes: 

- ``print``: prints out the scaling factors and their units
- ``dim``: dimensions the provided value for a given field
- ``adim``: adimensions the provided value for a given field

The script can dimension or adimension as many fields as you want, 
but can only work in one mode at a time therefore, it needs to be relaunched if
two distinct modes are needed.

.. code-block:: bash
  
  python examples/scaling_nondim.py
