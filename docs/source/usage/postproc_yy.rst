Yin-Yang 3D grid post-processing
================================

For a simple post-processing of StagYY output on a Yin-Yang grid,
the script ``postproc_3d_yinyang.py`` provides a way to write vtu files 
and a pvd timeseries file for visualization in Paraview. 
The script is located in the ``examples`` directory of the package.

The interface with the script uses a configuration file in yaml format.
The configuration file is structured in three sections:

1. Paths
--------

This section contains the paths to the input and output directories.
It is structured as follows:

.. code-block:: yaml

  paths:
    model: "model_name"
    directory: "model_directory"
    base:
      model: "path/to/model/directory"
      output: "path/to/output/directory"
    # Optional
    pvd: "pvd_fname.pvd"
    prefix: "prefix_for_output_files"

The script will automatically join the base path with the model directory to find the 
input files and to write the output files.
The output directory will be created if it does not exist and will be given the name of the model directory.

Optionally, the user can specify a name for the pvd timeseries file using the ``pvd`` key in ``paths``. 
If not specified, the pvd file will be attributed a default name based on the model's name.

A prefix can also be specified for the output files using the ``prefix`` key in ``paths``.
By default, if only the surface of the model is written, the prefix ``surface`` 
is appended to any prefix provided by the user. 
If the full 3D grid is written, no default prefix is added.

2. Fields
---------

This section contains the list of fields to be written in the output files.
It is structured as follows:

.. code-block:: yaml

  fields:
    process:
      - "field1"
      - "field2"
      - "field3"

    regions:
      - "region1"
      - "region2"
      - "region3"

    surface: true

The ``process`` key contains the list of fields to be written in the output files.
The currently available fields are:

- ``velocity``: the 3D velocity field, written as a vector field.

- ``velocity_r``: the velocity field in spherical coordinates.

- ``e2``: the second invariant of the strain rate tensor.

- ``temperature``: the temperature field.

- ``divergence``: the divergence of the velocity field in spherical coordinates with the radial component removed.

- ``composition``: the composition field.

- ``primordial``: the primordial field.

- ``proterozoic``: the proterozoic field.

- ``viscosity``: the viscosity field.

- ``grad_T``: the cartesian gradient of the temperature field.

- ``grad_T_r``: the spherical gradient of the temperature field.

- ``stress``: the second invariant of the deviatoric stress tensor.

- ``pressure``: the pressure field.

- ``nrc``

- ``vorticity``: the vorticity field of the velocity field in spherical coordinates with the radial component removed.

The ``regions`` are a special field. Depending on the stag version regions can be written as 
separate fields. To group them all under the same field called ``regions``, the user can provide a list 
of fields to be identified as a region.
Then, the script will take the fields provided in the list and write them as a single field 
called ``regions`` in the output files with an index corresponding to each region in the order 
they appear in the list starting from 1, 0 being reserved for the background region (the mantle).

The ``surface`` key is a boolean that indicates whether to write only the surface of the model 
or the full 3D grid. By default, the script writes the full 3D grid of the model.

By default, if a field is required but not available, the script will simply ignore it
and prompt a message to the user.
However, some fields require specific data and will generate an error if the required data is not available.
For example, the fields ``grad_T`` and ``grad_T_r`` require the temperature field to be available 
in the input files to compute the gradients.

3. Steps
--------

This section contains information about the steps to process. There are three possibilities.

3.1. Process a single step
..........................

To process a single step, the user can simply provide the following information:

.. code-block:: yaml

  steps:
    step: 1 # integer step number to process

The required step will be processed and a vtu as well as a pvd file will be written.

3.2. Process several steps
..........................

To process several steps, the user can provide the following information:

.. code-block:: yaml

  steps:
    start: 1 # integer step number to start processing
    end: 10 # integer step number to end processing
    delta: 1 # integer step number increment

In addition, if a pvd file is already existing in the output directory, 
the script will start from the last written step in the pvd file 
and process the next steps until the end step provided by the user and will automatically append the new steps
to the existing pvd file.
To avoid this behaviour and enforce to start from the provided start step, the user can use:

.. code-block:: yaml

  steps:
    start: 1 # integer step number to start processing
    end: 10 # integer step number to end processing
    delta: 1 # integer step number increment
    reset: true

Setting reset to true will ignore any existing pvd file and 
erase any existing pvd file in the output directory.

3.3. Process all available steps
................................

Finally to process all available steps, the user can provide the following information:

.. code-block:: yaml

  steps:
    start: 0 # any integer step number to start processing

Not providing an end step will make the script look for all available steps in 
the input directory and process them until the last available step.
Again, if a pvd file is already existing in the output directory, 
the script will start from the last written step in the pvd file 
and process the next steps until the last available step and will automatically append the new steps
to the existing pvd file. 
To avoid this behaviour and enforce to start from the provided start step and erase 
the already existing pvd file, the user can use:

.. code-block:: yaml

  steps:
    start: 0 # any integer step number to start processing
    reset: true

4. A complete example
---------------------
Here is a complete example of a configuration file for the script:

.. code-block:: yaml

  paths:
    model: "model_name"
    directory: "model_directory"
    base:
      model: "path/to/model/directory"
      output: "path/to/output/directory"

  fields:
    process:
      - "velocity"
      - "temperature"
      - "composition"
      - "grad_T"
      - "grad_T_r"
      - "divergence"

    regions:
      - "primordial"
      - "archean"
      - "proterozoic"

    surface: true

  steps:
    start: 0
    end: 10
    delta: 1
    reset: true

5. Running the script
----------------------
To run the script, simply execute the following command in the terminal:

.. code-block:: bash

  python postproc_3d_yinyang.py -f config.yaml

Where ``config.yaml`` is the name of the configuration file.