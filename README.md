# stagpyviz
Python package to post-process StagYY models.
Currently, only the 2D spherical and 3D Yin-Yang geometries are supported. 

For the Yin-Yang geometry, the python script `examples/postproc_3d_yinyang.py` allows passing a yaml configuration file (see online documentation and example `examples/yinyang3d_example.yaml`).
Usage is simply

``` shell
python examples/postproc_3d_yinyang.py -f configuration_file.yaml
```

## Installation
For now there is no setup file, so to "install" the package you need to add it to your `PYTHONPATH`:

``` bash
export PYTHONPATH=path/to/stagpyviz:$PYTHONPATH
```

### Requirements
The package has been developed with python 3.13 therefore, for compatibility, users are encouraged to use python => 3.13.
Other used packages:

- [numpy](https://numpy.org/)
- [pyvista](https://pyvista.org/)
- [scipy](https://scipy.org/)

