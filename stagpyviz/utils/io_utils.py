import os
import numpy as np

class IOutils:
  """
  Class to manage the paths and file management for the postprocessing of StagYY 3D YinYang models.
  This class is used to store the paths to the model directory and output directory, 
  as well as the list of fields to be added to the output mesh.
  This class also contains the stepping parameters for time series processing and the 
  time series information to be written in the pvd file.

  :param str model_name: Name of the model.
  :param str model_dir: Name of the model directory.
  :param str basedir: Absolute path to the base model directory.
  :param str pvd: Name of the output pvd file.
  :param str output_dir: Absolute path to the output directory.
  :param list[str] output_fields: List of fields that can be retrieved from binary files output by StagYY.
  :param list[str] regions: List of region fields to be merged into a single regions field in the output mesh, default: ["composition"].
  :param bool is_surface: Boolean indicating whether to output the surface (True) or the volume (False), default: False.
  :param int step: Step number to process, if None, the entire directory will be processed, default: None.
  :param int step_start: Step number to start processing, default: 0.
  :param int step_end: Step number to end processing, if None, the entire directory will be processed, default: None.
  :param int dstep: Step increment for processing, default: 1.
  :param bool reset_fields: Boolean indicating whether to reset the timeseries, default: False.
  :param str prefix: Prefix to be added to the output file names, default: "".

  :Attributes:

    .. py:attribute:: model

      Name of the model.

      :type: str
      :canonical: stagpyviz.IOutils.model

    .. py:attribute:: mdir

      Name of the model directory.

      :type: str
      :canonical: stagpyviz.IOutils.mdir

    .. py:attribute:: basedir

      Absolute path to the directory containing the model directory.

      :type: str
      :canonical: stagpyviz.IOutils.basedir

    .. py:attribute:: model_dir

      Absolute path to the model directory. Constructed from basedir and mdir.

      :type: str
      :canonical: stagpyviz.IOutils.model_dir

    .. py:attribute:: output_dir

      Absolute path to the output directory.

      :type: str
      :canonical: stagpyviz.IOutils.output_dir

    .. py:attribute:: output_fields

      List of fields that can be retrieved from binary files output by StagYY.
      
      :type: list[str]
      :canonical: stagpyviz.IOutils.output_fields

    .. py:attribute:: pvd

      Name of the output pvd file.

      :type: str
      :canonical: stagpyviz.IOutils.pvd

    .. py:attribute:: prefix

      Prefix to be added to the output file names.

      :type: str
      :canonical: stagpyviz.IOutils.prefix

    .. py:attribute:: time

      Time value for the current step. Initialized to None, filled when processing the binary files.

      :type: float|None
      :canonical: stagpyviz.IOutils.time

    .. py:attribute:: timeseries

      Dictionary to store the time series information to be written in the pvd file. Initialized with empty lists for "time" and "step".

      :type: dict[str, list]
      :canonical: stagpyviz.IOutils.timeseries

    .. py:attribute:: regions

      List of region fields to be merged into a single regions field in the output mesh.

      :type: list[str]
      :canonical: stagpyviz.IOutils.regions

    .. py:attribute:: is_surface

      Boolean indicating whether to output the surface (True) or the volume (False).

      :type: bool
      :canonical: stagpyviz.IOutils.is_surface

    .. py:attribute:: step

      Step number. Initialized to provided step argument, or None if not provided.

      :type: int|None
      :canonical: stagpyviz.IOutils.step

    .. py:attribute:: step_start

      Step number from which to start processing. Initialized to provided step_start argument, or 0 if not provided.

      :type: int
      :canonical: stagpyviz.IOutils.step_start

    .. py:attribute:: step_end

      Step number at which to end processing. Initialized to provided step_end argument, or None if not provided.

      :type: int|None
      :canonical: stagpyviz.IOutils.step_end

    .. py:attribute:: steps_idx

      Numpy array of step numbers to process, generated from step_start, step_end and dstep. Initialized to None if step_end is not provided.

      :type: np.ndarray|None
      :canonical: stagpyviz.IOutils.steps_idx

    .. py:attribute:: volume_fields

      Dictionary mapping the names of the volume fields that can be retrieved from binary files output by StagYY 
      to their raw names in the binary files.

      :type: dict[str, str|tuple[str]]
      :canonical: stagpyviz.IOutils.volume_fields

    .. py:attribute:: surface_fields

      Dictionary mapping the names of the surface fields that can be retrieved from binary files output by StagYY 
      to their raw names in the binary files.

      :type: dict[str, str|tuple[str]]
      :canonical: stagpyviz.IOutils.surface_fields

    .. py:attribute:: filelist

      Dictionary mapping the names of all the fields that can be retrieved from binary files output by StagYY
      (both volume and surface fields) to their raw names in the binary files.

      :type: dict[str, str|tuple[str]]
      :canonical: stagpyviz.IOutils.filelist

  Currently the supported fields that can be retrieved from binary files output by StagYY are:

  .. code-block:: python

    volume_fields = {
      "basalt":      "bs",
      "composition": "c",
      "density":     "rho",
      "divergence":  "div",
      "e2" :         "ed",
      "harzburgite": "hz",
      "nrc":         "nrc", # ??
      "pressure":    "vp", # pressure is stored in the 4th component of the velocity file
      "primordial":  "prm",
      "proterozoic": "prot",
      "stress" :     "str",
      "temperature": ("t", "T"),
      "tracer":      "tra",
      "velocity":    "vp",
      "viscosity":   "eta",
      "vorticity":   "vor"
    }

    surface_fields = {
      "topography":  "cs",
      "heatflux":    "hf",
    }

    filelist = {
      **volume_fields,
      **surface_fields
    }
      
  """
  def __init__(
      self,
      model_name:str, 
      model_dir:str, 
      basedir:str,
      pvd:str,
      output_dir:str,
      output_fields:list[str],
      **kwargs
    ):
    self.model:str               = model_name
    self.mdir:str                = model_dir
    self.basedir:str             = basedir
    self.output_fields:list[str] = output_fields
    self.pvd:str                 = pvd
    self.prefix:str = kwargs.get("prefix", "")
    # path to model directory
    self.model_dir = os.path.join(basedir,model_dir)
    self.time:float|None = None
    self.timeseries:dict[str,list] = {
      "time": [], # store time as string
      "step": [], # store step number as string
    }
    # output directory
    self.output_dir:str = output_dir
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
      print(f"Created output directory: {self.output_dir}")

    self.is_surface:bool = kwargs.get("is_surface", True)

    # stepping parameters for time series processing
    self.step:int|None  = kwargs.get("step", None)
    self.step_start:int = kwargs.get("step_start", 0)
    self.step_end:int   = kwargs.get("step_end", None)
    dstep:int           = kwargs.get("dstep", 1)
    self.steps_idx:np.ndarray|None = np.arange(self.step_start, self.step_end+1, dstep) if self.step_end is not None else None

    self.regions:list[str] = kwargs.get("regions", ["composition"])
    self.reset_fields:bool = kwargs.get("reset_fields", False)

    self.volume_fields = {
      "basalt":      "bs",
      "composition": "c",
      "density":     "rho",
      "divergence":  "div",
      "e2" :         "ed",
      "harzburgite": "hz",
      "nrc":         "nrc", # ??
      "pressure":    "vp", # pressure is stored in the 4th component of the velocity file
      "primordial":  "prm",
      "proterozoic": "prot",
      "stress" :     "str",
      "temperature": ("t", "T"),
      "tracer":      "tra",
      "velocity":    "vp",
      "viscosity":   "eta",
      "vorticity":   "vor"
    }

    self.surface_fields = {
      "topography":  "cs",
      "heatflux":    "hf",
    }

    self.filelist = {
      **self.volume_fields,
      **self.surface_fields
    }
    return
  
  def compose_filename(self, field:str, step:int) -> str|None:
    """
    Return the path to the binary file corresponding to the given field and step number, if it exists, otherwise return None.
    The filename is reconstructed such that:

    .. code-block:: python

      fname = f"{model_name}_{field}{step:05d}"
      fpath = os.path.join(basedir, model_dir, fname)

    :param str field: StagYY name of the field to retrieve.
    :param int step: Step number to retrieve.
    :return str|None: Path to the binary file corresponding to the given field and step number, if it exists, otherwise None.

    """
    fname = f"{self.model}_{field}{step:05d}"
    fpath = os.path.join(self.model_dir, fname)
    print(f"Checking for file: {fpath}")
    if os.path.exists(fpath):
      print("\tFound.")
      return fpath
    else:
      print("\tNot found.")
      return None

  def get_field_filename(self, field:str, step:int) -> str|None:
    """
    Return the path to the binary file corresponding to the given field and step number, if it exists, otherwise return None.
    This function uses the filelist dictionary to find the raw name of the field, and then uses the 
    :py:meth:`compose_filename <stagpyviz.IOutils.compose_filename>` method to reconstruct the filename and check if it exists. 
    If the field is not found in the filelist, or if the file does not exist, None is returned.

    :param str field: Name from the :py:attr:`filelist <stagpyviz.IOutils.filelist>` of the field to retrieve.
    :param int step: Step number to retrieve.
    :return str|None: Path to the binary file corresponding to the given field and step number, if it exists, otherwise None.
    """
    raw_name = self.filelist.get(field, None)
    if raw_name is None:
      print(f"Field {field} not found in filelist.")
      return None
    
    if isinstance(raw_name, tuple):
      for name in raw_name:
        fpath = self.compose_filename(name, step)
        if fpath is not None:
          return fpath
      print(f"None of the possible filenames for field {field} were found.")
      return None
    else:
      fpath = self.compose_filename(raw_name, step)
      if fpath is not None:
        return fpath
      else:
        return None
  
  def __str__(self) -> str:
    s  = f"Model: {self.model}\n"
    s += f"Model directory: {self.model_dir}\n"
    s += f"Output directory: {self.output_dir}\n"
    s += f"Output fields: {self.output_fields}\n"
    s += f"PVD file: {self.pvd}\n"
    s += f"Regions: {self.regions}\n"
    s += f"Reset: {self.reset_fields}\n"
    s += f"Prefix: {self.prefix}\n"
    s += f"Step: {self.step}\n"
    s += f"Step start: {self.step_start}\n"
    s += f"Step end: {self.step_end}\n"
    s += f"Delta step: {self.steps_idx[1] - self.steps_idx[0] if self.steps_idx is not None else 1}\n"
    s += f"Surface: {self.is_surface}\n"
    s += f"File list: {self.filelist}\n"
    return s
