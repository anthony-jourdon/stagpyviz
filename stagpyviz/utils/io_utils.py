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

  Currently the supported fields that can be retrieved from binary files output by StagYY are:

  .. code-block:: python

    filelist = {
      "composition": "c",
      "divergence" : "div",
      "e2"         : "ed",
      "viscosity"  : "eta",
      "nrc"        : "nrc", 
      "primordial" : "prm",
      "proterozoic": "prot", 
      "stress"     : "str",
      "temperature": "t",
      "tracer"     : "tra",
      "vorticity"  : "vor",
      "velocity"   : "vp",
      "pressure"   : "vp", # pressure is stored in the 4th component of the velocity file
    }

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

    Absolute path to the base directory containing the model directory.

    :type: str
    :canonical: stagpyviz.IOutils.basedir

  .. py:attribute:: model_dir

    Absolute path to the model directory

    :type: str
    :canonical: stagpyviz.IOutils.model_dir

  .. py:attribute:: output_fields

    List of fields that can be retrieved from binary files output by StagYY.

    :type: list[str]
    :canonical: stagpyviz.IOutils.output_fields

  .. py:attribute:: pvd

    Name of the output pvd file.

    :type: str
    :canonical: stagpyviz.IOutils.pvd

  .. py:attribute:: output_dir
  
    Absolute path to the output directory.

    :type: str
    :canonical: stagpyviz.IOutils.output_dir

  .. py:attribute:: is_surface

    Boolean that can be used to indicate whether to output the surface (True) or the volume (False).

    :type: bool
    :canonical: stagpyviz.IOutils.is_surface

  .. py:attribute:: step

    Current step number.

    :type: int|None
    :canonical: stagpyviz.IOutils.step

  .. py:attribute:: step_start

    If a range of steps is specified, the step number to start from.

    :type: int
    :canonical: stagpyviz.IOutils.step_start
    
  .. py:attribute:: step_end

    If a range of steps is specified, the step number to end at.

    :type: int|None
    :canonical: stagpyviz.IOutils.step_end

  .. py:attribute:: steps_idx

    If a range of steps is specified, the array of step numbers to process.

    :type: np.ndarray|None
    :canonical: stagpyviz.IOutils.steps_idx

  .. py:attribute:: prefix

    Prefix that can be used to name files

    :type: str
    :canonical: stagpyviz.IOutils.prefix

  .. py:attribute:: time

    Can be used to store the time of the model.

    :type: float
    :canonical: stagpyviz.IOutils.time

  .. py:attribute:: regions

    List of regions names corresponding to the outputed regions from StagYY.

    :type: list[str]
    :canonical: stagpyviz.IOutils.regions

  .. py:attribute:: filelist

    Directory in which keys are the name of the fields and values are the corresponding names
    of StagYY output files.

    :type: dict[str,str]
    :canonical: stagpyviz.IOutils.filelist
      
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

    self.filelist = {
      "composition": "c",
      "divergence": "div",
      "e2" : "ed",
      "nrc": "nrc", # ??
      "pressure": "vp", # pressure is stored in the 4th component of the velocity file
      "primordial": "prm",
      "proterozoic": "prot", # ??
      "stress" : "str",
      "temperature": "t",
      "tracer": "tra",
      "velocity": "vp",
      "viscosity": "eta",
      "vorticity": "vor"
    }
    return
  
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
