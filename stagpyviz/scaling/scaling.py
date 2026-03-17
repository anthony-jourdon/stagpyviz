try:
  from .. import units
except ImportError:
  from stagpyviz import units

from pint import Unit, Quantity
import numpy as np

class Scaling:
  """
  Class representing a scaling factor for a physical field, including its name, value and unit.
  The class provides methods to convert between dimensional and adimensional values, 
  as well as to convert between different units of the same field using the python package 
  `pint`_ for unit conversions.

  :param str name: Name of the physical field being scaled (e.g. ``"temperature"``, ``"length"``, etc.)
  :param float factor: Scaling factor for the physical field
  :param str|pint.Unit|None unit: 
    Unit of the physical field (optional), 
    see `pint`_ documentation for supported units. 
    If not provided, the unit will be set to None.
  
  :Methods:

  """
  def __init__(self, name:str, factor:float, unit:str|Unit|None=None):
    self.name:str       = name
    self.factor:float   = factor
    self.unit:Unit|None = None
    if unit is not None:
      try:
        if isinstance(unit, str):
          self.unit:Unit = units.Unit(unit)
        else:
          self.unit:Unit = unit
      except Exception as e:
        print(f"Error creating unit for field {name}: {e}")
    return

  def __str__(self) -> str:
    s  =  "Scaling instance:\n"
    s += f"  Name: {self.name}\n"
    s += f"  Factor: {self.factor}\n"
    s += f"  Unit: {self.unit}\n"
    return s

  def dim(self, field:np.ndarray) -> np.ndarray:
    """
    Scale a non-dimensional field to dimensional using the scaling factor.
    """
    return field * self.factor
  
  def a_dim(self, field:np.ndarray) -> np.ndarray:
    """
    Scale a dimensional field to non-dimensional using the scaling factor.
    """
    return field / self.factor
  
  def to(self, field:np.ndarray, unit:str|Unit) -> np.ndarray:
    """
    Convert a dimensional field to a different unit using the `pint`_ package for unit conversions.

    .. warning::
      Units must be compatible for conversion, 
      i.e. they must represent the same physical quantity 
      (e.g. ``"K"`` and ``"degC"`` are compatible, but ``"K"`` and ``"m"`` are not).
    
    """
    q:Quantity = units.Quantity(field, self.unit)
    q = q.to(unit)
    return q.magnitude
  
  def to_base(self, field:np.ndarray, unit:str|Unit) -> np.ndarray:
    """
    Convert a dimensional field from a given unit to the base unit 
    of the scaling factor using the `pint`_ package for unit conversions.

    .. warning::
      Units must be compatible for conversion, 
      i.e. they must represent the same physical quantity 
      (e.g. ``"K"`` and ``"degC"`` are compatible, but ``"K"`` and ``"m"`` are not).
    
    """
    q:Quantity = units.Quantity(field, unit)
    q = q.to(self.unit)
    return q.magnitude
  
def scaling_factors(**kwargs) -> dict[str, Scaling]:
  """
  Function to create a dictionary of Scaling instances for different physical fields,
  based on the provided keyword arguments. The function calculates the scaling factors
  for temperature, length, diffusivity, expansion, gravity, density, time, velocity,
  heat source, viscosity and pressure based on the provided values and / or the Rayleigh number (Ra).

  :param kwargs: Keyword arguments to specify the scaling factors for different fields.
    The following keyword arguments are supported (other quantities are derived from these):

    - ``Ra``: Rayleigh number (default: 1e7)
    - ``temperature_factor``: Scaling factor for temperature (default: 2700.0 K)
    - ``temperature_unit``: Unit for temperature (default: "K")
    - ``length_factor``: Scaling factor for length (default: 2.89e6 m)
    - ``length_unit``: Unit for length (default: "m")
    - ``diffusivity_factor``: Scaling factor for diffusivity (default: 1e-6 m^2/s)
    - ``diffusivity_unit``: Unit for diffusivity (default: "m**2/s")
    - ``expansion_factor``: Scaling factor for thermal expansion coefficient (default: 3e-5 1/K)
    - ``expansion_unit``: Unit for thermal expansion coefficient (default: "1/K")
    - ``gravity_factor``: Scaling factor for gravity (default: 9.81 m/s**2)
    - ``gravity_unit``: Unit for gravity (default: "m/s**2")
    - ``density_factor``: Scaling factor for density (default: 3300.0 kg/m**3)
    - ``density_unit``: Unit for density (default: "kg/m**3")
    - ``viscosity_factor``: Scaling factor for viscosity (default: calculated from Ra if not provided)
    - ``viscosity_unit``: Unit for viscosity (default: calculated from Ra if not provided)

  :return: Dictionary of Scaling instances for different physical fields.
  :rtype: dict[str, Scaling]
  """
  if "Ra" not in kwargs:
    s = "Warning: Rayleigh number (Ra) not provided.\n"
    s += "Using default value of 1e7 for viscosity scaling.\n"
    s += "To specify a different value, pass Ra as a keyword argument to scaling_factors() function.\n"
    print(s)
  
  scalings = {
    "temperature": Scaling(
      name="temperature",
      factor=float(kwargs.get("temperature_factor", 2700.0)),
      unit=kwargs.get("temperature_unit", "K")
    ),
    "length": Scaling(
      name="length",
      factor=float(kwargs.get("length_factor", 2.89e6)),
      unit=kwargs.get("length_unit", "m")
    ),
    "diffusivity": Scaling(
      name="diffusivity",
      factor=float(kwargs.get("diffusivity_factor", 1e-6)),
      unit=kwargs.get("diffusivity_unit", "m**2/s")
    ),
    "expansion": Scaling(
      name="expansion",
      factor=float(kwargs.get("expansion_factor", 3e-5)),
      unit=kwargs.get("expansion_unit", "1/K")
    ),
    "gravity": Scaling(
      name="gravity",
      factor=float(kwargs.get("gravity_factor", 9.81)),
      unit=kwargs.get("gravity_unit", "m/s**2")
    ),
    "density": Scaling(
      name="density",
      factor=float(kwargs.get("density_factor", 3300.0)),
      unit=kwargs.get("density_unit", "kg/m**3")
    )
  }
  scalings["time"] = Scaling(
    name="time",
    factor=scalings["length"].factor**2 / scalings["diffusivity"].factor,
    unit=scalings['length'].unit**2 / scalings['diffusivity'].unit
  )
  scalings["velocity"] = Scaling(
    name="velocity",
    factor=scalings["length"].factor / scalings["time"].factor,
    unit=scalings['length'].unit / scalings['time'].unit
  )
  scalings["heat_source"] = Scaling(
    name="heat_source", 
    factor=scalings["temperature"].factor / scalings["time"].factor,
    unit=scalings["temperature"].unit / scalings["time"].unit
  )
  eta_0 = (
    scalings["density"].factor * scalings["gravity"].factor * scalings["expansion"].factor *
    scalings["temperature"].factor * scalings["length"].factor**3 /
    (float(kwargs.get("Ra", 1.0e7)) * scalings["diffusivity"].factor)
  )
  eta_u = (
    scalings["density"].unit * scalings["gravity"].unit * scalings["expansion"].unit *
    scalings["temperature"].unit * scalings["length"].unit**3 / scalings["diffusivity"].unit
  )
  scalings["viscosity"] = Scaling(
    name="viscosity",
    factor=float(kwargs.get("viscosity_factor", eta_0)),
    unit=kwargs.get("viscosity_unit", eta_u)
  )
  scalings["pressure"] = Scaling(
    name="pressure",
    factor=scalings["viscosity"].factor / scalings["time"].factor,
    unit=scalings["viscosity"].unit / scalings["time"].unit
  )
  scalings["strain_rate"] = Scaling(
    name="strain_rate",
    factor=1.0 / scalings["time"].factor,
    unit=scalings["time"].unit**(-1)
  )
  return scalings

def test():
  temperature = np.linspace(0.12, 1.12, 11)
  scaling = Scaling(
    name="Temperature", 
    factor=2700.0, 
    unit="K"
  )
  print(f"Original array: {temperature}")
  t_si = scaling.dim(temperature)
  print(f"Dimensional temperature: {t_si} {scaling.unit}")
  t_dimless = scaling.a_dim(t_si)
  print(f"Adimensional temperature: {t_dimless}")
  t_deg = scaling.to(t_si, "degC")
  print(f"Temperature in degC: {t_deg} degC")

  scalings = scaling_factors()
  for name in scalings:
    print(scalings[name])

  return

if __name__ == "__main__":
  test()