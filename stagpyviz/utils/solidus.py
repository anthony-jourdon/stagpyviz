import numpy as np

try:
  from ..scaling import Scaling
  from ..mesh import ShellMesh, YinYangMesh
except ImportError:
  from stagpyviz.scaling import Scaling
  from stagpyviz.mesh import ShellMesh, YinYangMesh

class Solidus:
  """
    Class to compute the solidus temperature.

    :param dict[str, Scaling] scalings:
      A dictionary of Scaling objects for the model's fields (temperature, pressure, etc.)
    :param bool log:
      If True, print some information for debugging purposes.
    :param bool adim:
      If True, the model is expected to be adimensional.
    :param float max_pressure:
      The maximum pressure for which the solidus is defined. Default is 10 GPa,
      converted to adimensional units if `adim` is True.
    :param bool add_adiabatic:
      If True, the adiabatic temperature is added to the temperature field before determining if the temperature crosses the solidus.

  """
  def __init__(self, scalings:dict[str, Scaling], log:bool=False, adim:bool=True, **kwargs) -> None:
    self.scalings = scalings
    self.log      = log
    self.adim     = adim
    # maximum pressure for which the solidus is defined (10 GPa for Hirschmann 2010), converted to adimensional units
    max_pressure = scalings['pressure'].to_base(10.0, unit='GPa')
    if self.adim:
      max_pressure = scalings['pressure'].a_dim(max_pressure)
    self.max_pressure = kwargs.get("max_pressure", max_pressure)

    self.add_adiabatic:bool = kwargs.get("add_adiabatic", True)
    return

  def compute_solidus(self, *args, **kwargs) -> np.ndarray:
    raise NotImplementedError("compute_solidus method must be implemented in subclasses.")

  def adiabatic_temperature_from_depth(self, temperature:np.ndarray, depth:np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the adiabatic temperature from depth using the formula:

    .. math::
      T(z) = T_0 \\exp \\left( - \\frac{\\alpha}{C_p} g z \\right)

    where :math:`T_0` is the incompressible temperature at :math:`z=0`, :math:`\\alpha` is the thermal expansion coefficient,
    :math:`C_p` is the specific heat, :math:`g` is the gravity, and :math:`z` is the depth.

    .. note::
      Either :math:`g` or :math:`z` is negative, but not both.

    :param numpy.ndarray temperature: the incompressible temperature field (K).
    :param numpy.ndarray depth: the depth field (m).
    :param float alpha: the thermal expansion coefficient (1/K).
    :param float g: the gravity (m/s^2).
    :param float Cp: the specific heat (J/kg/K).

    :return numpy.ndarray: the adiabatic temperature (K).
    """
    alpha = kwargs.get("alpha", 5e-5) # thermal expansion coefficient (1/K)
    g     = kwargs.get("g", 9.81)     # gravity (m/s^2)
    Cp    = kwargs.get("Cp", 800.0)   # specific heat
    # adiabatic temperature gradient: dT/dr = -alpha * g / Cp * T
    # integrate to get T(z) = T0 * exp(-alpha * g * r / Cp)
    return temperature * np.exp(-alpha * g * depth / Cp)

  def adiabatic_temperature_from_pressure(self, temperature:np.ndarray, pressure:np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the adiabatic temperature from pressure using the formula:

    .. math::
      T(p) = T_0 \\exp \\left( \\frac{\\alpha}{\\rho C_p} p \\right)

    where :math:`T_0` is the incompressible temperature at :math:`p=0`, :math:`\\alpha` is the thermal expansion coefficient,
    :math:`\\rho` is the density, :math:`C_p` is the specific heat, and :math:`p` is the pressure.

    .. note::
      This expression assumes that the density is constant and that the pressure is hydrostatic.

    :param numpy.ndarray temperature: the incompressible temperature field (K).
    :param numpy.ndarray pressure: the pressure field (Pa).
    :param float alpha: the thermal expansion coefficient (1/K).
    :param float rho: the density (kg/m^3).
    :param float Cp: the specific heat (J/kg/K).

    :return numpy.ndarray: the adiabatic temperature (K).
    """
    alpha = kwargs.get("alpha", 5e-5) # thermal expansion coefficient (1/K)
    rho   = kwargs.get("rho", 3300.0) # density (kg/m^3)
    Cp    = kwargs.get("Cp", 800.0)   # specific heat
    return temperature * np.exp(alpha / (rho * Cp) * pressure)

  def adiabatic_temperature(self, temperature:np.ndarray, variable:np.ndarray, from_depth:bool=False, **kwargs) -> np.ndarray:
    """
    Compute the adiabatic temperature profile from either depth or pressure, depending on the value of ``from_depth``.
    If ``from_depth`` is True, the adiabatic temperature is computed using :py:meth:`adiabatic_temperature_from_depth <stagpyviz.Solidus.adiabatic_temperature_from_depth>`, 
    otherwise it is computed using :py:meth:`adiabatic_temperature_from_pressure <stagpyviz.Solidus.adiabatic_temperature_from_pressure>`.

    :param numpy.ndarray temperature: the incompressible temperature field (K).
    :param numpy.ndarray variable: the depth (m) or pressure (Pa) field, depending on the value of ``from_depth``.
    :param bool from_depth: if True, compute the adiabatic temperature from depth; if False, compute it from pressure.
    :param float alpha: the thermal expansion coefficient (1/K).
    :param float g: the gravity (m/s^2).
    :param float rho: the density (kg/m^3).
    :param float Cp: the specific heat (J/kg/K).

    :return numpy.ndarray: the adiabatic temperature (K).
    """
    params = {
      "alpha": kwargs.get("alpha", 5e-5), # thermal expansion coefficient (1/K)
      "g":     kwargs.get("g", 9.81),     # gravity (m/s^2)
      "rho":   kwargs.get("rho", 3300.0), # density (kg/m^3)
      "Cp":    kwargs.get("Cp", 800.0)    # specific heat
    }
    if from_depth:
      return self.adiabatic_temperature_from_depth(temperature, variable, **params)
    else:
      return self.adiabatic_temperature_from_pressure(temperature, variable, **params)

  def get_melting_indices(
    self, 
    pressure_dynamic:np.ndarray,
    pressure_lithostatic:np.ndarray,
    temperature:np.ndarray,
    **kwargs
  ):
    """
    Return a boolean array (0 or 1) indicating which points are above the solidus temperature.

    :param numpy.ndarray pressure_dynamic: the dynamic pressure field (Pa).
    :param numpy.ndarray pressure_lithostatic: the lithostatic pressure field (Pa).
    :param numpy.ndarray temperature: the temperature field (K).
    :param float alpha: the thermal expansion coefficient (1/K).
    :param float g: the gravity (m/s^2).
    :param float rho: the density (kg/m^3).
    :param float Cp: the specific heat (J/kg/K).

    :return numpy.ndarray: a boolean array (0 or 1) indicating which points are above the solidus temperature.
    """
    pressure = pressure_dynamic + pressure_lithostatic
    # consider only points where the pressure is below the maximum pressure for which the solidus is defined
    valid_pidx     = pressure <= self.max_pressure
    pdyn_solidus   = pressure_dynamic[ valid_pidx ]
    plitho_solidus = pressure_lithostatic[ valid_pidx ]
    T_solidus      = self.compute_solidus(pdyn_solidus, plitho_solidus, **kwargs)
    if self.add_adiabatic:
      if self.adim:
        p = self.scalings["pressure"].dim(pressure_lithostatic)
        t = self.scalings["temperature"].dim(temperature)
        temperature = self.adiabatic_temperature(t, p, from_depth=False, **kwargs)
        temperature = self.scalings["temperature"].a_dim(temperature)
      else:
        temperature = self.adiabatic_temperature(temperature, pressure_lithostatic, from_depth=False, **kwargs)
    
    # get the points where the temperature is above the solidus temperature
    melting_idx = temperature[valid_pidx] > T_solidus
    # create a point field to store the melting index (1 if melting, 0 if not)
    melting = np.zeros_like(temperature, dtype=int)
    melting[valid_pidx] = melting_idx.astype(int)

    return melting

  def add_to_shell_mesh(
      self, 
      mesh:YinYangMesh, 
      surface_mesh:ShellMesh, 
      melting_idx:np.ndarray, 
      radial_levels:list[int], 
      radius:float,
      depth_name:str="solidus_depth",
      vr_name:str="solidus_vr"
    ) -> None:
    """
    Add the depth and vertical velocity of the deepest melting points to the surface mesh.

    :param YinYangMesh mesh: the full 3D mesh of the model.
    :param ShellMesh surface_mesh: the surface mesh of the model.
    :param numpy.ndarray melting_idx: a boolean array (0 or 1) indicating which points are above the solidus temperature.
    :param list[int] radial_levels: the radial levels of the mesh to iterate over (from top to bottom).
    :param float radius: the radius of the model.
    :param str depth_name: 
      the name of the point field to store the depth of the deepest melting points in the surface mesh. 
      Default is `"solidus_depth"`.
    :param str vr_name: 
      the name of the point field to store the vertical velocity of the deepest melting points in the surface mesh.
      Default is `"solidus_vr"`.
    """
    # extract vertical velocity and depth for the points where melting occurs
    surface_mesh[depth_name] = np.full(surface_mesh.number_of_points, np.nan)
    surface_mesh[vr_name]    = surface_mesh.create_point_field()
    # iterate over the radial levels of the mesh and fill the surface mesh fields 
    # with the depth and vertical velocity where melting occurs, 
    # going from top to bottom to get the deepest melting points
    for k in range(radial_levels[1],radial_levels[0],-1):
      ridx = mesh.get_radial_indices(k)
      good = melting_idx[ridx] > 0.1
      surface_mesh[depth_name][good] = mesh.points_spherical[ridx][good,0] - radius
      if "velocity_r" in mesh.point_data:
        surface_mesh[vr_name][good]  = mesh["velocity_r"][ridx,0][good]
    return

  def compute_solidus_YYmesh_adim(
      self, 
      mesh:YinYangMesh, 
      surface_mesh:ShellMesh, 
      radial_levels:list[int], 
      radius:float,
      **kwargs
    ) -> None:
    """
    Compute the solidus and register the depth and vertical velocity of the deepest melting points in the surface mesh.
    Function specifically designed for Yin-Yang meshes and adimensional models.
    For other types of meshes or dimensional models, use the lower level methods.

    :param YinYangMesh mesh: the full 3D mesh of the model.
    :param ShellMesh surface_mesh: the surface mesh of the model.
    :param list[int] radial_levels: the radial levels of the mesh to iterate over (from top to bottom).
    :param float radius: the radius of the model.
    :param float rho: the density of the model.
    :param float g: the gravitational acceleration of the model.
    :param float alpha: the thermal expansion coefficient of the model.
    :param float Cp: the specific heat of the model.
    :param str depth_name:
      the name of the point field to store the depth of the deepest melting points in the surface mesh. 
      Default is ``"solidus_depth"``.
    :param str vr_name:
      the name of the point field to store the vertical velocity of the deepest melting points in the surface mesh. 
      Default is ``"solidus_vr"``.

    """
    # StagYY manual to convert depth to pressure:
    # depth2pressure = rho * g * L**3 / (eta * kappa)
    scalings = self.scalings
    depth2pressure = (
      scalings["density"].factor * scalings["gravity"].factor * scalings["length"].factor**3 / 
      (scalings["viscosity"].factor * scalings["diffusivity"].factor)
    )
    # depth
    depth = mesh.points_spherical[:,0] - radius
    # approx. lithostatic pressure
    p_litho = -depth * depth2pressure

    # compute the dynamic pressure
    int_P = mesh.integrate_over_cell(mesh["pressure"])
    P_mean = np.sum(int_P) / np.sum(mesh.cells_volume)
    mesh["pressure"] -= P_mean

    kwargs["rho"]   = kwargs.get("rho",   scalings["density"].factor)
    kwargs["g"]     = kwargs.get("g",     scalings["gravity"].factor)
    kwargs["alpha"] = kwargs.get("alpha", scalings["expansion"].factor)
    melting_idx = self.get_melting_indices(mesh["pressure"], p_litho, mesh["temperature"])
    self.add_to_shell_mesh(
      mesh, 
      surface_mesh, 
      melting_idx, 
      radial_levels, 
      radius,
      depth_name=kwargs.pop("depth_name", "solidus_depth"),
      vr_name=kwargs.pop("vr_name", "solidus_vr")
    )
    return

class SolidusHirschmann2000(Solidus):
  """
  Class to compute the solidus temperature using the Hirschmann (2000) formula for peridotite.

  .. note::
    **Reference**:

    Hirschmann (2000), Mantle solidus: Experimental constraints and the effects of peridotite composition, 
    `Geochemistry, Geophysics, Geosystems`, 1, 2000GC000070.

  :param dict[str, Scaling] scalings:
    A dictionary of Scaling objects for the model's fields (temperature, pressure, etc.)
  :param bool log:
    If True, print some information for debugging purposes.
  :param bool adim:
    If True, the model is expected to be adimensional.
  :param float max_pressure:
    The maximum pressure for which the solidus is defined. Default is 10 GPa,
    converted to adimensional units if `adim` is True.
  :param bool add_adiabatic:
    If True, the adiabatic temperature is added to the temperature field before determining if the temperature crosses the solidus.
  """
  def __init__(self, scalings:dict[str, Scaling], log:bool=False, adim:bool=True, **kwargs) -> None:
    super().__init__(scalings, log=log, adim=adim, **kwargs)
    return
  
  def Hirschmann2000_coefficients(self) -> tuple[float, float, float]:
    """
    Scale coefficients for the Hirschmann (2000) solidus of peridotite 
    to S.I. units and if necessary adimensionalize them.

    :param dict[str, Scaling] scalings: 
      A dictionary of Scaling objects for the model's fields (temperature, pressure, etc.)

    :return tuple[float, float, float]:
      The coefficients a, b, c for the Hirschmann (2000) solidus of peridotite, 
      scaled to model's base units and adimensionalized if necessary.
    """
    hirschmann_scalings = {
      "a": Scaling(
        name="a",
        factor=self.scalings["temperature"].factor / self.scalings["pressure"].factor**2,
        unit=self.scalings["temperature"].unit / self.scalings["pressure"].unit**2
      ),
      "b": Scaling(
        name="b",
        factor=self.scalings["temperature"].factor / self.scalings["pressure"].factor,
        unit=self.scalings["temperature"].unit / self.scalings["pressure"].unit
      ),
      "c": Scaling(
        name="c",
        factor=self.scalings["temperature"].factor,
        unit=self.scalings["temperature"].unit
      )
    }
    a = hirschmann_scalings["a"].to_base(-5.104, unit="degC/GPa^2")
    b = hirschmann_scalings["b"].to_base(132.899, unit="degC/GPa")
    c = hirschmann_scalings["c"].to_base(1120.661, unit="degC")
    if self.log:
      s = f"Hirschmann coefficients in base units:\n"
      s += f"a = {a:.6e} {hirschmann_scalings['a'].unit},\n"
      s += f"b = {b:.6e} {hirschmann_scalings['b'].unit},\n"
      s += f"c = {c:.6e} {hirschmann_scalings['c'].unit}"
      print(s)
    # If the models is adimensional, adimensionalize the coefficients 
    if self.adim:
      a = hirschmann_scalings["a"].a_dim(a)
      b = hirschmann_scalings["b"].a_dim(b)
      c = hirschmann_scalings["c"].a_dim(c)
    return a, b, c

  def compute_solidus(
    self,
    pressure_dynamic:np.ndarray,
    pressure_lithostatic:np.ndarray,
    **kwargs
  ) -> np.ndarray:
    """
    Compute the solidus temperature using the Hirschmann (2000) formula for peridotite:

    .. math::
      T_{solidus} = a P^2 + b P + c

    where :math:`P` is the pressure and :math:`a`, :math:`b`, :math:`c` are the coefficients from Hirschmann (2000).

    .. warning::
      The solidus temperature is only valid for pressures below 10 GPa according to Hirschmann (2000).

    :param numpy.ndarray pressure_dynamic: the dynamic pressure field (Pa).
    :param numpy.ndarray pressure_lithostatic: the lithostatic pressure field (Pa).
    :param bool add_adiabatic: if True, subtract the adiabatic temperature from the solidus temperature.
    :param bool adim: if True, the pressures are adimensional and the solidus temperature will be returned in adimensional units.
    :param float alpha: the thermal expansion coefficient (1/K).
    :param float g: the gravity (m/s^2).
    :param float rho: the density (kg/m^3).
    :param float Cp: the specific heat (J/kg/K).

    :return numpy.ndarray: the solidus temperature field (K).  
    """
    a, b, c   = self.Hirschmann2000_coefficients()
    pressure  = pressure_dynamic + pressure_lithostatic
    T_solidus = a * pressure**2 + b * pressure + c
    return T_solidus