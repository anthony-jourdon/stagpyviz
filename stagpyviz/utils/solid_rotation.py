import os
import numpy as np
from ..mesh import YinYangMesh
from scipy.spatial.transform import Rotation

class SolidBodyRotation:
  """
  Class to compute and manage the solid body rotation of a 
  3D domain represented by a Yin-Yang mesh.
  This class provides methods to compute the angular momentum, 
  angular inertia, angular velocity, and solid body velocity of the domain. 
  It also allows for the removal of the solid body velocity from the velocity field, 
  and for the computation of rotation angles and axes.

  Can be applied to a timeseries to track the evolution of the solid body rotation over time.

  :param YinYangMesh mesh: 
    The Yin-Yang mesh representing the 3D domain.
    Optional. 
    If not provided, the class can still be used by loading the mesh later, but 
    the mesh must be set before calling methods that require it i.e., 
    :py:meth:`cell_angular_momentum <stagpyviz.SolidBodyRotation.cell_angular_momentum>`, 
    :py:meth:`cell_angular_inertia <stagpyviz.SolidBodyRotation.cell_angular_inertia>`,
    :py:meth:`solid_body_velocity <stagpyviz.SolidBodyRotation.solid_body_velocity>`, and 
    :py:meth:`remove_solid_body_velocity <stagpyviz.SolidBodyRotation.remove_solid_body_velocity>`.
  :param str rule:
    Integration rule to use for computing the angular momentum and inertia.
    Default is ``"1pt"``. 
    Other option is ``"3x2pt"``.
  """
  def __init__(self, mesh:YinYangMesh|None=None, rule:str="1pt"):
    self.mesh = mesh
    self.rule = rule

    self._cell_L = None
    self._cell_I = None
    self._L_vol  = None
    self._I_vol  = None
    self._omega  = None
    self._axis   = None
    self._angle  = None
    self._q_dt   = None

    self._quaternion = np.array([1, 0, 0, 0], dtype=np.float64) # identity quaternion - initial condition
    return

  @property
  def cell_L(self) -> np.ndarray:
    """
    Cell-wise angular momentum of the domain.
    Array of shape ``(number_of_cells, 3)``.
    """
    if self._cell_L is None:
      self._cell_L = self.cell_angular_momentum()
    return self._cell_L

  @property
  def cell_I(self) -> np.ndarray:
    """
    Cell-wise angular inertia of the domain.
    Array of shape ``(number_of_cells, 3, 3)``.
    """
    if self._cell_I is None:
      self._cell_I = self.cell_angular_inertia()
    return self._cell_I

  @property
  def angular_momentum(self) -> np.ndarray:
    """
    Angular momentum of the domain.
    Array of shape ``(3,)``.
    """
    if self._L_vol is None:
      self._L_vol = np.sum(self.cell_L, axis=0)
    return self._L_vol

  @property
  def angular_inertia(self) -> np.ndarray:
    """
    Angular inertia of the domain.
    Array of shape ``(3, 3)``.
    """
    if self._I_vol is None:
      self._I_vol = np.sum(self.cell_I, axis=0)
    return self._I_vol

  @property
  def angular_velocity(self) -> np.ndarray:
    """
    Angular velocity of the domain defined as:


    .. math::
      \\boldsymbol{\\omega} = \\boldsymbol{I}^{-1} \\mathbf{L}


    where :math:`\\boldsymbol{I}` is the angular inertia 
    and :math:`\\mathbf{L}` is the angular momentum.
    Array of shape ``(3,)``.
    """
    if self._omega is None:
      L_vol = self.angular_momentum
      I_vol = self.angular_inertia
      self._omega = np.linalg.solve(I_vol, L_vol)
    return self._omega

  @angular_velocity.setter
  def angular_velocity(self, omega:np.ndarray):
    self._omega = omega
    return

  @property
  def quaternion_dt(self) -> np.ndarray:
    """
    Quaternion representing the instantaneous rotation of the domain over a time step ``dt``.
    Array of shape ``(4,)``.
    """
    if self._q_dt is None:
      raise ValueError("Instantaneous quaternion has not been computed yet. Call instantaneous_quaternion(angle, axis) first.")
    return self._q_dt

  @quaternion_dt.setter
  def quaternion_dt(self, q:np.ndarray):
    self._q_dt = q
    return

  @property
  def quaternion(self) -> np.ndarray:
    """
    Quaternion representing the accumulated rotation of the domain.
    Array of shape ``(4,)``.
    """
    return self._quaternion

  @quaternion.setter
  def quaternion(self, q:np.ndarray):
    self._quaternion = q
    return

  def cell_angular_momentum(self) -> np.ndarray:
    """
    Compute the cell-wise angular momentum of the domain as:

    .. math::
      \\mathbf{L}_e = \\int_{\\Omega_e} \\rho (\\mathbf{x}) \\left( \\mathbf{x} \\times \\mathbf{u}  \\right) \\, d{V_e}

    where :math:`\\Omega_e` is the element, 
    :math:`\\rho` is the density field, 
    :math:`\\mathbf{x}` is the position vector, 
    and :math:`\\mathbf{u}` is the velocity vector.

    :return: 
      Cell-wise angular momentum of the domain.
      Array of shape ``(number_of_cells, 3)``.
    :rtype: numpy.ndarray

    .. note::
      The ``"density"`` and ``"velocity"`` fields must be present in the mesh's point data (``mesh.point_data``).

    """
    mesh:YinYangMesh = self.mesh
    if mesh is None:
      raise ValueError("mesh is None. Cannot compute angular momentum.")
    # compute the angular momentum:
    # L = \int_V rho r x u dV
    rho = mesh.point_data["density"]
    u   = mesh.point_data["velocity"]
    r   = mesh.points
    # compute the pointwise integrand: rho * r x u
    w = np.cross(r, u, axis=1)
    integrand = np.einsum("e,ei->ei", rho, w)
    # integrate over the volume of the mesh
    int_L = mesh.integrate_over_cell(integrand, rule=self.rule)
    return int_L

  def cell_angular_inertia(self) -> np.ndarray:
    """
    Compute the cell-wise angular inertia of the domain as:

    .. math::
      \\boldsymbol{I}_e = \\int_{\\Omega_e} \\rho (\\mathbf{x}) \\left( ||\\mathbf{x}||^2 \\, \\mathbb{I} - \\mathbf{x} \\mathbf{x}^T \\right) \\, d{V_e}

    where :math:`\\Omega_e` is the element, 
    :math:`\\rho` is the density field, 
    :math:`\\mathbf{x}` is the position vector, 
    and :math:`\\mathbb{I}` is the identity matrix.

    :return: 
      Cell-wise angular inertia of the domain.
      Array of shape ``(number_of_cells, 3, 3)``.
    :rtype: numpy.ndarray

    .. note::
      The ``"density"`` field must be present in the mesh's point data (``mesh.point_data``).

    """
    mesh:YinYangMesh = self.mesh
    if mesh is None:
      raise ValueError("mesh is None. Cannot compute angular inertia.")
    # compute the angular inertia:
    # I = \int_V rho (r^2 I - r r^T) dV
    rho = mesh.point_data["density"]
    r   = mesh.points
    # compute the pointwise integrand: rho * (r^2 I - r r^T)
    r2rrT = np.zeros((mesh.number_of_points, 3, 3))
    r2rrT[:,0,0] = r[:,1]**2 + r[:,2]**2
    r2rrT[:,1,1] = r[:,0]**2 + r[:,2]**2
    r2rrT[:,2,2] = r[:,0]**2 + r[:,1]**2
    r2rrT[:,0,1] = r2rrT[:,1,0] = -r[:,0]*r[:,1]
    r2rrT[:,0,2] = r2rrT[:,2,0] = -r[:,0]*r[:,2]
    r2rrT[:,1,2] = r2rrT[:,2,1] = -r[:,1]*r[:,2]
    integrand = np.einsum("e, eij -> eij", rho, r2rrT)
    # integrate over the volume of the mesh
    int_I = mesh.integrate_over_cell(integrand, rule=self.rule)
    return int_I

  def solid_body_velocity(self) -> np.ndarray:
    """
    Compute the solid body velocity field of the domain as:

    .. math::
      \\mathbf{u}_{solid} = \\boldsymbol{\\omega} \\times \\mathbf{x}

    where :math:`\\boldsymbol{\\omega}` is the angular velocity
    and :math:`\\mathbf{x}` is the position vector.

    :return:
      Solid body velocity field of the domain.
      Array of shape ``(number_of_points, 3)``.
    :rtype: numpy.ndarray
    """
    mesh:YinYangMesh = self.mesh
    if mesh is None:
      raise ValueError("mesh is None. Cannot compute solid body velocity.")
    omega   = self.angular_velocity
    r       = mesh.points
    u_solid = np.cross(omega, r, axisa=0, axisb=1)
    return u_solid

  def remove_solid_body_velocity(self) -> np.ndarray:
    """
    Remove the solid body velocity from the velocity field of the domain as:

    .. math::
      \\mathbf{u}' = \\mathbf{u} - \\mathbf{u}_{solid}

    where :math:`\\mathbf{u}` is the original velocity field
    and :math:`\\mathbf{u}_{solid}` is the solid body velocity field.

    :return:
      Velocity field of the domain corrected for solid body velocity.
      Array of shape ``(number_of_points, 3)``.
    :rtype: numpy.ndarray
    """
    mesh:YinYangMesh = self.mesh
    if mesh is None:
      raise ValueError("mesh is None. Cannot remove solid body velocity.")
    u       = mesh.point_data["velocity"]
    u_solid = self.solid_body_velocity()
    u_corr  = u - u_solid
    return u_corr

  def compute_angle(self, dt:float) -> float:
    """
    Compute the rotation angle of the domain over a time step ``dt`` as:

    .. math::
      \\theta = ||\\boldsymbol{\\omega}|| \\, dt

    where :math:`\\boldsymbol{\\omega}` is the angular velocity.

    :param float dt: 
      Time step over which to compute the rotation angle.

    :return:
      Rotation angle of the domain.
      Scalar value.
    :rtype: float
    """
    omega = self.angular_velocity
    angle = np.linalg.norm(omega) * dt
    return angle

  def compute_axis(self) -> np.ndarray:
    """
    Compute the rotation axis of the domain as:

    .. math::
      \\hat{\\mathbf{k}} = \\frac{\\boldsymbol{\\omega}}{||\\boldsymbol{\\omega}||}

    where :math:`\\boldsymbol{\\omega}` is the angular velocity.

    :return:
      Rotation axis of the domain.
      Array of shape ``(3,)``.
    :rtype: numpy.ndarray
    """
    omega = self.angular_velocity
    axis  = omega / np.linalg.norm(omega)
    return axis

  def instantaneous_quaternion(self, angle:float, axis:np.ndarray) -> np.ndarray:
    """
    Construct the instantaneous quaternion representing the rotation of the domain over a time step ``dt`` as:

    .. math::
      q = \\left[ \\cos\\left(\\frac{\\theta}{2}\\right), \\quad \\hat{\\mathbf{k}} \\sin\\left(\\frac{\\theta}{2}\\right) \\right]

    where :math:`\\theta` is the rotation angle and :math:`\\hat{\\mathbf{k}}` is the rotation axis.
    
    :param float angle: Rotation angle of the domain.
    :param numpy.ndarray axis: Rotation axis of the domain.
    
    :return:
      Instantaneous quaternion representing the rotation of the domain.
      Array of shape ``(4,)``.
    :rtype: numpy.ndarray
    """
    self.quaternion_dt = np.array([np.cos(angle/2.0), *(np.sin(angle/2.0) * axis)], dtype=np.float64)
    return 

  def quaternion_multiply(self, q1:np.ndarray, q2:np.ndarray) -> np.ndarray:
    """
    Helper function to multiply two quaternions:

    .. math::
      \\mathbf q = \\mathbf q_1 \\otimes \\mathbf q_2 = 
      \\begin{bmatrix}
      w_1 w_2 - \\mathbf a \\cdot \\mathbf b \\\\
      w_1 \\mathbf b + w_2 \\mathbf a + \\mathbf a \\times \\mathbf b
      \\end{bmatrix}
    
    where :math:`\\mathbf q_1 = [w_1, \\mathbf a]` 
    and :math:`\\mathbf q_2 = [w_2, \\mathbf b]` 
    are the two quaternions.

    :param numpy.ndarray q1: First quaternion.
    :param numpy.ndarray q2: Second quaternion.
    
    :return:
      Product of the two quaternions.
      Array of shape ``(4,)``.
    :rtype: numpy.ndarray
    """
    w1 = q1[0]
    w2 = q2[0]
    a  = q1[1:]
    b  = q2[1:]
    w  = w1*w2 - np.dot(a, b)
    c  = w1*b + w2*a + np.cross(a, b)
    return np.array([w, *c], dtype=np.float64)

  def update_quaternion(self) -> np.ndarray:
    """
    Update the accumulated quaternion representing the accumulated rotation of the domain such that:
    
    .. math::
      \\mathbf q_{new} = \\mathbf q_{dt} \\otimes \\mathbf q_{old}

    where :math:`\\mathbf q_{dt}` is the instantaneous quaternion
    and :math:`\\mathbf q_{old}` is the accumulated quaternion.

    :return:
      Updated accumulated quaternion representing the accumulated rotation of the domain.
      Array of shape ``(4,)``.
    :rtype: numpy.ndarray
    """
    # update the quaternion
    print(f"Updating quaternion: {self.quaternion} with {self.quaternion_dt}")
    self.quaternion = self.quaternion_multiply(self.quaternion_dt, self.quaternion)
    return

  def rotate_points(self, points:np.ndarray, quaternion:np.ndarray) -> np.ndarray:
    """
    Rotate points in 3D space using a given quaternion.

    :param numpy.ndarray points: 
      Points to be rotated. 
      Array of shape ``(number_of_points, 3)``.
    :param numpy.ndarray quaternion: 
      Quaternion representing the rotation. 
      Array of shape ``(4,)``.

    :return:
      Rotated points. 
      Array of shape ``(number_of_points, 3)``.
    :rtype: numpy.ndarray
    """
    # Implementation for rotating points using the given quaternion
    rot = Rotation.from_quat(quaternion, scalar_first=True)
    return rot.apply(points, inverse=True)

  def save_to_file(self, fname:str, step:int|None=None, time:float|None=None, reset:bool=False) -> None:
    """
    Save the current state of the solid body rotation to a file.

    :param str fname: 
      Name of the file to save the state to.
    :param int|None step: 
      Current step number. Optional. If None, the step number will not be saved.
    :param float|None time: 
      Current time. Optional. If None, the time will not be saved.
    :param bool reset: 
      Whether to reset the file if it already exists. 
      If *True*, the file will be overwritten, if *False*, the data will be appended to the file. 
      Default is *False*.
    """
    names = []
    vals  = []
    fmts  = []
    if step is not None:
      names.append("step")
      vals.append(step)
      fmts.append("%d")
    if time is not None:
      names.append("time")
      vals.append(time)
      fmts.append("%1.6e")
    for d in range(3):
      names.append(f"angular_velocity_{d}")
      vals.append(self.angular_velocity[d])
      fmts.append("%1.6e")
    for d in range(4):
      names.append(f"quaternion_dt_{d}")
      vals.append(self.quaternion_dt[d])
      fmts.append("%1.6e")
    for d in range(4):
      names.append(f"quaternion_{d}")
      vals.append(self.quaternion[d])
      fmts.append("%1.6e")
    if not os.path.exists(fname) or reset:
      with open(fname, "w") as f:
        f.write(",".join(names) + "\n")
        f.write(",".join([fmts[i] % vals[i] for i in range(len(vals))]) + "\n")
    else:
      with open(fname, "a") as f:
        f.write(",".join([fmts[i] % vals[i] for i in range(len(vals))]) + "\n")
    return

  def load_step_from_file(self, fname:str, line_num:int|None=None) -> tuple[int|None, float|None]:
    """
    Load the state of the solid body rotation from a file.
    Directly sets the class attributes
    :py:attr:`angular_velocity <stagpyviz.SolidBodyRotation.angular_velocity>`, 
    :py:attr:`quaternion_dt <stagpyviz.SolidBodyRotation.quaternion_dt>`, and 
    :py:attr:`quaternion <stagpyviz.SolidBodyRotation.quaternion>`.

    :param str fname:
      Name of the file to load the state from.
    :param int|None line_num:
      Line number to load the state from. Optional. 
      If *None*, the last line will be loaded.
    :return:
      Tuple containing the step number and time. 
      If the step number or time is not present in the file, 
      they will be returned as None.
    :rtype: tuple[int|None, float|None]
    """
    if not os.path.exists(fname):
      raise FileNotFoundError(f"File {fname} does not exist.")
    with open(fname, "r") as f:
      header = f.readline().strip().split(",")
      if line_num is None:
        data = f.readlines()[-1].strip().split(",")
      else:
        data = f.readlines()[line_num].strip().split(",")
    data_dict = {header[i]: float(data[i]) for i in range(len(header))}
    self.angular_velocity = np.array([data_dict[f"angular_velocity_{d}"] for d in range(3)], dtype=np.float64)
    self.quaternion_dt    = np.array([data_dict[f"quaternion_dt_{d}"] for d in range(4)], dtype=np.float64)
    self.quaternion       = np.array([data_dict[f"quaternion_{d}"] for d in range(4)], dtype=np.float64)
    step = data_dict.get("step", None)
    step = int(step) if step is not None else None
    time = data_dict.get("time", None)
    time = float(time) if time is not None else None
    return step, time

  def load_data_from_file(self, fname:str) -> dict[str, np.ndarray]:
    """
    Load file containing information about the solid body rotation for one or multiple steps.

    :param str fname:
      Name of the file to load the state(s) from.
    :return:
      Dictionary containing the loaded data. 
      Keys are the field names and values are numpy arrays of the corresponding data.
      Always contains 
      ``"angular_velocity"``: shape ``(n, 3)``, 
      ``"quaternion_dt"``: shape ``(n, 4)``, and 
      ``"quaternion"``: shape ``(n, 4)``. 
      Optionally contains ``"step"``: shape ``(n,)`` and ``"time"``: shape ``(n,)`` 
      if present in the file.
    :rtype: dict[str, numpy.ndarray]
    """
    data_dict = {}
    if not os.path.exists(fname):
      raise FileNotFoundError(f"File {fname} does not exist.")
    with open(fname, "r") as f:
      header = f.readline().strip().split(",")
      lines = f.readlines()

    for field in header:
      data_dict[field] = []

    for line in lines:
      data = line.strip().split(",")
      for field in header:
        if field == "step":
          value = int(data[header.index(field)])
        else:
          value = float(data[header.index(field)])
        data_dict[field].append(value)

    result = {}
    if "step" in data_dict:
      result["step"] = np.array(data_dict["step"], dtype=np.int64)
    if "time" in data_dict:
      result["time"] = np.array(data_dict["time"], dtype=np.float64)
    result["angular_velocity"] = np.array([data_dict[f"angular_velocity_{d}"] for d in range(3)], dtype=np.float64).T
    result["quaternion_dt"]    = np.array([data_dict[f"quaternion_dt_{d}"] for d in range(4)], dtype=np.float64).T
    result["quaternion"]       = np.array([data_dict[f"quaternion_{d}"] for d in range(4)], dtype=np.float64).T

    return result