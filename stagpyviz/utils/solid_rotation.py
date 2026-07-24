import os
import numpy as np
from ..mesh import YinYangMesh

class SolidBodyRotation:
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

  def cell_angular_momentum(self) -> np.ndarray:
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

  @property
  def cell_L(self) -> np.ndarray:
    if self._cell_L is None:
      self._cell_L = self.cell_angular_momentum()
    return self._cell_L

  @property
  def cell_I(self) -> np.ndarray:
    if self._cell_I is None:
      self._cell_I = self.cell_angular_inertia()
    return self._cell_I

  @property
  def angular_momentum(self) -> np.ndarray:
    if self._L_vol is None:
      self._L_vol = np.sum(self.cell_L, axis=0)
    return self._L_vol

  @property
  def angular_inertia(self) -> np.ndarray:
    if self._I_vol is None:
      self._I_vol = np.sum(self.cell_I, axis=0)
    return self._I_vol

  @property
  def angular_velocity(self) -> np.ndarray:
    if self._omega is None:
      L_vol = self.angular_momentum
      I_vol = self.angular_inertia
      self._omega = np.linalg.solve(I_vol, L_vol)
    return self._omega

  @angular_velocity.setter
  def angular_velocity(self, omega:np.ndarray):
    self._omega = omega
    return

  def solid_body_velocity(self) -> np.ndarray:
    mesh:YinYangMesh = self.mesh
    if mesh is None:
      raise ValueError("mesh is None. Cannot compute solid body velocity.")
    omega   = self.angular_velocity
    r       = mesh.points
    u_solid = np.cross(omega, r, axisa=0, axisb=1)
    return u_solid

  def remove_solid_body_velocity(self) -> np.ndarray:
    mesh:YinYangMesh = self.mesh
    if mesh is None:
      raise ValueError("mesh is None. Cannot remove solid body velocity.")
    u       = mesh.point_data["velocity"]
    u_solid = self.solid_body_velocity()
    u_corr  = u - u_solid
    return u_corr

  def compute_angle(self, dt:float) -> float:
    omega = self.angular_velocity
    angle = np.linalg.norm(omega) * dt
    return angle

  def compute_axis(self) -> np.ndarray:
    omega = self.angular_velocity
    axis  = omega / np.linalg.norm(omega)
    return axis

  @property
  def quaternion_dt(self) -> np.ndarray:
    if self._q_dt is None:
      raise ValueError("Instantaneous quaternion has not been computed yet. Call instantaneous_quaternion(angle, axis) first.")
    return self._q_dt

  @quaternion_dt.setter
  def quaternion_dt(self, q:np.ndarray):
    self._q_dt = q
    return

  def instantaneous_quaternion(self, angle:float, axis:np.ndarray) -> np.ndarray:
    self.quaternion_dt = np.array([np.cos(angle/2.0), *(np.sin(angle/2.0) * axis)], dtype=np.float64)
    return 

  def quaternion_multiply(self, q1:np.ndarray, q2:np.ndarray) -> np.ndarray:
    w1 = q1[0]
    w2 = q2[0]
    a  = q1[1:]
    b  = q2[1:]
    w  = w1*w2 - np.dot(a, b)
    c  = w1*b + w2*a + np.cross(a, b)
    return np.array([w, *c], dtype=np.float64)

  @property
  def quaternion(self) -> np.ndarray:
    return self._quaternion

  @quaternion.setter
  def quaternion(self, q:np.ndarray):
    self._quaternion = q
    return

  def update_quaternion(self) -> np.ndarray:
    # update the quaternion
    print(f"Updating quaternion: {self.quaternion} with {self.quaternion_dt}")
    self.quaternion = self.quaternion_multiply(self.quaternion_dt, self.quaternion)
    return

  def save_to_file(self, fname:str, step:int|None=None, time:float|None=None) -> None:
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
    if not os.path.exists(fname):
      with open(fname, "w") as f:
        f.write(",".join(names) + "\n")
        f.write(",".join([fmts[i] % vals[i] for i in range(len(vals))]) + "\n")
    else:
      with open(fname, "a") as f:
        f.write(",".join([fmts[i] % vals[i] for i in range(len(vals))]) + "\n")
    return

  def load_from_file(self, fname:str, line_num:int|None=None) -> tuple[int|None, float|None]:
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