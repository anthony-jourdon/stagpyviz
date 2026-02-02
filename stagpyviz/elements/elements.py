import numpy as np

class Element:
  def __init__(self):
    self.dim = None
    self.basis_per_el = None
    return
  
  def evaluate_Ni(self, xi:np.ndarray):
    raise NotImplementedError("evaluate_Ni method defined in subclasses.")
  
  def Ni_centroid(self):
    raise NotImplementedError("Ni_centroid method defined in subclasses.")
  
  def evaluate_GNi(self, xi:np.ndarray):
    raise NotImplementedError("evaluate_GNi method defined in subclasses.")
  
  def GNi_centroid(self):
    raise NotImplementedError("GNi_centroid method defined in subclasses.")
  
  def evaluate_invJ(self, xi:np.ndarray, xe:np.ndarray):
    raise NotImplementedError("evaluate_invJ method defined in subclasses.")
  
  def evaluate_Jacobian(self, GNi:np.ndarray, xe:np.ndarray):
    if xe.ndim == 2:
      J = np.matmul(xe.T, GNi)
    elif xe.ndim == 3:
      # Compute Jacobian for all elements: J = xe^T @ GNi
      # elcoor is (n_cells, nnodes, dim), GNi is (nnodes, dim)
      # We need: J[e,i,j] = sum_k GNi[k,j] * elcoor[e,k,i]
      # Result: (n_cells, dim, dim)
      J = np.einsum('kj,eki->eij', GNi, xe)
    else:
      raise ValueError("xe must be 2D or 3D array.")
    return J
  
  def evaluate_dNidx(self, invJ:np.ndarray, GNi:np.ndarray) -> np.ndarray:
    if invJ.ndim == 2:
      dNidx = np.matmul(invJ.T, GNi)
    elif invJ.ndim == 3:
      # Compute global derivatives for all elements
      # GNi is (nnodes, dim), invJ is (n_cells, dim, dim)
      # We need: dNidx[e,k,i] = sum_j invJ[e,j,i] * GNi[k,j] (note the invJ transpose)
      # Result: (n_cells, nnodes, dim)
      dNidx = np.einsum('eji,kj->eki', invJ, GNi)
    return dNidx

class Element2D(Element):
  def __init__(self):
    super().__init__()
    self.dim = 2
    return

  def evaluate_detJ(self, J:np.ndarray) -> float|np.ndarray:
    if J.ndim == 2:
      detJ:float = J[0,0]*J[1,1] - J[0,1]*J[1,0]
    elif J.ndim == 3:
      detJ:np.ndarray = J[:,0,0]*J[:,1,1] - J[:,0,1]*J[:,1,0]
    else:
      raise ValueError("J must be 2D or 3D array.")
    return detJ
  
  def evaluate_invJ(self, J:np.ndarray, detJ:float|np.ndarray) -> np.ndarray:
    if J.ndim == 2:
      invJ = np.zeros((2,2),dtype=np.float64)
      invJ[0,0] =  J[1,1]
      invJ[0,1] = -J[0,1]
      invJ[1,0] = -J[1,0]
      invJ[1,1] =  J[0,0]
      invJ /= detJ
    elif J.ndim == 3:
      invJ = np.zeros_like(J)
      invJ[:, 0, 0] =  J[:, 1, 1]
      invJ[:, 0, 1] = -J[:, 0, 1]
      invJ[:, 1, 0] = -J[:, 1, 0]
      invJ[:, 1, 1] =  J[:, 0, 0]
      invJ /= detJ[:, None, None]
    else:
      raise ValueError("J must be 2D or 3D array.")
    return invJ
  
class Element3D(Element):
  def __init__(self):
    super().__init__()
    self.dim = 3
    return
  
  def evaluate_detJ(self, J:np.ndarray) -> float|np.ndarray:
    if J.ndim == 2:
      detJ:float = (
        J[0,0]*(J[1,1]*J[2,2]-J[1,2]*J[2,1]) -
        J[0,1]*(J[1,0]*J[2,2]-J[1,2]*J[2,0]) +
        J[0,2]*(J[1,0]*J[2,1]-J[1,1]*J[2,0])
      )
    elif J.ndim == 3:
      detJ:np.ndarray = (
        J[:,0,0]*(J[:,1,1]*J[:,2,2]-J[:,1,2]*J[:,2,1]) -
        J[:,0,1]*(J[:,1,0]*J[:,2,2]-J[:,1,2]*J[:,2,0]) +
        J[:,0,2]*(J[:,1,0]*J[:,2,1]-J[:,1,1]*J[:,2,0])
      )
    else:
      raise ValueError("J must be 2D or 3D array.")
    return detJ
  
  def evaluate_invJ(self, J:np.ndarray, detJ:float|np.ndarray) -> np.ndarray:
    if J.ndim == 2:
      invJ = np.zeros((3,3),dtype=np.float64)
      invJ[0,0] =  (J[1,1]*J[2,2]-J[1,2]*J[2,1])
      invJ[0,1] = -(J[0,1]*J[2,2]-J[0,2]*J[2,1])
      invJ[0,2] =  (J[0,1]*J[1,2]-J[0,2]*J[1,1])
      invJ[1,0] = -(J[1,0]*J[2,2]-J[1,2]*J[2,0])
      invJ[1,1] =  (J[0,0]*J[2,2]-J[0,2]*J[2,0])
      invJ[1,2] = -(J[0,0]*J[1,2]-J[0,2]*J[1,0])
      invJ[2,0] =  (J[1,0]*J[2,1]-J[1,1]*J[2,0])
      invJ[2,1] = -(J[0,0]*J[2,1]-J[0,1]*J[2,0])
      invJ[2,2] =  (J[0,0]*J[1,1]-J[0,1]*J[1,0])
      invJ /= detJ
    elif J.ndim == 3:
      invJ = np.zeros_like(J)
      invJ[:,0,0] =  (J[:,1,1]*J[:,2,2]-J[:,1,2]*J[:,2,1])
      invJ[:,0,1] = -(J[:,0,1]*J[:,2,2]-J[:,0,2]*J[:,2,1])
      invJ[:,0,2] =  (J[:,0,1]*J[:,1,2]-J[:,0,2]*J[:,1,1])
      invJ[:,1,0] = -(J[:,1,0]*J[:,2,2]-J[:,1,2]*J[:,2,0])
      invJ[:,1,1] =  (J[:,0,0]*J[:,2,2]-J[:,0,2]*J[:,2,0])
      invJ[:,1,2] = -(J[:,0,0]*J[:,1,2]-J[:,0,2]*J[:,1,0])
      invJ[:,2,0] =  (J[:,1,0]*J[:,2,1]-J[:,1,1]*J[:,2,0])
      invJ[:,2,1] = -(J[:,0,0]*J[:,2,1]-J[:,0,1]*J[:,2,0])
      invJ[:,2,2] =  (J[:,0,0]*J[:,1,1]-J[:,0,1]*J[:,1,0])
      invJ /= detJ[:, None, None]
    else:
      raise ValueError("J must be 2D or 3D array.")
    return invJ