"""solves the time-dependent backward-euler spatial transient.
responds to rod movements in the reactor and how they modify xs"""

from spatial import Sptl
import numpy as np
import scipy.sparse as ss
from scipy.sparse.linalg import splu

class Transient():
    def __init__(self,spatial, v= np.array([1.4e9, 2.2e5])):
        self.spatial=spatial
        self.v=v

        self.invv=ss.diags(np.repeat(1/np.asarray(v),len(spatial.active)*spatial.G)).tocsc()

    def prep(self,dt):
        self.dt=dt
        A=self.invv/dt+self.spatial.L-self.spatial.F
        self.lu=splu(A.tocsc())

    def step(self):
        rhs=(self.invv/self.dt)@self.phi
        self.phi=self.lu.solve(rhs)
        return self.phi
    