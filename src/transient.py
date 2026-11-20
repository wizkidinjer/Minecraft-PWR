"""solves the time-dependent backward-euler spatial transient.
responds to rod movements in the reactor and how they modify xs"""

from spatial import Sptl, s
import numpy as np
import scipy.sparse as ss
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt


class Transient:
    def __init__(self, spatial, dt, v=np.array([1.4e9, 2.2e5])):
        self.spatial = spatial
        self.dt = dt
        self.v = v

        self.invv = ss.diags(
            s.V * np.repeat(1 / np.asarray(v), len(spatial.active))
        ).tocsc()

    def prep(self):
        A = self.invv / self.dt + self.spatial.L - self.spatial.F
        self.lu = splu(A.tocsc())

    def step(self):
        rhs = (self.invv / self.dt) @ self.phi
        self.phi = self.lu.solve(rhs)
        return self.phi

    def run(self, howlong, phi0):
        self.phi = phi0.copy()
        self.prep()

        times = np.arange(0, howlong, self.dt)
        hist = np.empty((len(times), len(phi0)))
        hist[0] = self.phi
        for i in range(1, len(times)):
            hist[i] = self.step()

        return times, hist

    




    import matplotlib.pyplot as plt


T = Transient(s, 1e-5)


times, hist = T.run(0.01, phi0=s.phi)

N = len(s.active)

fast = hist[:, :N].sum(axis=1)
thermal = hist[:, N:].sum(axis=1)

plt.plot(times, fast / fast[0], label="Fast")
plt.plot(times, thermal / thermal[0], label="Thermal")

plt.xlabel("Time (s)")
plt.ylabel("Relative neutron flux")
plt.legend()
plt.grid()
plt.show()
