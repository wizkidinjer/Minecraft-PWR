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
        # sparse diag matrix of 1/velocity. only 2 velocities

    def prep(self):
        self.c = self.invv.diagonal() / self.dt
        A = (ss.diags(self.c) + self.spatial.L - self.spatial.F).tocsc()
        self.lu = splu(A)

    def step(self):
        self.phi = self.lu.solve(self.c * self.phi)
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

    def evDecomp(self, times, phi0):
        from scipy.sparse.linalg import eigs

        rhs = self.spatial.F - self.spatial.L
        A = (ss.diags(1 / self.invv.diagonal()) @ rhs) # = M^-1 (F-L) phi
        # is fine to compute because invv is only diagonal, so just multiplies
        lam, ev = eigs(A,6,which='LR')

        initialEig = np.linalg.lstsq(ev, phi0,rcond=None)[0]
        hist = np.array([ev @ (np.exp(lam * t) * initialEig) for t in times])
        return hist
        # evdecomp gives phi= Ve^lam*t V^-1 phio
        




    




T = Transient(s, 1e-5)
print('T load done')

times, hist = T.run(0.01, phi0=s.phi)
print('BE run done')
xact=T.evDecomp(times,s.phi)
N = len(s.active)

numerical = hist.sum(axis=1)
analytical=xact.sum(axis=1)

error = np.linalg.norm(hist - xact, axis=1) / np.linalg.norm(s.phi)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(times, numerical / numerical[0], label="Backward Euler")
ax[0].plot(times, analytical / analytical[0], "--", label="Analytical")

ax[1].plot(times, error)

ax[0].set(xlabel="Time (s)", ylabel="Relative flux")
ax[1].set(xlabel="Time (s)", ylabel="Relative L2 error")

for a in ax:
    a.grid()
    a.legend() if a is ax[0] else None

plt.tight_layout()
plt.show()




# plt.plot(times, tFlux / tFlux[0])
# plt.xlabel("Time (s)")
# plt.ylabel("Relative neutron flux")
# plt.title(f"Backward Euler with k = {s.k:.5f} derived from spatial solver")
# # plt.legend()
# plt.grid()
# plt.show()
