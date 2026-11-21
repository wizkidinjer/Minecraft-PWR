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

    def evDecomp(self, times, phi0, k=6):
        from scipy.sparse.linalg import eigs
        lam, ev = eigs(self.spatial.F - self.spatial.L, k=k,
                    M=self.invv, sigma=0.0, which='LM')
        c = np.linalg.lstsq(ev, phi0, rcond=None)[0]

        print("Eigenvector matrix shape:", ev.shape)
        print("Initial state shape:", phi0.shape)

        initialEig = np.linalg.ls(ev, phi0)
        reconstructed = ev @ initialEig

        print(
            "Reconstruction error:",
            np.linalg.norm(reconstructed - phi0) / np.linalg.norm(phi0)
        )
        return ((np.exp(np.outer(times, lam)) * c) @ ev.T).real
        # evdecomp gives phi= Ve^lam*t V^-1 phio
        
        




    

import scipy.linalg as sl

T = 1.024e-3                                   # divisible by every dt = 1e-6 * 2**i
Mdiag = s.V * np.repeat(1 / np.array([1.4e9, 2.2e5]), len(s.active))
A = (ss.diags(1 / Mdiag) @ (s.F - s.L)).toarray()   # s_invv_diag = Transient(s, 1).invv.diagonal()
exact = sl.expm(A * T) @ s.phi

dts, errs = [], []
for i in range(10):
    dt = 1e-6 * 2**i
    Tr = Transient(s, dt)
    Tr.phi = s.phi.copy()
    Tr.prep()
    for _ in range(round(T / dt)):
        Tr.step()
    dts.append(dt)
    errs.append(np.linalg.norm(Tr.phi - exact) / np.linalg.norm(exact))

# print([[dts[i],errs[i]] for i in range(len(dts))])
plt.figure()
plt.loglog(dts, errs, "o-", label="Backward Euler error")
plt.loglog(dts, errs[-1] * np.array(dts) / dts[-1], "k--", label="first order")
plt.xlabel("Time step (s)")
plt.ylabel("Relative L2 error")
plt.title("Temporal Convergence")
plt.grid(True, which="both", alpha=0.4)
plt.legend()
plt.show()

# fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# ax[0].plot(times, numerical / numerical[0], label="Backward Euler")
# ax[0].plot(times, analytical / analytical[0], "--", label="Analytical")

# ax[1].plot(times, error)

# ax[0].set(xlabel="Time (s)", ylabel="Relative flux")
# ax[1].set(xlabel="Time (s)", ylabel="Relative L2 error")

# for a in ax:
#     a.grid()
#     a.legend() if a is ax[0] else None

# plt.tight_layout()
# plt.show()




# plt.plot(times, tFlux / tFlux[0])
# plt.xlabel("Time (s)")
# plt.ylabel("Relative neutron flux")
# plt.title(f"Backward Euler with k = {s.k:.5f} derived from spatial solver")
# # plt.legend()
# plt.grid()
# plt.show()
