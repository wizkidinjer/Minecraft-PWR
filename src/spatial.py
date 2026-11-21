"""3d cc finite vol. spatial solver intaking grid.py and calling mats.py

solves the 2/3d steady state spatial neutron diffusion eigenvalue problem
"""

from grid import Grid
from mats import M
import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt


class Sptl:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.N = grid.N
        self.h = 100 * grid.h
        self.V = np.prod(self.h)
        self.G = len(M[0].a)  # 2

        # original indices where active cells are
        self.activeIDX = np.where(self.grid.mat >= 0)[0]
        # index of actualcompressed 3d map

        # LOOPING ARRAY MAIN: gives material of each cell of interest
        self.active = self.grid.mat[self.activeIDX]  # row i is active[i]

    def matCoef(self):
        n = len(self.active)  # total num cells

        self.D = np.zeros((n, 2))  # all these are [cell, group]
        self.rem = np.zeros((n, 2))
        self.s = np.zeros((n, 2, 2))  # cell, from, to
        self.f = np.zeros((n, 2))
        self.chi = np.zeros((n, 2))

        for matID, material in M.items():
            # example 0, fuel so i can pull fuel props from 0 in th emat array
            mask = self.active == matID  # mask where matID is fuel
            self.D[mask] = material.D
            # if we are on 0 for fuel, all the fuel cells will get fuel.D, etc
            self.rem[mask] = material.rem
            self.s[mask] = material.s
            self.f[mask] = material.f
            self.chi[mask] = material.chi

    def build(self, alfa=0):
        self.matCoef()

        Nx = self.N[0]
        if len(self.N) == 3:
            strides = np.array(
                [
                    1,
                    -1,
                    Nx * self.N[2],
                    -Nx * self.N[2],
                    Nx,
                    -Nx,
                ]
            ).astype(int)
        # length to x nbr, y nbr, z nbr
        else:
            strides = np.array([1, -1, Nx, -Nx]).astype(int)

        nbrIDX = self.activeIDX[:, None] + strides[None, :]  # [cell, nbridx]
        # gives real index for each nbr for each cell
        nbr = self.grid.mat[nbrIDX]  # retrieve material of each nbr, for each cell
        # print(nbr)
        # print(nbrIDX)
        interior = nbr >= 0  # where real cells are
        bdry = nbr == -1  # where air

        hdir = np.repeat(self.h, 2)
        Area = self.V / hdir
        # returns area based on which face direction you are lookings

        Bs = []  # B matrix for each group
        Ss = [[None for k in range(self.G)] for l in range(self.G)]
        # i want to make this generalized instead of hardcoding it.o
        # only 0,1 and 1,0 are used in a 2 group case and theres no upscatter
        # so really only 1,0 here
        Fs = [[None for k in range(self.G)] for l in range(self.G)]

        ##################################### BUILDING B
        for g in range(self.G):
            rows, cols, vals = [], [], []

            i, d = np.where(interior)
            # returns row, nbr directions that are interiors. if i=[0] d=[1] there is interior on the -x face
            j = np.searchsorted(self.activeIDX, nbrIDX[i, d])
            # nbrIDX[i,d] is real index of face d of cell i. searchsorted finds the matrix index where
            # that real nbr index fits into the real index.
            Di = self.D[i, g]
            Dj = self.D[j, g]  # nbr D
            Df = 2 * Di * Dj / (Di + Dj)  # harmonic

            fancyD = Df * Area[d] / hdir[d]

            # bdry
            ib, db = np.where(bdry)
            # ib is the bdry cell and db is the bdry direction. db used just for geometry
            h = hdir[db]

            bc = self.D[ib, g] * Area[db] / (h / 2 + alfa * self.D[ib, g])
            # given robin bc substituting for D

            rows = np.concatenate((i, i, ib))
            cols = np.concatenate((j, i, ib))
            vals = np.concatenate((-fancyD, fancyD, bc))
            # this is a way to design sparse matrices by defining column and row values.
            # so at all the i,j slots, the negative fancy D will go in that slot. There are six of these per row i
            # on the diagonals the positives are placed, B_ii = fancy D and on boundary diagonal values the BC goes in.

            # REMOVAL ###
            m = np.arange(len(self.active))
            # just make a list of every row/col
            rows = np.concatenate((rows, m))
            cols = np.concatenate((cols, m))
            vals = np.concatenate((vals, self.V * self.rem[:, g]))
            # just go down the line with precalcualted removals

            B = ss.coo_matrix(
                (vals, (rows, cols)), shape=(len(self.active), len(self.active))
            ).tocsr()
            # this is how coordinate sparse matrix is build. specify values, their locations, and shape of real matrix
            Bs.append(B)
            ######################################################
            ### SCATTERING and FISSION

            for h in range(self.G):  # swap out if too slow
                Ss[g][h] = ss.diags(
                    self.V * self.s[:, h, g]
                ).tocsr()  # s is [cell, from, to]
                # and g is to, row landing in
                # i dont think i have to deal with h==g because they are set to zero in mats.py
            for h in range(self.G):
                Fs[g][h] = ss.diags(self.chi[:, g] * self.V * self.f[:, h]).tocsr()
                # Fs[g][h] means row g col h. row g is the neutron source, so chi is for g.
                # col h is the fissioned group, the source of the fission, multiplied by flux h so its a column

        # print([b.toarray() for b in Bs])
        B = ss.block_diag(Bs, format="csr")
        # print(B.toarray())
        S = ss.bmat(Ss, format="csr")
        self.F = ss.bmat(Fs, format="csr")
        self.L = B - S

        # for name in ("F", "L"):
        #     A = getattr(self, name)
        #     memory = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes

        #     print(f"{name}: {memory / 1024**2:.3f} MB")

    def keig(self, tol=1e-10, maxit=10_000):
        """L0= 1/k F0
        L0^w+1 = 1/k F0^w"""
        L = self.L
        F = self.F

        # kL phi = F phi
        # k phi = L^-1 F phi = M phi

        # solving with GS:
        # LO = ss.tril(L, format="csr")
        # U = ss.triu(L, k=1, format="csr")
        lu = ss.linalg.splu(L.tocsc())

        fi = np.ones(self.L.shape[0])  # initial guess flat
        fi /= np.linalg.norm(fi)  # len 1

        k = 1  # initial guess
        for kit in range(maxit):

            b = F @ fi
            x = fi.copy()
            # now Lphi = b
            # or LO phi_new = b - U@phi_old

            x = lu.solve(b)

            knew = np.linalg.norm(x)  # technically div by old phi but it is 1
            finew = x / knew

            # if kit % 100 == 0:
            #     print(f"ITER: {kit}")
            #     print(f"K: {knew},    k Diff: {knew-k}")
            #     print(f"PHI DIFF: {np.linalg.norm(finew-fi)}")
            if abs(knew - k) / knew <= tol and np.linalg.norm(finew - fi) <= tol:
                # print(f"OUTERS= {kit}")
                self.k, self.phi = knew, finew
                return knew, finew
            k, fi = knew, finew
        raise RuntimeError("k EIGENVALUE POWER DID NOT CONVERGE")

    def export(self, phi, filename="flux.vti"):
        import pyvista as pv

        Nx, Ny, Nz = map(int, self.N)
        n = len(self.active)

        img = pv.ImageData(
            dimensions=(Nx + 1, Ny + 1, Nz + 1),
            spacing=tuple(self.grid.h),
            origin=(0, 0, 0),
        )

        # Convert your x,z,y ordering into VTK's x,y,z ordering
        def vtk_order(arr):
            return arr.reshape(Ny, Nz, Nx).transpose(2, 0, 1).ravel(order="F")

        # Material IDs, including -1 air cells
        img.cell_data["Material"] = vtk_order(self.grid.mat)

        # Restore each energy group's compressed flux
        for g in range(self.G):
            full = np.full(len(self.grid.mat), np.nan)
            full[self.activeIDX] = phi[g * n : (g + 1) * n]

            img.cell_data[f"Flux Group {g + 1}"] = vtk_order(full)

        img.save("fancy/pv/" + filename)
        print("VTK saved")

    def analy(self, phi_n, homomat, alfa=0):
        """verifies the 2d and 3d numerical solver via the analytical
        homogeneou solution using multiplicative block-centered cosines.

        verifies numerical k with the buckling of the homogeneous bare core."""
        D = M[homomat].D[0]

        coords = self.grid.map[self.activeIDX]

        h = self.h
        L = np.ptp(coords, axis=0) * 100 + h
        x = (coords - coords.min(axis=0)) * 100 + h / 2

        B = np.pi / (L + alfa * 2 * D)

        exact = np.prod(np.cos(B * (x - L / 2)), axis=1)
        exact /= np.linalg.norm(exact)

        N = len(coords)
        numerical = phi_n[:N] / np.linalg.norm(phi_n[:N])
        # numerical /= np.linalg.norm(numerical)

        error = np.linalg.norm(exact - numerical)

        return error, np.max(h), N


A = np.array(
    [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
    ]
)

MMat = np.array([0, 0, 0, 0, 0, 0, 0, 0])

G = Grid((A, MMat))
G.subdivide(4, 2, 4)
s = Sptl(G)
s.build()
print("build done")
s.keig()
print("keig done")


################################## TESTING
# from time import perf_counter

# t = perf_counter()
# G = Grid((A, MMat))
# print(f"Grid initialization: {perf_counter()-t:.2f} s")

# t = perf_counter()
# G.subdivide(5, 5, 5)
# print(f"Subdivision: {perf_counter()-t:.2f} s")

# t = perf_counter()
# s = Sptl(G)
# s.matCoef()
# print(f"Material coefficients: {perf_counter()-t:.2f} s")

# t = perf_counter()
# s.build()
# print(f"Sparse matrix assembly: {perf_counter()-t:.2f} s")

# t = perf_counter()
# k, p = s.keig()
# print(f"k-eigenvalue solve: {perf_counter()-t:.2f} s")
# print(f"k_eff: {s.k:.8f}")


import matplotlib.pyplot as plt, scipy.sparse.linalg as ssl, contextlib, io

A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
MMat = np.array([0, 0, 0, 0])

A = np.array(
    [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
    ]
)

MMat = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# exact bare-box solution: phi_g = c_g * prod sin(pi x_d / L_d), zero flux on faces
f, dim = M[0], A.shape[1]
Lc = (np.ptp(A, axis=0) + 1) * 100.0  # box width per direction (cm)
B2 = np.sum((np.pi / Lc) ** 2)  # geometric buckling
c21 = f.s[0][1] / (f.a[1] + f.D[1] * B2)  # thermal/fast flux ratio
k_ex = f.f[1] * c21 / (f.rem[0] + f.D[0] * B2)  # exact k

hs, ephi, ek = [], [], []
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for n in [2**i for i in range(4)]:  # use range(1, 4) in 3D
    G = Grid((A, MMat))
    G.subdivide(*[n] * dim)
    s = Sptl(G)
    s.matCoef()
    with contextlib.redirect_stdout(io.StringIO()):
        s.build(0)  # zero flux on faces
        k, _ = s.keig(tol=1e-12)

    nc = len(s.active)
    xc = (
        G.map[s.activeIDX] - G.map[s.activeIDX].min(0)
    ) * 100 + s.h / 2  # cell centers (cm)
    S = np.prod(
        np.sin(np.pi * xc / Lc), axis=1
    )  # product of cosines about the box center
    ex = np.concatenate((S, c21 * S))

    Q = np.concatenate(((f.D[0] * B2 + f.rem[0]) * S, np.zeros(nc)))
    phi = ssl.spsolve(s.L.tocsc(), s.V * Q)

    hs.append(s.h.max())
    ephi.append(np.linalg.norm(phi - ex) / np.linalg.norm(ex))
    ek.append(abs(k - k_ex))
    p = lambda e: f"{np.log2(e[-2] / e[-1]):6.3f}" if len(e) > 1 else "   -  "
    print(
        f"n={n:3d} h={hs[-1]:7.3f}  phi err={ephi[-1]:.3e} p={p(ephi)}  |  k={k:.8f} err={ek[-1]:.3e} p={p(ek)}"
    )

    if n <= 8:  # centerline profile
        mid = [
            np.unique(xc[:, d])[len(np.unique(xc[:, d])) // 2] for d in range(1, dim)
        ]
        row = np.where(np.all(np.isclose(xc[:, 1:], mid), axis=1))[0]
        row = row[np.argsort(xc[row, 0])]
        ax.plot(
            xc[row, 0],
            phi[row] / np.prod(np.sin(np.pi * np.array(mid) / Lc[1:])),
            "o",
            ms=4,
            label=f"n={n}",
        )

xf = np.linspace(0, Lc[0], 400)
ax.plot(xf, np.sin(np.pi * xf / Lc[0]), "k--", lw=1.2, label="exact")
ax.set(
    xlabel="x (cm)",
    ylabel="Flux along yz centerline",
    title=f"3D Bare homogeneous, k = {k_ex:.6f}",
)
ax.legend()
ax.grid(True)

hs = np.array(hs)
ax2.loglog(hs, ephi, "o-", label="Flux")
ax2.loglog(hs, ek, "s-", label="k")
for e in (ephi, ek):
    ax2.loglog(hs, e[-1] * (hs / hs[-1]) ** 2, "k--", lw=0.8, label="2 order")
ax2.set(xlabel="h (cm)", ylabel="error", title="Convergence [2nd order]")
ax2.legend()
ax2.grid(
    True,
    which="both",
)
fig.tight_layout()
plt.savefig("fancy/3dSpatialConvergence.png", dpi=400)
plt.show()
