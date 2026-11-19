"""3d cc finite vol. spatial solver intaking grid.py and calling mats.py

solves the 2/3d steady state spatial neutron diffusion eigenvalue problem
"""

from grid import Grid
from mats import M
import numpy as np
from scipy.sparse import diags


class Sptl:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.N = grid.N
        self.h = 100 * grid.h

        # original indices where active cells are
        self.activeIDX = np.where(self.grid.mat >= 0)[0]
        # index of actualcompressed 3d map

        # LOOPING ARRAY MAIN: gives material of each cell of interest
        self.active = self.grid.mat[self.activeIDX]  # row i is active[i]

    def matCoef(self):
        n = len(self.active)  # total num cells

        self.D = np.zeros((n, 2)) #all these are [cell, group]
        self.rem = np.zeros((n, 2))
        self.s = np.zeros((n, 2, 2)) # cell, from, to
        self.f = np.zeros((n, 2))
        self.chi = np.zeros((n, 2))

        for matID, material in M.items():
            # example 0, fuel so i can pull fuel props from 0 in th emat array
            mask = self.grid.mat == matID  # mask where matID is fuel
            self.D[mask] = material.D
            # if we are on 0 for fuel, all the fuel cells will get fuel.D, etc
            self.rem[mask] = material.rem
            self.s[mask] = material.s
            self.f[mask] = material.f
            self.chi[mask] = material.chi

        print(self.s[2105, :, :])

    # def build(self):
    #     """build the A matrix. first build B, abs plus leakage values of each cell. should be 5/7diag for 2/3d"""
    #     mat = self.grid.mat
    #     N = self.grid.N
    #     h = self.grid.h * 100
    #     V = np.prod(h)

    #     rem = np.zeros(((len(self.grid.mat)), 2))  # [G0, G1]

    #     # xnbr = +1
    #     # z nbr = num cells in x
    #     # y nbr = num x* num z
    #     self.nbr = np.array([1, -1, N[0], -N[0], N[0] * N[2], -N[0] * N[2]])
    #     # north south east west top bottom
    #     # print(self.nbr)

    #     B = np.zeros_like((len(real), len(real)))  # mxm matrx

    # def d(self, m, f, g):  # more efficient to precompute?
    #     return (
    #         2
    #         * (M[m].D[g] + M[f].D[g])
    #         / (self.h[m] * M[f].D[g] + self.h[f] * M[m].D[g])
    #     )


# A = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
# M = np.array([0, 2, 2, 0])

A = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
    ]
)
MMat = np.array([1, 2, 3, 4, 5, 6, 7, 8])

G = Grid((A, MMat))
G.subdivide()

s = Sptl(G)
# print(s.d(1, 2, 0))

s.cellCoef()
