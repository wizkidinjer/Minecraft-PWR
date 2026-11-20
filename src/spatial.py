"""3d cc finite vol. spatial solver intaking grid.py and calling mats.py

solves the 2/3d steady state spatial neutron diffusion eigenvalue problem
"""

from grid import Grid
from mats import M
import numpy as np
import scipy.sparse as ss


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

    def leakage(self):

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
        self.Bs = []  # B matrix for each group

        ##################################### BUILDING B
        for g in range(1):
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

            bc = self.D[ib, g] * Area[db] / (h * (h / 2 + 2.1312 * self.D[ib, g]))
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
            self.Bs.append(B)
        ######################################################


# A = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
# M = np.array([0, 2, 2, 0])

A = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        # [1, 1, 0],
        # [1, 0, 0],
        # [0, 0, 1],
        # [0, 1, 1],
        # [1, 1, 1],
        # [1, 0, 1],
    ]
)
MMat = np.array(
    [
        0,
        1,
        # 2, 3, 1, 0, 1, 1
    ]
)

G = Grid((A, MMat))
# G.subdivide()

s = Sptl(G)
# print(s.d(1, 2, 0))

s.matCoef()
s.leakage()
