"""3d cc finite vol. spatial solver intaking grid.py and calling mats.py

solves the 2/3d steady state spatial neutron diffusion eigenvalue problem
"""

from grid import Grid
from mats import M
import numpy as np


class Sptl:
    def __init__(self, grid: Grid):
        self.xyz = grid.map
        self.matindex = grid.mat
        self.N = grid.N

        self.h = 100 * grid.h
        self.V = np.prod(self.h)
        # print(self.h, self.V)

        # xnbr = +1
        # z nbr = num cells in x
        # y nbr = num x* num z
        self.nbr = np.array(
            [1, -1, self.N[0], -self.N[0], self.N[0] * self.N[2], self.N[0] * self.N[2]]
        )  # north south east west top bottom
        print(self.nbr)

    def d(self, m, f,g): #more efficient to precompute?
        return 2 * (M[m].D[g] + M[f].D[g]) / (self.h[m] * M[f].D[g] + self.h[f] * M[m].D[g])

    @property
    def A(self):
        """build the A matrix. first build B, abs plus leakage values of each cell. should be 5/7diag for 2/3d"""
        B=np.zeros_like()
        for g in range(len.M[0].D): # for g in number of groups #MAINOUTERLOOP G major
            
            


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
print(s.d(1,2,0))

print(M[0].D[0],)