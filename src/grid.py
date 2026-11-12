"""grid construction. build array indexing from 3d layout"""

"""intakes a bunch of block locations (or preordained coordinates for testing)

    The idea is to pad the grid with an extra block around the furthest out core blocks
    and make the entire volume of interest strictly rectangular.
    This way i can maintain a raveled index state instead of needing to locate cells to
    determine if they are valid cells."""
import numpy as np
from dataclasses import dataclass


class Grid:
    """intakes an array of (array: block locations, array: associated block materials)
    blocks include ref and fuel. unsure on incorporation of rods yet"""


    def __init__(self, blocks: tuple[tuple, tuple[int]]):
        self.coo = blocks[0]
        self.mat = blocks[1]
        self.rebuild()

        

    def rebuild(self):
        coo = self.coo
        xs = np.arange(np.min(coo[:, 0]), np.max(coo[:, 0]) + 1)
        ys = np.arange(
            np.min(coo[:, 1]), np.max(coo[:, 1]) + 1
        )  # making a box encapsulating the entire core and any air in that box
        if coo.shape[1] == 3:
            zs = np.arange(np.min(coo[:, 2]), np.max(coo[:, 2]) + 1)

            self.meshd = np.meshgrid(xs, ys, zs, indexing="ij")
        else:
            self.meshd = np.meshgrid(xs, ys, indexing="ij")



    def subdivide(self, N: tuple[int, int, int]):
        xnew = np.linspace(meshd[:, 0])

    # def gridIndex(loc: tuple[float, float], matidx):
    # """3d grid to matrix indexing"""
    #     x, y = loc
    #     if matidx != 0:
    #         return y + 3 * x
    #     else:
    #         pass

    # def getBlocks():
    #     """update blocks function if geometry changes"""
    #     pass


A = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
G = Grid((A, (0, 0, 0, 0)))

print(A)

print(G.coo)
