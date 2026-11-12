"""grid construction. build array indexing from 3d layout"""

"""intakes a bunch of block locations (or preordained coordinates for testing)

    The idea is to pad the grid with an extra block around the furthest out core blocks
    and make the entire volume of interest strictly rectangular.
    This way i can maintain a raveled index state instead of needing to locate cells to
    determine if they are valid cells."""
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


class Grid:
    """intakes an array of (array: block locations, array: associated block materials)
    blocks include ref and fuel. unsure on incorporation of rods yet"""

    def __init__(self, blocks: tuple[tuple, tuple[int]]):
        self.coo = np.array(blocks[0])
        self.mat = np.array(blocks[1])
        assert self.coo.shape[0] == len(self.mat)
        self.rebuild()

    def rebuild(self):
        """builds the location state of the core."""
        coo = self.coo
        coo -= coo.min(axis=0)  # normalize

        if coo.shape[1] == 3:
            sortidx = np.lexsort((coo[:, 0], coo[:, 2], coo[:, 1]))
        else:
            sortidx = np.lexsort((coo[:, 0], coo[:, 1]))
        coo = coo[sortidx]
        # make coordinates vary quickest in x then z then y. my pref because minecraft does y as height.
        mat = self.mat[sortidx]

        # add padding to fill out the square

        maxs = coo.max(axis=0)
        mins = coo.min(axis=0)

        dirs = [np.arange(low - 1, hi + 2) for low, hi in zip(mins, maxs)]
        # make an arange for each axis, 1 beyond the lowest and 1 beyond the max
        padding = np.meshgrid(*dirs, indexing="ij")  # merge all the combos into coords
        # print(padding)
        padcoords = np.stack(padding, axis=-1).reshape(-1, coo.shape[1])
        print("PADCOORDS")
        print(padcoords.shape)
        # print(padcoords)  ######?????????????????

        map = np.unique(np.vstack((coo, padcoords)),sorted=False,axis=0)
        print(map)

        #add the padding after the known blocks. this way i can resort with the material values too
 
        plt.scatter(padcoords[:, 0] + 0.5, padcoords[:, 1] + 0.5, s=100, c="g")
        plt.scatter(
            coo[:, 0] + 0.5,
            coo[:, 1] + 0.5,
            s=100,
        )
        plt.scatter(map[:, 0] + 0.5, map[:, 1] + 0.5, s=100, c="b")
        plt.xlim(-2, 4)
        plt.ylim(-2, 4)
        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.show()

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
M=np.array([1,2,3,4])
G = Grid((A, M))
