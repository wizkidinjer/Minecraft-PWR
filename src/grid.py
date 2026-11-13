"""grid construction. build array indexing from 3d layout"""

"""intakes a bunch of block locations (or preordained coordinates for testing)

    The idea is to pad the grid with an extra block around the furthest out core blocks
    and make the entire volume of interest strictly rectangular.
    This way i can maintain a raveled index state instead of needing to locate cells to
    determine if they are valid cells."""
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pyvista as pv


class Grid:
    """intakes an array of (array: block locations [2/3D], array: associated block materials)
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
        # need plus 1 pad block but arange goes to -1 top or range

        # make an arange for each axis, 1 beyond the lowest and 1 beyond the max
        padding = np.meshgrid(*dirs, indexing="ij")
        # this makes all possible x and y combos. it stores an x array and y array

        padcoords = np.stack(padding, axis=-1).reshape(-1, coo.shape[1])
        # axis -1 pairs the X and Y arrays such that they make coordinate pairs. makes a weird array tho so reshape it
        sameidx = ~np.any(
            np.all(padcoords[:, None, :] == coo[None, :, :], axis=2), axis=1
        )
        aircoords = padcoords[sameidx]
        map = np.vstack((coo, aircoords))

        # print(padcoords[:,None,:].shape)
        # print(coo[None,:,:].shape)
        # add the padded elements to the end of the known blocks. this way i can sort -1 material values where i want
        air = -1 * np.ones(padcoords.shape[0] - coo.shape[0])  # air is -1 mat number
        mat = np.hstack((mat, air))

        # sort map and mats
        if map.shape[1] == 3:
            sorter = np.lexsort((map[:, 0], map[:, 2], map[:, 1]))
        else:
            sorter = np.lexsort((map[:, 0], map[:, 1]))
        map = map[sorter]
        mat = mat[sorter]

        self.map = map.astype(float)  # helps in subdividing
        self.mat = mat
        # self.subdivide()

    def subdivide(self, Nx=5, Ny=5, Nz=5):
        """how many times to divide each block in x, y, z. constant for now.
        This makes each h_i 1/Ni because original block is 1 m"""
        dim = self.map.shape[1]
        map = self.map

        # x: add linspace between 0 and 1. can easily be applied to any block
        # if self.map.shape[1]==3:
        Ns = [Nx, Ny, Nz][:dim]

        sublevels = np.meshgrid(
            *[np.linspace(0, 1, N + 1)[:-1] for N in Ns], indexing="ij"
        )
        # this makes x y and z arrays for each possible coordinate combination but separates them. can impose them back and reshape to see the coords
        sublevels = np.stack(sublevels, axis=-1).reshape(-1, dim)
        # all real relative coords now
        # print(sublevels)

        map = map[:, None, :] + sublevels[None, :, :]  # maintain [block,subdiv, coord]
        submats = np.repeat(self.mat, sublevels.shape[0])
        # since the meshgrid is block major we can sort the mats the same way because we just add copies of the same material value
        map = map.reshape(-1, dim)  # at this point we are block major

        if dim == 3:
            sort = np.lexsort((map[:, 0], map[:, 2], map[:, 1]))
        else:
            sort = np.lexsort((map[:, 0], map[:, 1]))

        self.map = map[sort]  # sort xz major
        self.mat = submats[sort]
        self.h = 1 / np.array(Ns)
        # FIXME should generalize to hx[m] hy[m]  for x y z
        print(len(self.map), len(self.mat))
        print("LENS")

    def xport(self, filename="grid.vti", Nx=5, Ny=5, Nz=5):
        h = 1 / np.array([Nx, Ny, Nz])

        shape = np.rint(np.ptp(self.map, axis=0) / h).astype(int) + 1

        grid = pv.ImageData(
            dimensions=tuple(shape + 1),
            spacing=tuple(h),
            origin=tuple(self.map.min(axis=0)),
        )

        # ParaView/VTK expects X fastest, then Y, then Z
        order = np.lexsort((self.map[:, 0], self.map[:, 1], self.map[:, 2]))
        grid.cell_data["material"] = self.mat[order].astype(int)

        grid.save(f"fancy/pv/{filename}")


# A = np.array(
#     [
#         [0, 0, 0],
#         [0, 1, 0],
#         [1, 1, 0],
#         [1, 0, 0],
#         [0, 0, 1],
#         [0, 1, 1],
#         [1, 1, 1],
#         [1, 0, 1],
#     ]
# )
# M = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# A = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
# M = np.array([1, 2, 3, 4])
# G = Grid((A, M))
# G.subdivide()
# # G.xport()

