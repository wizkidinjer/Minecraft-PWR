"""grid construction. build array indexing from 3d layout"""

"""intakes a bunch of block locations (or preordained coordinates for testing)"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    block: tuple[tuple, int]

    xyzcoords = block[0]
    coo = np.array(xyzcoords)
    mat = block[1]

    # map into m index
    if len(coo[0]) == 3:
        sortedcoo = np.lexsort(
            (coo[:, 2], coo[:, 1], coo[:, 0])
        )  # sort in x, y, z order
    else:
        sortedcoo = np.lexsort((coo[:, 1], coo[:, 0]))

    xs = np.arange(np.min(coo[:, 0]), np.max(coo[:, 0]) + 1)
    ys = np.arange(
        np.min(coo[:, 1]), np.max(coo[:, 1]) + 1
    )  # making a box encapsulating the entire core and any air in that box
    zs = np.arange(np.min(coo[:, 2]), np.max(coo[:, 2]) + 1)

    
    meshd = np.meshgrid(xs, ys, zs, indexing="ijk")

    def getBlocks():
        """update blocks function if geometry changes"""

    # find mat based on m


def gridIndex(loc: tuple[float, float], matidx):
    """3d grid to matrix indexing"""
    x, y = loc
    if matidx != 0:
        return y + 3 * x
    else:
        pass
