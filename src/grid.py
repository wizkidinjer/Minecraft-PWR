"""grid construction. build array indexing from 3d layout"""

"""intakes a bunch of """
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    mat


def gridIndex(loc: tuple[float, float], matidx):
    """3d grid to matrix indexing"""
    x, y = loc
    if matidx != 0:
        return y + 3 * x
    else:
        pass
