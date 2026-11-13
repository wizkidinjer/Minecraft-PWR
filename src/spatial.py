"""3d cc finite vol. spatial solver intaking grid.py and calling mats.py

solves the 2/3d steady state spatial neutron diffusion eigenvalue problem
"""

from grid import Grid
from mats import *
import numpy as np


class Sptl():
    def __init__(self,grid: Grid):
        self.grid=grid
        self.h=100*grid.h
        self.V=np.prod(self.h)
        print(self.h, self.V)
    
    # def left(self):






A = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
M = np.array([0, 2, 2, 0])

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
M = np.array([1, 2, 3, 4, 5, 6, 7, 8])

G = Grid((A, M))
G.subdivide()

s=Sptl(G)