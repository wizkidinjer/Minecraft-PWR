"""3d cc finite vol. spatial solver intaking grid.py and calling mats.py

solves the 2/3d steady state spatial neutron diffusion eigenvalue problem
"""

from grid import Grid
from mats import *


class Sptl():
    def __init__(self,grid: Grid):
        self.grid=grid

    
    def left(self):
        