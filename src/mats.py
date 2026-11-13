"""building materials to call within the grid
so that the solver can simply query the properties of the local blocks"""

from dataclasses import dataclass


class Material: #nvm i dont
    def __init__(self, name, D, a, s: tuple[tuple, tuple], f, chi=(1.0, 0.0)):
        self.name = name
        self.D = D
        self.a = a
        self.s = s
        self.f = f  # nu Sf together
        self.chi = chi

    @property
    def rem(self):
        return tuple(
            self.abs[h] + sum(self.s[h][g] for g in range(2) if g != h)
            for h in range(2)
        )  # abs plus outscatters


# fuel1 = Material(
#     "Fuel 1", (1.5, 0.4), (0.010, 0.080), ((0.0, 0.020), (0.0, 0.0)), (0.0, 0.135)
# )

fuel = Material(
    "Fuel", (1.5, 0.4), (0.010, 0.085), ((0.0, 0.020), (0.0, 0.0)), (0.0, 0.135)
)

frod = Material(
    "FuelRod", (1.5, 0.4), (0.010, 0.130), ((0.0, 0.020), (0.0, 0.0)), (0.0, 0.135)
)

refl = Material(
    "Reflector", (2.0, 0.3), (0.0, 0.010), ((0.0, 0.040), (0.0, 0.0)), (0.0, 0.0)
)

rrod = Material(
    "ReflectorRod", (3.0, 0.3), (0.0, 0.055), ((0.0, 0.040), (0.0, 0.0)), (0.0, 0.0)
)

MATERIALS = {0: fuel, 1: frod, 2: refl, 21: rrod}
EMPTY = -1
