# Minecraft-PWR
### Three-Dimensional Transient Reactor Physics Simulation in Minecraft

**A Python-based reactor physics framework for translating Minecraft block geometries into spatially discretized nuclear reactor models.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)
![ParaView](https://img.shields.io/badge/ParaView-3D%20Visualization-0099CC)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

---
<p align="center">
  <img src="fancy/pv3d2.png" alt="MCCORE" width="750">

</p>

---

## Overview

**Minecraft-PWR** is an experimental three-dimensional reactor physics simulation framework designed to model a pressurized water reactor (PWR) constructed within Minecraft.

The project aims to translate Minecraft block coordinates and material assignments into a numerical representation suitable for solving the time-dependent multigroup neutron diffusion equations. The resulting framework is intended to support spatial neutron flux calculations, delayed neutron precursor dynamics, and transient reactor behavior.

The reactor model is based on the **IAEA two-energy-group PWR benchmark**, with material cross sections and reactor configurations informed by the benchmark literature.

The long-term objective is to develop a simulation that can respond to changes in reactor geometry, material placement, and control rod configurations within Minecraft.

### Numerical framework

| Component | Method |
|---|---|
| Geometry | 2D/3D Cartesian block representation |
| Spatial discretization | Cell-centered finite volume method (FVM), planned |
| Energy discretization | Two-group neutron diffusion, planned |
| Time integration | Backward Euler, planned |
| Delayed neutron treatment | Six precursor groups, planned |
| Material properties | Multigroup macroscopic cross sections |
| Spatial indexing | Flattened Cartesian arrays |
| Visualization | ParaView / VTK ImageData |

> **Development status:** The repository currently contains the beginnings of the geometry construction, uniform spatial subdivision, material property definition, and visualization export systems. The coupled neutronics solver and live Minecraft integration remain development objectives.

---

## Project Architecture

The simulation separates **reactor geometry** from **nuclear material properties**, allowing numerical methods to operate on a consistent spatial mesh while querying material-dependent parameters independently.

```text
Minecraft-PWR/
│
├── src/
│   ├── grid.py          # Geometry, indexing, subdivision, export
│   └── mats.py          # Nuclear materials and cross sections
│
├── docs/                # Documentation and benchmark references
├── fancy/               # Visualization and generated output assets
└── README.md
```
<!-- 
### Computational workflow

```text
Minecraft Block Geometry
          │
          ▼
   Block Coordinates
   + Material IDs
          │
          ▼
     Grid.rebuild()
   ├─ Normalize positions
   ├─ Generate bounding grid
   ├─ Insert surrounding air cells
   └─ Sort coordinates and materials
          │
          ▼
    Grid.subdivide()
   ├─ Generate local offsets
   ├─ Construct refined mesh
   ├─ Propagate material IDs
   └─ Restore spatial ordering
          │
          ▼
   Discretized Reactor
          │
          ├────────► ParaView Export
          │
          ▼
   Spatial FVM Assembly
          │
          ▼
  Multigroup Transient Solver
          │
          ▼
 Neutron Flux / Power / Kinetics
```

The last three computational stages represent the planned solver architecture. -->

<!-- ---

## 1. `grid.py` — Geometry and Spatial Discretization

The `Grid` class provides the geometric foundation of the reactor model. It converts a collection of Minecraft block locations and associated material IDs into a structured Cartesian representation.

Instead of requiring the spatial solver to search for neighboring cells, the grid is constructed as a complete rectangular domain with a predictable flattened indexing convention.

### 1.1 Grid initialization

The class accepts a tuple containing two corresponding arrays:

- **Coordinates:** An array of block locations in either two or three dimensions.
- **Materials:** An integer material ID for every block.

```python
import numpy as np
from grid import Grid

coords = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
])

materials = np.array([0, 1, 2, 21])

grid = Grid((coords, materials))
```

Each row in the coordinate array corresponds directly to the material ID at the same index.

This association is maintained throughout geometry construction and subdivision.

### 1.2 Geometry normalization and padding

The `rebuild()` method constructs the initial computational domain.

First, the coordinates are normalized relative to their minimum values:

\[
\mathbf{x}_i'=\mathbf{x}_i-\mathbf{x}_{\min}
\]

A rectangular bounding region is then constructed using the minimum and maximum coordinate values along each spatial axis.

An additional block of padding is introduced beyond the original geometry in both directions of every axis.

The padding serves two purposes:

1. It creates a complete Cartesian domain, including locations not explicitly occupied by the original reactor blocks.
2. It allows neighboring cells to be accessed using a consistent indexing scheme rather than repeated coordinate searches.

Padding coordinates not occupied by an existing reactor block are assigned a material ID of `-1`, representing air or an external computational region.

The domain therefore becomes rectangular regardless of the original arrangement of reactor blocks.

### 1.3 Coordinate indexing convention

A major design choice is the ordering of the flattened spatial array.

The current implementation uses:

**X fastest → Z second → Y slowest**

This convention treats Minecraft's Y coordinate as the vertical direction, preserving the natural interpretation of horizontal reactor layers.

The ordering is implemented using NumPy's lexicographic sorting:

```python
sorter = np.lexsort((
    map[:, 0],
    map[:, 2],
    map[:, 1]
))
```

The same permutation is applied to the coordinate and material arrays, ensuring that corresponding elements remain aligned.

For a rectangular grid with dimensions \(n_x,n_y,n_z\), the resulting linear index is

\[
\boxed{i=i_x+n_xi_z+n_xn_zi_y}
\]

where each spatial index begins at zero.

This makes the neighbor offsets particularly convenient:

| Direction | Linear index offset |
|---|---|
| X | \(\pm 1\) |
| Z | \(\pm n_x\) |
| Y | \(\pm n_xn_z\) |

These offsets provide a basis for assembling the face-leakage terms of the finite-volume diffusion matrix.

Neighbor accesses must still respect the physical domain boundaries.

### 1.4 Uniform spatial subdivision

The `subdivide()` method generates a finer computational mesh by subdividing each Minecraft block into a prescribed number of cells.

```python
grid.subdivide(Nx=5, Ny=5, Nz=5)
```

Assuming each Minecraft block represents a cube with a side length of one meter, the spatial cell widths are

\[
h_x=\frac{1}{N_x},\qquad
h_y=\frac{1}{N_y},\qquad
h_z=\frac{1}{N_z}
\]

For example, a subdivision of \(5\times5\times5\) produces

\[
5^3=125
\]

computational cells per original block, each with a volume of

\[
V_i=0.2^3=0.008\ \mathrm{m^3}
\]

The subdivision procedure uses NumPy broadcasting to generate all possible combinations of local spatial offsets.

```python
sublevels = np.meshgrid(
    *[
        np.linspace(0, 1, N + 1)[:-1]
        for N in Ns
    ],
    indexing="ij"
)

sublevels = np.stack(
    sublevels, axis=-1
).reshape(-1, dim)
```

Each local offset is then applied to every original block:

```python
map = (
    map[:, None, :] + sublevels[None, :, :]
).reshape(-1, dim)
```

The expanded mesh is subsequently sorted to restore the established X–Z–Y ordering.

**Material propagation**

All subdivisions of an original block inherit its material ID.

This is accomplished using:

```python
submats = np.repeat(
    self.mat,
    sublevels.shape[0]
)
```

The coordinate and material arrays are then sorted using the same permutation, maintaining a one-to-one correspondence between every computational cell and its material.

The resulting mesh is suitable for subsequent assignment of spatially dependent diffusion coefficients, cross sections, and reactor state variables.

### 1.5 ParaView export

The `xport()` method exports the computational mesh as a VTK ImageData (`.vti`) file using PyVista.

The export includes structured grid dimensions, cell spacing, grid origin, and a material-ID field.

```python
grid.xport("reactor.vti")
```

The resulting file can be opened in ParaView to visualize reactor geometry and material distributions.

The export mechanism is intended to be extended to include numerical solution fields such as neutron flux, fission power density, and delayed neutron precursor concentrations.

**Current limitation:** The export uses an explicit subdivision count and assumes a regular three-dimensional mesh. The subdivision parameters supplied during export must match the current mesh.

---

## 2. `mats.py` — Nuclear Material Definitions

The `mats.py` module defines the nuclear properties associated with each reactor material.

Rather than storing a separate collection of cross sections for every computational cell, each cell refers to a material definition containing the required two-group nuclear data.

This separates the physical description of reactor materials from the geometric arrangement of those materials.

### 2.1 Material data structure

Material properties are represented using a frozen Python dataclass:

```python
@dataclass(frozen=True, slots=True)
class Material:
    name: str
    D: tuple
    abs: tuple
    s: tuple[tuple, tuple]
    f: tuple
    chi: tuple = (1.0, 0.0)
```

The frozen dataclass provides an immutable material definition, while `slots=True` restricts dynamically assigned attributes and reduces instance overhead.

A material contains the following nuclear parameters:

| Property | Symbol | Description |
|---|---|---|
| `name` | — | Material identifier |
| `D` | \(D_g\) | Group diffusion coefficients |
| `abs` | \(\Sigma_{a,g}\) | Macroscopic absorption cross sections |
| `s` | \(\Sigma_{s,g\rightarrow h}\) | Intergroup scattering cross sections |
| `f` | \(\nu\Sigma_{f,g}\) | Neutron-production cross sections |
| `chi` | \(\chi_g\) | Fission neutron spectrum |
| `rem` | \(\Sigma_{r,g}\) | Derived removal cross sections |

Here, \(g\) and \(h\) indicate neutron energy groups.

### 2.2 Two-group representation

The current implementation assumes two neutron energy groups, conventionally interpreted as:

- **Group 1:** Fast neutrons
- **Group 2:** Thermal neutrons

Material properties are provided as tuples containing one value for each group.

For example:

```python
fuel = Material(
    "Fuel",
    (1.5, 0.4),
    (0.010, 0.085),
    ((0.0, 0.020), (0.0, 0.0)),
    (0.0, 0.135)
)
```

The scattering matrix follows a *from-group → to-group* convention:

\[
\mathbf{\Sigma}_s=
\begin{bmatrix}
\Sigma_{s,1\rightarrow1} & \Sigma_{s,1\rightarrow2}\\
\Sigma_{s,2\rightarrow1} & \Sigma_{s,2\rightarrow2}
\end{bmatrix}
\]

This convention makes it straightforward to distinguish downscattering from upscattering.

### 2.3 Automatic removal cross sections

The material class provides a computed `rem` property.

For energy group \(g\), the removal cross section is

\[
\boxed{
\Sigma_{r,g}
=
\Sigma_{a,g}
+
\sum_{h\ne g}\Sigma_{s,g\rightarrow h}
}
\]

It accounts for neutrons lost from an energy group through absorption or scattering into another energy group.

The property is calculated directly from the supplied absorption and scattering data:

```python
@property
def rem(self):
    return tuple(
        self.abs[h] + sum(
            self.s[h][g]
            for g in range(2)
            if g != h
        )
        for h in range(2)
    )
```

This avoids independently specifying removal cross sections and reduces the possibility of inconsistent nuclear data.

### 2.4 Implemented materials

The current module defines four reactor material configurations.

| ID | Material | Description |
|---|---|---|
| `0` | `fuel` | Standard fuel region |
| `1` | `frod` | Fuel region incorporating a control rod |
| `2` | `refl` | Reflector region |
| `21` | `rrod` | Reflector region incorporating a control rod |
| `-1` | Air | Geometry-padding identifier used by `grid.py` |

The four reactor material objects are organized in a lookup dictionary:

```python
MATERIALS = {
    0: fuel,
    1: frod,
    2: refl,
    21: rrod
}
```

This allows the numerical solver to retrieve material properties by integer index without hardcoding individual materials into the spatial assembly routines.

For example:

```python
material = MATERIALS[0]

print(material.D)
print(material.abs)
print(material.rem)
```

The current definitions represent benchmark-style, homogenized reactor regions. They are not intended to reproduce individual fuel pellets, cladding layers, or control rod geometries explicitly.

### 2.5 Role in the neutron diffusion solver

The material definitions will provide the coefficients required to discretize the multigroup neutron diffusion equations.

The steady-state multigroup equation may be written as

\[
-\nabla\cdot(D_g\nabla\phi_g)
+\Sigma_{r,g}\phi_g
=
\sum_{h\ne g}\Sigma_{s,h\rightarrow g}\phi_h
+
\frac{\chi_g}{k_{\mathrm{eff}}}
\sum_h\nu\Sigma_{f,h}\phi_h
\]

where:

- \(\phi_g\) is the neutron flux in energy group \(g\).
- \(D_g\) is the diffusion coefficient.
- \(\Sigma_{r,g}\) is the removal cross section.
- \(\Sigma_{s,h\rightarrow g}\) is the intergroup scattering cross section.
- \(\nu\Sigma_{f,h}\) is the neutron-production cross section.
- \(\chi_g\) is the fission neutron spectrum.
- \(k_{\mathrm{eff}}\) is the effective multiplication factor.

The spatial grid determines cell geometry and neighboring-cell connectivity, while `mats.py` supplies the physics coefficients associated with each cell.

The two modules therefore provide complementary inputs to the future numerical solver.

---

## 3. Planned Transient Reactor Solver

The next major development stage is the implementation of a three-dimensional, two-energy-group neutron diffusion solver with delayed neutron precursor dynamics.

The time-dependent neutron balance equation is

\[
\frac{1}{v_g}\frac{\partial\phi_g}{\partial t}
=
\nabla\cdot(D_g\nabla\phi_g)
-\Sigma_{r,g}\phi_g
+\sum_{h\ne g}\Sigma_{s,h\rightarrow g}\phi_h
+\chi_g\left[
(1-\beta)\sum_h\nu\Sigma_{f,h}\phi_h
+\sum_{j=1}^{6}\lambda_jC_j
\right]
\]

with delayed neutron precursor equations

\[
\frac{\partial C_j}{\partial t}
=
\beta_j\sum_g\nu\Sigma_{f,g}\phi_g
-\lambda_jC_j
\]

The planned numerical approach consists of:

1. **Spatial discretization:** Cell-centered finite volume formulation with face-based neutron leakage.
2. **Group coupling:** Simultaneous treatment of fast and thermal neutron fluxes.
3. **Delayed neutron dynamics:** Six precursor groups with production and radioactive decay.
4. **Implicit integration:** Backward Euler time stepping for the coupled transient system.
5. **Minecraft interaction:** Updates to material assignments and reactor geometry in response to block changes and control rod movement.

The longer-term objective is to combine these elements into a continuously updated reactor model driven by in-game geometry. -->

---

## Installation and Usage

### Requirements

The current geometry and export components use:

- Python 3
- NumPy
- PyVista
- Matplotlib (imported by the current grid module)

Install the dependencies:

```bash
pip install numpy pyvista matplotlib
```

### Example geometry

A manual three-dimensional block assembly can be created using:

```python
import numpy as np
from grid import Grid

A = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 0, 1]
])

M = np.array([1, 2, 3, 4, 5, 6, 7, 8])

G = Grid((A, M))
G.subdivide(5, 5, 5)
G.xport()
```

This reproduces the current grid-module demonstration using test material IDs.

Note that the example IDs `3` through `8` are demonstration values and are not all defined in the `MATERIALS` lookup dictionary.

### Viewing the mesh in ParaView

The exported `.vti` file can be loaded into ParaView.

After opening the dataset and clicking **Apply**:

1. Set the coloring variable to `material`.
2. Select **Surface With Edges** to display individual voxel boundaries.
3. Apply a **Threshold** filter to isolate reactor materials and exclude the surrounding air region.
4. Use clipping or slicing filters to inspect the internal grid structure.

The current export writes to the `fancy/pv/` directory relative to the program's working directory.

---

## 5. Development Roadmap

- [x] Define two-group nuclear material data structures.
- [x] Implement 2D and 3D block-coordinate input.
- [x] Create 3D padded domains for efficient consistent indexing of material cross sections.
- [x] Implement uniform Cartesian spatial subdivision.
- [x] Propagate material assignments to subdivided cells.
- [x] Implement initial VTK/ParaView mesh export.
- [ ] Implement boundary-condition treatment.
- [ ] Assemble the cell-centered finite-volume diffusion operators.
- [ ] Implement the coupled two-group steady-state solver.
- [ ] Verify the numerical methods against analytical and benchmark solutions.
- [ ] Implement eigenvalue calculations for \(k\).
- [ ] Add delayed neutron precursor groups.
- [ ] Implement backward Euler transient integration.
- [ ] Verify Transient solution convergence.
- [ ] Develop Minecraft block-face and control rod interfaces.
- [ ] Integrate reactor state updates with Minecraft simulation ticks.
- [ ] Export spatial neutron flux and power distributions to ParaView.

Items marked complete refer to initial implementations and may require additional testing and refinement.

---

## 6. Benchmark and Verification

The intended reactor physics reference is the IAEA two-group PWR benchmark, with cross-section data and configurations drawn from the benchmark literature available in the repository.

The numerical model will require verification at multiple levels, including spatial discretization, material interfaces, eigenvalue calculations, and transient integration.

Planned verification methods include analytical diffusion solutions, spatial mesh convergence studies, benchmark eigenvalue comparisons, and independently calculated transient solutions.

The benchmark reference document is available in the repository's `docs/` directory.

---

## Contributing

Minecraft-PWR is an academic and experimental scientific computing project. Contributions involving reactor physics, numerical methods, Python optimization, verification, and Minecraft integration are welcome.

<!-- The codebase is being developed modularly so that its geometry, nuclear data, numerical solution, and game-interface components can be tested and improved independently. -->

---

**Minecraft-PWR**  
*inji*
