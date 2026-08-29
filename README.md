# Bicontinuous Percolation Simulation

Geometrical simulation of through-thickness bicontinuity in Janus-particle and blend-particle models.

## Overview

This repository contains the Python code used to evaluate phase connectivity in particle-based geometrical models.

Particle centers are arranged on a hexagonal close-packed (HCP) lattice. A periodic Voronoi tessellation is applied in the x and y directions, while the z direction remains nonperiodic.

Two models are compared:

- **Janus-particle model:** Each Voronoi cell is divided into A and B regions by a randomly oriented planar interface.
- **Blend-particle model:** Each complete Voronoi cell is randomly assigned to either phase A or phase B.

Connectivity is determined from same-phase contact areas on shared Voronoi faces. The fractions of each phase belonging to components spanning the evaluation slab in the z direction are calculated as $f_A$ and $f_B$. The bicontinuous spanning fraction is defined as:

$$
f_{\mathrm{bi}} = \min(f_A, f_B)
$$

## Files

- `Bicontinuous_percolation_main.py`  
  Core functions for lattice generation, periodic Voronoi construction, phase assignment, shared-face contact detection, and spanning-component analysis.

- `Bicontinuous_percolation_run.py`  
  Production calculations, parameter sweeps, data export, and visualization.

- `Bicontinuous_percolation_validation.py`  
  Numerical validation and diagnostic visualization of the geometrical model.

## Requirements

The simulations were run using Python 3.14.7 with the following packages:

- NumPy
- SciPy
- NetworkX
- Matplotlib
- Plotly

Install the required packages using:

```bash
pip install numpy scipy networkx matplotlib plotly
```

## Usage

Run the production calculations using:

```bash
python Bicontinuous_percolation_run.py
```

The calculation settings can be changed in the `if __name__ == "__main__":` section of `Bicontinuous_percolation_run.py`.

The principal parameters include:

- `trials`: number of random realizations
- `nx_cell`, `ny_cell`, `nz_cell`: HCP simulation-box dimensions
- `fractions`: prescribed A-phase fractions
- `random_seed`: random seed
- `min_contact_area`: minimum same-phase contact area
- `min_boundary_area`: minimum boundary-contact area

The production script exports the calculated spanning-volume fractions as CSV files and generates the corresponding plots.

Validation functions are provided in `Bicontinuous_percolation_validation.py`. Uncomment the validation or visualization to be executed in its `if __name__ == "__main__":` section, and then run:

```bash
python Bicontinuous_percolation_validation.py
```

## Citation

Citation information will be added after publication.
