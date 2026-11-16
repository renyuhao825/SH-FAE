"""Physics modeling package: 1D grid, material properties, and Richards solver."""

from .constitutive import ConstitutiveRelation, TorchAEConstitutive, VGMConstitutive
from .grid import Grid1D
from .material import Material
from .solver import (
    BoundaryConditions,
    NewtonOptions,
    SimulationResult,
    compute_fluxes,
    hydraulic_head,
    richards_residual,
    solve_richards,
)

__all__ = [
    "Grid1D",
    "Material",
    "ConstitutiveRelation",
    "TorchAEConstitutive",
    "VGMConstitutive",
    "BoundaryConditions",
    "NewtonOptions",
    "SimulationResult",
    "solve_richards",
    "compute_fluxes",
    "hydraulic_head",
    "richards_residual",
]
