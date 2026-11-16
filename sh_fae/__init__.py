"""SH-FAE: Soil Hydraulic Functional AutoEncoder for Physics-Constrained Inversion."""

from .models.decoder import ConstrainedDecoder
from .models.encoder import SetEncoder
from .models.fae import SHFAE
from .physics.constitutive import TorchAEConstitutive, VGMConstitutive
from .physics.grid import Grid1D
from .physics.material import Material
from .physics.solver import BoundaryConditions, NewtonOptions, solve_richards

__version__ = "0.1.0"

__all__ = [
    "SHFAE",
    "SetEncoder",
    "ConstrainedDecoder",
    "Grid1D",
    "Material",
    "BoundaryConditions",
    "NewtonOptions",
    "TorchAEConstitutive",
    "VGMConstitutive",
    "solve_richards",
]
