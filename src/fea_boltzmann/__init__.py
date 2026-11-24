"""Finite-element solution of the linear Boltzmann equation on a sphere."""

from fea_boltzmann.fields import gaussian_blob, manufactured_source, rotation_field
from fea_boltzmann.mesh import SphereMesh, icosphere
from fea_boltzmann.solver import BoltzmannSolver

__all__ = [
    "BoltzmannSolver",
    "SphereMesh",
    "gaussian_blob",
    "icosphere",
    "manufactured_source",
    "rotation_field",
]
