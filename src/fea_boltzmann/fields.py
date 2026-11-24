"""Streaming fields and scalar data on the sphere."""

from __future__ import annotations

import numpy as np

from fea_boltzmann.mesh import SphereMesh


def rotation_field(omega: np.ndarray):
    """Return the rigid rotation ``x ↦ ω × x`` on the unit sphere.

    The field is tangent to the sphere and divergence-free, so it streams
    particles without creating or destroying them.
    """
    angular_velocity = np.asarray(omega, dtype=float).reshape(3)

    def beta(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return np.cross(angular_velocity, points)

    return beta


def gaussian_blob(mesh: SphereMesh, center: np.ndarray, width: float) -> np.ndarray:
    """Nodal samples of a geodesic Gaussian peaked at ``center``.

    ``width`` is the angular standard deviation in radians.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    direction = np.asarray(center, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)
    cosine = np.clip(mesh.vertices @ direction, -1.0, 1.0)
    angle = np.arccos(cosine)
    return np.exp(-0.5 * (angle / width) ** 2)


def manufactured_source(
    mesh: SphereMesh,
    tau: float,
    absorption: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nodal solution ``f = x`` and the matching continuous source.

    With angular velocity ``ω = e_z`` the streaming field is ``β = (−y, x, 0)``.
    On the unit sphere ``β · ∇x = −y`` and the spherical mean of ``x`` is zero,
    so the stationary linear Boltzmann equation holds with

        q = −y + (1/τ + σ_a) x.

    Both fields are linear in the ambient coordinates, so their P1 interpolants
    are exact on every flat facet.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if absorption < 0.0:
        raise ValueError("absorption must be nonnegative")
    x = mesh.vertices[:, 0].copy()
    y = mesh.vertices[:, 1]
    source = -y + (1.0 / tau + absorption) * x
    return x, source
