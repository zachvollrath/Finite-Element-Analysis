"""P1 surface operators on a closed spherical triangulation."""

from __future__ import annotations

import numpy as np

from fea_boltzmann.mesh import SphereMesh


def assemble_mass(mesh: SphereMesh) -> np.ndarray:
    """Consistent P1 mass matrix, ``M_ij = ∫ φ_i φ_j dS``."""
    areas, _, _ = mesh.face_geometry()
    n = mesh.n_vertices
    mass = np.zeros((n, n))
    faces = mesh.faces
    for local_i in range(3):
        vi = faces[:, local_i]
        np.add.at(mass, (vi, vi), areas / 6.0)
        for local_j in range(local_i + 1, 3):
            vj = faces[:, local_j]
            np.add.at(mass, (vi, vj), areas / 12.0)
            np.add.at(mass, (vj, vi), areas / 12.0)
    return mass


def assemble_advection(mesh: SphereMesh, omega: np.ndarray) -> np.ndarray:
    """Weak streaming operator for the divergence-free field ``β = ω × x``.

    On each facet, ``β`` is taken constant and tangent, ``β_K = ω × n_K``,
    which is the rigid rotation evaluated on the facet normal. The naive
    element form

        C_ij = −∫ (∇φ_i · β) φ_j dS

    has exact zero column sums (constants produce no net mass) but, because
    the faceted field is only approximately divergence-free across edges, it
    is not exactly skew. The matrix returned here is the Euclidean projection
    of that form onto matrices that are skew and annihilate constant vectors:

        C ← skew(C),  then  C ← C − (r 1ᵀ − 1 rᵀ) / n,  r = C 1.

    The projection is consistent: its size is the discrete divergence defect,
    which vanishes as the icosphere approaches the smooth sphere. Skewness
    gives ``fᵀ C f = 0`` (streaming does not change the L² norm) and
    ``C 1 = 0`` (a uniform distribution is unaffected by streaming).
    """
    angular_velocity = np.asarray(omega, dtype=float).reshape(3)
    areas, normals, gradients = mesh.face_geometry()
    beta = np.cross(angular_velocity, normals)
    gdot = np.einsum("fij,fj->fi", gradients, beta)
    # ∫ λ_j dS = A/3 on a linear triangle, and ∇λ_i is constant on the facet.
    weight = -(areas / 3.0)[:, None] * gdot

    n = mesh.n_vertices
    advection = np.zeros((n, n))
    faces = mesh.faces
    for local_i in range(3):
        for local_j in range(3):
            np.add.at(advection, (faces[:, local_i], faces[:, local_j]), weight[:, local_i])

    advection = 0.5 * (advection - advection.T)
    ones = np.ones(n)
    residual = advection @ ones
    correction = (np.outer(residual, ones) - np.outer(ones, residual)) / n
    advection = advection - correction
    # Remove roundoff so the structural identities hold to a tight tolerance.
    advection = 0.5 * (advection - advection.T)
    return advection


def vertex_integrals(mass: np.ndarray) -> np.ndarray:
    """Row sums ``m_i = ∫ φ_i dS``. For a P1 function, ``∫ f = m · f``."""
    return mass @ np.ones(mass.shape[0])
