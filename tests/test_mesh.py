"""Geometry of the icosphere."""

import numpy as np

from fea_boltzmann.mesh import icosphere


def test_level_zero_counts_and_outward_normals():
    mesh = icosphere(0)
    assert mesh.n_vertices == 12
    assert mesh.n_faces == 20
    areas, normals, gradients = mesh.face_geometry()
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    assert np.all(areas > 0.0)
    assert np.all(np.sum(normals * centroids, axis=1) > 0.0)
    assert np.allclose(gradients.sum(axis=1), 0.0, atol=1e-12)


def test_vertices_lie_on_the_unit_sphere():
    mesh = icosphere(2)
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-12)


def test_area_approaches_the_unit_sphere():
    areas = [icosphere(level).area for level in range(5)]
    target = 4.0 * np.pi
    errors = [abs(area - target) for area in areas]
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))
    assert errors[-1] < 0.05


def test_refinement_quadruples_faces():
    coarse = icosphere(1)
    fine = icosphere(2)
    assert fine.n_faces == 4 * coarse.n_faces
    assert fine.n_vertices > coarse.n_vertices
