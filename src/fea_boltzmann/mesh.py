"""Icosahedral triangulation of the unit sphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Twelve vertices of a regular icosahedron, indexed so the face table below
# is the standard outward triangulation.
_GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0
_ICOSAHEDRON_VERTICES = np.array(
    [
        [-1.0, _GOLDEN, 0.0],
        [1.0, _GOLDEN, 0.0],
        [-1.0, -_GOLDEN, 0.0],
        [1.0, -_GOLDEN, 0.0],
        [0.0, -1.0, _GOLDEN],
        [0.0, 1.0, _GOLDEN],
        [0.0, -1.0, -_GOLDEN],
        [0.0, 1.0, -_GOLDEN],
        [_GOLDEN, 0.0, -1.0],
        [_GOLDEN, 0.0, 1.0],
        [-_GOLDEN, 0.0, -1.0],
        [-_GOLDEN, 0.0, 1.0],
    ],
    dtype=float,
)
_ICOSAHEDRON_VERTICES /= np.linalg.norm(_ICOSAHEDRON_VERTICES, axis=1)[:, None]

_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=int,
)


@dataclass(frozen=True)
class SphereMesh:
    """Piecewise-linear approximation of the unit sphere.

    ``vertices`` has shape ``(n, 3)`` and lies on the unit sphere.
    ``faces`` has shape ``(m, 3)`` and stores outward-oriented triangles.
    Geometry derived from the flat facets (areas, normals, P1 gradients)
    is the integration domain used by the surface finite-element method.
    """

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        if self.faces.size and (
            self.faces.min() < 0 or self.faces.max() >= len(self.vertices)
        ):
            raise ValueError("face indices out of range")

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def face_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(areas, normals, gradients)`` for every facet.

        ``gradients[t, i]`` is the in-plane gradient of the barycentric
        coordinate of local vertex ``i`` on facet ``t``. Each gradient is a
        vector in R^3 tangent to that facet, with length ``1 / height``.
        """
        triangles = self.vertices[self.faces]
        x0 = triangles[:, 0]
        x1 = triangles[:, 1]
        x2 = triangles[:, 2]
        normal_vec = np.cross(x1 - x0, x2 - x0)
        areas = 0.5 * np.linalg.norm(normal_vec, axis=1)
        if np.any(areas <= 0.0):
            raise ValueError("mesh contains a degenerate face")
        normals = normal_vec / (2.0 * areas)[:, None]
        gradients = np.empty((self.n_faces, 3, 3))
        # ∇λ_i = n × (x_k − x_j) / (2A), (i, j, k) cyclic.
        gradients[:, 0] = np.cross(normals, x2 - x1) / (2.0 * areas)[:, None]
        gradients[:, 1] = np.cross(normals, x0 - x2) / (2.0 * areas)[:, None]
        gradients[:, 2] = np.cross(normals, x1 - x0) / (2.0 * areas)[:, None]
        return areas, normals, gradients

    @property
    def area(self) -> float:
        areas, _, _ = self.face_geometry()
        return float(areas.sum())


def _orient_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centroids = triangles.mean(axis=1)
    if float(np.sum(normals * centroids)) < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
    return faces


def _subdivide(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split each triangle into four and project new vertices onto the sphere."""
    vertex_list = [vertices[i].copy() for i in range(len(vertices))]
    midpoint: dict[tuple[int, int], int] = {}

    def middle(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        cached = midpoint.get(key)
        if cached is not None:
            return cached
        point = vertices[i] + vertices[j]
        point /= np.linalg.norm(point)
        index = len(vertex_list)
        vertex_list.append(point)
        midpoint[key] = index
        return index

    new_faces = np.empty((len(faces) * 4, 3), dtype=int)
    for t, (i, j, k) in enumerate(faces):
        a = middle(int(i), int(j))
        b = middle(int(j), int(k))
        c = middle(int(k), int(i))
        base = t * 4
        new_faces[base] = (i, a, c)
        new_faces[base + 1] = (j, b, a)
        new_faces[base + 2] = (k, c, b)
        new_faces[base + 3] = (a, b, c)
    return np.asarray(vertex_list, dtype=float), new_faces


def icosphere(subdivisions: int) -> SphereMesh:
    """Build an icosphere with ``subdivisions`` refinement steps.

    Level 0 is the regular icosahedron. Each additional level quadruples the
    number of faces and projects edge midpoints onto the unit sphere.
    """
    if not isinstance(subdivisions, (int, np.integer)) or subdivisions < 0:
        raise ValueError("subdivisions must be a nonnegative integer")
    vertices = _ICOSAHEDRON_VERTICES.copy()
    faces = _orient_outward(vertices, _ICOSAHEDRON_FACES.copy())
    for _ in range(int(subdivisions)):
        vertices, faces = _subdivide(vertices, faces)
    return SphereMesh(vertices=vertices, faces=faces)
