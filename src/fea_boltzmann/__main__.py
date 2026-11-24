"""Run a short streaming-and-relaxation demo on the sphere."""

from __future__ import annotations

import numpy as np

from fea_boltzmann.fields import gaussian_blob
from fea_boltzmann.mesh import icosphere
from fea_boltzmann.solver import BoltzmannSolver


def main() -> None:
    mesh = icosphere(3)
    solver = BoltzmannSolver(mesh, omega=(0.0, 0.0, 1.0), tau=0.5)
    initial = gaussian_blob(mesh, center=(1.0, 0.0, 0.0), width=0.45)
    t_end = 1.0
    dt = 0.05
    final = solver.run(initial, t_end=t_end, dt=dt)

    mass0 = solver.integrate(initial)
    mass1 = solver.integrate(final)
    dev0 = solver.l2_deviation(initial)
    dev1 = solver.l2_deviation(final)
    moment = solver.moment(final)
    angle = np.arctan2(moment[1], moment[0])

    print(f"icosphere level 3: {mesh.n_vertices} vertices, {mesh.n_faces} faces")
    print(f"surface area {solver.area:.6f} (unit sphere 4π = {4 * np.pi:.6f})")
    print(f"mass {mass0:.6f} -> {mass1:.6f}  (relative drift {(mass1 - mass0) / mass0:.3e})")
    print(f"L2 deviation {dev0:.6f} -> {dev1:.6f}")
    print(f"continuous BGK factor exp(-t/τ) = {np.exp(-t_end / solver.tau):.6f}")
    print(f"observed deviation ratio {dev1 / dev0:.6f}")
    print(f"moment angle after streaming about z: {angle:.4f} rad")


if __name__ == "__main__":
    main()
