"""Transport a localized distribution a quarter turn around the sphere and plot it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from fea_boltzmann.fields import gaussian_blob
from fea_boltzmann.mesh import icosphere
from fea_boltzmann.solver import BoltzmannSolver


def _scatter(ax: plt.Axes, vertices: np.ndarray, values: np.ndarray, title: str) -> None:
    colors = values - values.min()
    span = colors.max()
    if span > 0.0:
        colors = colors / span
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        c=colors,
        cmap="inferno",
        s=8,
        depthshade=False,
    )
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=30)


def main() -> None:
    mesh = icosphere(3)
    # No scattering: the blob should ride the rotation β = e_z × x.
    solver = BoltzmannSolver(mesh, omega=(0.0, 0.0, 1.0), tau=np.inf)
    initial = gaussian_blob(mesh, center=(1.0, 0.0, 0.0), width=0.35)
    quarter_turn = 0.5 * np.pi
    dt = quarter_turn / 40.0
    final = solver.run(initial, t_end=quarter_turn, dt=dt)

    start = solver.moment(initial)
    end = solver.moment(final)
    print(f"mass drift {solver.integrate(final) - solver.integrate(initial):.3e}")
    print(f"moment {start} -> {end}")
    print(f"moment angle {np.arctan2(end[1], end[0]):.4f} rad (expected {quarter_turn:.4f})")

    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    _scatter(fig.add_subplot(121, projection="3d"), mesh.vertices, initial, "t = 0")
    _scatter(fig.add_subplot(122, projection="3d"), mesh.vertices, final, "t = π/2")
    fig.tight_layout()
    path = output / "blob_transport.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
