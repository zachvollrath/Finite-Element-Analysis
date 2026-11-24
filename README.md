# Finite Element Analysis

P1 finite-element solver for the **linear Boltzmann equation** on the unit sphere. The unknown is a particle distribution `f` on `Γ = S²`. With rigid-rotation streaming and isotropic scattering it reads

```
∂f/∂t + div_Γ(β f) + σ_a f = (1/τ) (f̄ − f) + q
β(x) = ω × x
f̄ = (1/|Γ|) ∫_Γ f dS
```

`β` is tangent and divergence-free, so streaming moves particles without creating them. The collision term is the isotropic linear Boltzmann operator: it relaxes `f` to the uniform equilibrium `f̄` that carries the same particle mass. `σ_a` is absorption and `q` is an external source.

The sphere is approximated by an icosphere. `f` is continuous and piecewise linear on that triangulation. The streaming form is projected onto skew matrices that annihilate constants, so a uniform distribution is a discrete steady state and streaming does not change the L² norm. Time stepping is backward Euler.

This is the standard linear (BGK) collision model used for transport. It is not the full five-fold nonlinear Boltzmann collision integral.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
fea-boltzmann
python examples/transport_blob.py
pytest
```

`fea-boltzmann` streams a Gaussian blob about the z-axis while it scatters, and prints mass drift and the decay of the L² deviation from equilibrium. The example disables scattering and writes `examples/output/blob_transport.png` after a quarter turn.

## Use

```python
from fea_boltzmann import BoltzmannSolver, gaussian_blob, icosphere

mesh = icosphere(3)
solver = BoltzmannSolver(mesh, omega=(0.0, 0.0, 1.0), tau=0.5)
f = gaussian_blob(mesh, center=(1.0, 0.0, 0.0), width=0.4)
f = solver.run(f, t_end=1.0, dt=0.05)
```

`tau=np.inf` turns scattering off. `solve_stationary(source, mean=...)` solves the steady problem. Without absorption the source must have zero spherical mean, and `mean` pins the conserved mass.

## Tests

The tests check four structural facts:

- the faceted area approaches `4π`, with outward normals
- particle mass is conserved and a uniform state does not move
- scattering damps the L² deviation at the backward-Euler rate `e^{−t/τ}`
- the manufactured solution `f = x`, whose source is `q = −y + x/τ`, converges in L² under icosphere refinement
- a blob advected by `ω = e_z` rotates by a quarter turn

Matrices are dense. Icospheres through level 4 (2562 vertices) are a comfortable size; level 3 is what the demo uses.
