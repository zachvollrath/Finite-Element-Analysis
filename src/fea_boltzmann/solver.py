"""Implicit P1 solver for the linear Boltzmann equation on the sphere."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from fea_boltzmann.mesh import SphereMesh
from fea_boltzmann.operators import assemble_advection, assemble_mass, vertex_integrals


class BoltzmannSolver:
    """Surface finite-element discretization of the linear Boltzmann equation.

    The unknown ``f`` is the particle distribution on the unit sphere ``Γ``.
    With isotropic scattering it satisfies

        ∂f/∂t + div_Γ(β f) + σ_a f = (1/τ) (f̄ − f) + q,

    where ``β(x) = ω × x`` is rigid-rotation streaming, ``σ_a`` is absorption,
    ``τ`` is the scattering time, ``q`` is a source, and

        f̄ = (1/|Γ|) ∫_Γ f dS

    is the isotropic equilibrium that conserves particles. The spatial
    unknown is a continuous piecewise-linear function on an icosphere.
    Time stepping is backward Euler. The streaming matrix is built so that
    a uniform distribution is a discrete steady state and streaming is
    skew-symmetric, so it neither creates mass nor changes the L² norm.
    """

    def __init__(
        self,
        mesh: SphereMesh,
        omega: np.ndarray = (0.0, 0.0, 1.0),
        tau: float = 1.0,
        absorption: float = 0.0,
    ) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be positive; use np.inf to disable scattering")
        if absorption < 0.0:
            raise ValueError("absorption must be nonnegative")
        self.mesh = mesh
        self.omega = np.asarray(omega, dtype=float).reshape(3).copy()
        self.tau = float(tau)
        self.absorption = float(absorption)
        self.mass = assemble_mass(mesh)
        self.advection = assemble_advection(mesh, self.omega)
        self.vertex_mass = vertex_integrals(self.mass)
        self.area = float(self.vertex_mass.sum())
        self._lu: tuple[np.ndarray, np.ndarray] | None = None
        self._lu_dt: float | None = None

    @property
    def n(self) -> int:
        return self.mesh.n_vertices

    def integrate(self, f: np.ndarray) -> float:
        """Return ``∫ f_h dS = m · f``."""
        values = self._as_dofs(f)
        return float(self.vertex_mass @ values)

    def mean(self, f: np.ndarray) -> float:
        """Return the spherical mean ``(1/|Γ|) ∫ f dS``."""
        return self.integrate(f) / self.area

    def equilibrium(self, f: np.ndarray) -> np.ndarray:
        """Nodal values of the isotropic equilibrium with the same mass as ``f``."""
        return np.full(self.n, self.mean(f))

    def l2_norm(self, f: np.ndarray) -> float:
        """Return ``(∫ f_h² dS)^(1/2)``."""
        values = self._as_dofs(f)
        return float(np.sqrt(values @ self.mass @ values))

    def l2_deviation(self, f: np.ndarray) -> float:
        """L² distance from ``f`` to its isotropic equilibrium."""
        values = self._as_dofs(f)
        return self.l2_norm(values - self.equilibrium(values))

    def moment(self, f: np.ndarray) -> np.ndarray:
        """Return ``∫ f_h x_h dS``, using the P1 interpolant of the coordinates."""
        values = self._as_dofs(f)
        return self.mesh.vertices.T @ (self.mass @ values)

    def step(self, f: np.ndarray, dt: float, source: np.ndarray | None = None) -> np.ndarray:
        """Advance one backward-Euler step of length ``dt``.

        ``source`` holds nodal values of a P1 source. A missing source is zero.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        values = self._as_dofs(f)
        rhs = self.mass @ values
        if source is not None:
            rhs = rhs + dt * (self.mass @ self._as_dofs(source))
        return lu_solve(self._factor(dt), rhs)

    def run(
        self,
        f: np.ndarray,
        t_end: float,
        dt: float,
        source: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate from ``t = 0`` to ``t_end`` with a fixed step ``dt``.

        The number of steps is ``round(t_end / dt)``. ``t_end`` must be a
        positive multiple of ``dt`` within a small rounding tolerance.
        """
        if t_end < 0.0:
            raise ValueError("t_end must be nonnegative")
        if t_end == 0.0:
            return self._as_dofs(f).copy()
        steps = int(round(t_end / dt))
        if steps < 1 or abs(steps * dt - t_end) > 1e-8 * max(1.0, t_end):
            raise ValueError("t_end must be a positive integer multiple of dt")
        values = self._as_dofs(f).copy()
        factor = self._factor(dt)
        mass_source = None if source is None else self.mass @ self._as_dofs(source)
        for _ in range(steps):
            rhs = self.mass @ values
            if mass_source is not None:
                rhs = rhs + dt * mass_source
            values = lu_solve(factor, rhs)
        return values

    def solve_stationary(
        self,
        source: np.ndarray,
        mean: float | None = None,
    ) -> np.ndarray:
        """Solve the steady equation for a P1 source.

        With no absorption the operator annihilates constants, so ``mean``
        prescribes ``(1/|Γ|) ∫ f`` and defaults to 0. The source itself must
        have zero spherical mean, otherwise no steady state exists. With
        absorption the mean is fixed by the equation and ``mean`` must be
        left unset.
        """
        nodal_source = self._as_dofs(source)
        load = self.mass @ nodal_source
        mass_of_source = float(self.vertex_mass @ nodal_source)

        if self.absorption == 0.0:
            if not np.isfinite(self.tau):
                raise ValueError("pure streaming has no unique steady state")
            scale = 1.0 + self.l2_norm(nodal_source)
            if abs(mass_of_source) > 1e-8 * scale:
                raise ValueError("stationary source must have zero spherical mean")
            if mean is None:
                mean = 0.0
            prescribed = float(mean) * self.area
            matrix = np.zeros((self.n + 1, self.n + 1))
            matrix[: self.n, : self.n] = self._spatial_operator()
            matrix[: self.n, self.n] = self.vertex_mass
            matrix[self.n, : self.n] = self.vertex_mass
            rhs = np.zeros(self.n + 1)
            rhs[: self.n] = load
            rhs[self.n] = prescribed
            solution = np.linalg.solve(matrix, rhs)
            return solution[: self.n]

        if mean is not None:
            raise ValueError("mean is fixed by the equation when absorption is positive")
        return np.linalg.solve(self._spatial_operator(), load)

    def _spatial_operator(self) -> np.ndarray:
        scattering = 0.0 if not np.isfinite(self.tau) else 1.0 / self.tau
        operator = self.advection + (self.absorption + scattering) * self.mass
        if scattering:
            operator = operator - (scattering / self.area) * np.outer(
                self.vertex_mass, self.vertex_mass
            )
        return operator

    def _system_matrix(self, dt: float) -> np.ndarray:
        scattering = 0.0 if not np.isfinite(self.tau) else 1.0 / self.tau
        matrix = self.mass + dt * self.advection
        matrix = matrix + dt * (self.absorption + scattering) * self.mass
        if scattering:
            matrix = matrix - (dt * scattering / self.area) * np.outer(
                self.vertex_mass, self.vertex_mass
            )
        return matrix

    def _factor(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self._lu is None or self._lu_dt != dt:
            self._lu = lu_factor(self._system_matrix(dt))
            self._lu_dt = dt
        return self._lu

    def _as_dofs(self, f: np.ndarray) -> np.ndarray:
        values = np.asarray(f, dtype=float).reshape(-1)
        if values.shape != (self.n,):
            raise ValueError(f"expected {self.n} nodal values, got shape {values.shape}")
        return values
