"""Conservation, relaxation, transport, and manufactured-solution tests."""

import numpy as np
import pytest

from fea_boltzmann.fields import gaussian_blob, manufactured_source
from fea_boltzmann.mesh import icosphere
from fea_boltzmann.solver import BoltzmannSolver


def test_mass_matrix_integrates_the_constant_function():
    mesh = icosphere(2)
    solver = BoltzmannSolver(mesh, tau=1.0)
    ones = np.ones(solver.n)
    assert solver.integrate(ones) == pytest.approx(solver.area, abs=1e-12)
    assert solver.area == pytest.approx(mesh.area, abs=1e-12)
    assert ones @ solver.mass @ ones == pytest.approx(solver.area, abs=1e-12)


def test_streaming_is_skew_and_preserves_constants():
    solver = BoltzmannSolver(icosphere(2), tau=np.inf)
    ones = np.ones(solver.n)
    advection = solver.advection
    assert np.allclose(advection, -advection.T, atol=1e-12)
    assert np.allclose(advection @ ones, 0.0, atol=1e-12)
    assert np.allclose(advection.T @ ones, 0.0, atol=1e-12)


def test_uniform_distribution_is_a_steady_state():
    solver = BoltzmannSolver(icosphere(2), omega=(0.2, -0.4, 0.8), tau=0.3, absorption=0.0)
    ones = np.ones(solver.n)
    updated = solver.step(ones, dt=0.17)
    assert np.allclose(updated, ones, atol=1e-9)
    assert abs(solver.integrate(updated) - solver.integrate(ones)) < 1e-10


def test_mass_is_conserved_while_a_blob_streams_and_scatters():
    mesh = icosphere(2)
    solver = BoltzmannSolver(mesh, omega=(0.0, 0.0, 1.0), tau=0.4)
    initial = gaussian_blob(mesh, center=(0.0, 0.0, 1.0), width=0.5)
    final = solver.run(initial, t_end=0.8, dt=0.05)
    assert abs(solver.integrate(final) - solver.integrate(initial)) < 1e-9


def test_scattering_drives_the_distribution_toward_equilibrium():
    mesh = icosphere(2)
    tau = 0.5
    solver = BoltzmannSolver(mesh, omega=(0.0, 0.0, 1.0), tau=tau)
    initial = gaussian_blob(mesh, center=(1.0, 0.0, 0.0), width=0.45)
    t_end = 1.5
    final = solver.run(initial, t_end=t_end, dt=0.05)
    ratio = solver.l2_deviation(final) / solver.l2_deviation(initial)
    # Backward Euler on y' = -y/τ multiplies the deviation by (1 + dt/τ)^(-n).
    # Streaming is skew, so it does not feed the norm; allow modest geometric error.
    continuous = np.exp(-t_end / tau)
    assert ratio < 2.0 * continuous
    assert solver.l2_deviation(final) < solver.l2_deviation(initial)


def test_blob_is_carried_a_quarter_turn_by_rotation():
    mesh = icosphere(3)
    solver = BoltzmannSolver(mesh, omega=(0.0, 0.0, 1.0), tau=np.inf)
    initial = gaussian_blob(mesh, center=(1.0, 0.0, 0.0), width=0.4)
    quarter_turn = 0.5 * np.pi
    final = solver.run(initial, t_end=quarter_turn, dt=quarter_turn / 50.0)

    start = solver.moment(initial)
    end = solver.moment(final)
    # (m_x + i m_y) evolves as e^{i t} when β = e_z × x.
    rotated = np.array([-start[1], start[0]])
    assert abs(solver.integrate(final) - solver.integrate(initial)) < 1e-8
    assert np.linalg.norm(end[:2] - rotated) < 0.05 * np.linalg.norm(start[:2])
    assert end[1] > 0.0
    assert abs(end[0]) < 0.15 * abs(start[0])


def test_manufactured_solution_converges_under_refinement():
    errors = []
    for level in (1, 2, 3):
        mesh = icosphere(level)
        tau = 0.75
        solver = BoltzmannSolver(mesh, omega=(0.0, 0.0, 1.0), tau=tau)
        exact, source = manufactured_source(mesh, tau=tau)
        # The spherical mean of the coordinate function is zero. Pin the
        # discrete solution to the mean of the interpolant so the comparison
        # is not polluted by the kernel of the collision operator.
        computed = solver.solve_stationary(source, mean=solver.mean(exact))
        error = solver.l2_norm(computed - exact) / solver.l2_norm(exact)
        errors.append(error)

    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))
    # P1 elements on a faceted sphere are second-order in L2. Level 3
    # (642 vertices) should already be well under a few percent.
    assert errors[-1] < 0.05
    ratio = np.log(errors[-2] / errors[-1]) / np.log(2.0)
    assert ratio > 1.4


def test_absorption_balances_a_uniform_source_of_particles():
    # With absorption and no scattering, σ_a f = q. A constant source
    # q = σ_a is balanced by the constant distribution f = 1.
    mesh = icosphere(2)
    absorption = 1.5
    solver = BoltzmannSolver(mesh, tau=np.inf, absorption=absorption)
    source = np.full(solver.n, absorption)
    computed = solver.solve_stationary(source)
    assert np.allclose(computed, np.ones(solver.n), atol=1e-8)
