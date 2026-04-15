"""Integrator order-of-convergence tests.

Verifies that the KDK leap-frog integrator achieves 2nd-order global
convergence by comparing particle trajectories at multiple step sizes
against a fine-grid reference solution.

Run with:
    pytest tests/test_convergence.py -v -s
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from src.core.box import Box
from src.core.filters import Cutoff, Potential, Power_law, Scale
from src.core.ops import garfield
from src.physics.cosmology import EDS_PRESET
from src.physics.initial_conds import Zeldovich
from src.physics.system import PoissonVlasov
from src.solver.integrator import step_chunk
from src.solver.state import State


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(system, state_init, dt, n_steps):
    """JIT-compile and run exactly n_steps leapfrog steps with fixed dt."""
    fn = jax.jit(functools.partial(step_chunk, system, dt=dt, save_every=n_steps))
    return fn(state_init)


# ---------------------------------------------------------------------------
# 1. Global 2nd-order convergence of particle trajectories
# ---------------------------------------------------------------------------

def test_leapfrog_2nd_order_convergence():
    """KDK leap-frog converges at 2nd order: trajectory error ∝ dt².

    Setup
    -----
    2D EdS PM simulation, N=16, a: 0.1 → 0.3 (interval = 0.2 in scale factor).

    Method
    ------
    Run with dt ∈ {0.04, 0.02, 0.01} and a fine-grid reference dt_ref = 0.005.
    Error = mean L2 distance between final particle positions.
    Fit log-log slope → expect slope ≈ 2.
    """
    N, L = 16, 20.0
    bm   = Box(2, N, L)
    bf   = Box(2, N * 2, L)

    P   = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
    phi = garfield(bm, P, Potential(), seed=7) * 5.0
    za  = Zeldovich(bm, bf, EDS_PRESET, phi)

    a_start, a_end = 0.1, 0.3
    state_init = za.state(a_start)
    system     = PoissonVlasov(bf, EDS_PRESET, za.particle_mass)

    interval  = a_end - a_start
    dt_values = np.array([0.04, 0.02, 0.01])
    dt_ref    = 0.005          # fine-grid reference (8× finer than coarsest)

    # Reference solution
    ref = _run(system, state_init, dt_ref, round(interval / dt_ref))
    x_ref = np.array(ref.position)

    errors = []
    for dt in dt_values:
        final = _run(system, state_init, dt, round(interval / dt))
        err   = float(np.mean(np.linalg.norm(np.array(final.position) - x_ref, axis=-1)))
        errors.append(err)
        print(f"  dt={dt:.3f}  mean position error={err:.3e}")

    errors = np.array(errors)
    slope, _ = np.polyfit(np.log(dt_values), np.log(errors), 1)
    print(f"\n  Fitted convergence order: {slope:.3f}  (expected ≈ 2.0)")

    assert 1.5 < slope < 2.8, (
        f"Expected 2nd-order convergence (slope ≈ 2), got {slope:.3f}. "
        f"dt={dt_values}, errors={errors}"
    )


# ---------------------------------------------------------------------------
# 2. Total momentum conservation (Newton's 3rd law via FFT symmetry)
# ---------------------------------------------------------------------------

def test_total_momentum_conservation():
    """KDK leap-frog conserves total momentum to machine precision.

    In a periodic PM simulation the total force is exactly zero (the FFT-based
    Poisson solver is self-adjoint and the k=0 mode carries no force).  Starting
    from zero total momentum, sum(p) must remain zero after any number of steps.
    """
    N, L = 8, 10.0
    bm   = Box(2, N, L)
    bf   = Box(2, N * 2, L)

    P   = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
    phi = garfield(bm, P, Potential(), seed=3) * 3.0
    za  = Zeldovich(bm, bf, EDS_PRESET, phi)

    # Force zero initial total momentum
    state = za.state(0.05)
    state = State(state.time, state.position, jnp.zeros_like(state.momentum))

    system = PoissonVlasov(bf, EDS_PRESET, za.particle_mass)
    final  = _run(system, state, dt=0.01, n_steps=20)

    total_mom = jnp.abs(final.momentum.sum(axis=0))
    print(f"\n  |Σp| after 20 steps: {total_mom}")
    assert jnp.all(total_mom < 1e-8), (
        f"Total momentum not conserved: {total_mom}  (expect < 1e-8)"
    )


# ---------------------------------------------------------------------------
# 3. Energy error scales as dt² for a conservative test case
# ---------------------------------------------------------------------------

def test_position_error_halving():
    """Halving dt roughly quarters the position error (2nd-order scaling).

    Uses only two dt values (dt and dt/2) for a fast single-assertion check
    that is complementary to the full convergence test above.
    """
    N, L = 12, 15.0
    bm   = Box(2, N, L)
    bf   = Box(2, N * 2, L)

    P   = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
    phi = garfield(bm, P, Potential(), seed=11) * 4.0
    za  = Zeldovich(bm, bf, EDS_PRESET, phi)

    state_init = za.state(0.1)
    system     = PoissonVlasov(bf, EDS_PRESET, za.particle_mass)

    dt_coarse, dt_fine, dt_ref = 0.02, 0.01, 0.0025
    interval = 0.1     # a: 0.1 → 0.2

    ref    = _run(system, state_init, dt_ref,    round(interval / dt_ref))
    coarse = _run(system, state_init, dt_coarse, round(interval / dt_coarse))
    fine   = _run(system, state_init, dt_fine,   round(interval / dt_fine))

    x_ref    = np.array(ref.position)
    err_c = float(np.mean(np.linalg.norm(np.array(coarse.position) - x_ref, axis=-1)))
    err_f = float(np.mean(np.linalg.norm(np.array(fine.position)   - x_ref, axis=-1)))

    ratio = err_c / (err_f + 1e-30)
    print(f"\n  err(dt=0.02)={err_c:.3e}  err(dt=0.01)={err_f:.3e}  ratio={ratio:.2f}")
    print(f"  Expected ratio ≈ 4 (2nd order: halving dt → quarter error)")

    # For a 2nd-order method, ratio should be close to 4.  Allow 2–8 due to
    # nonlinearity and the fact that the reference is not the true solution.
    assert 2.0 < ratio < 8.0, (
        f"Error ratio {ratio:.2f} not consistent with 2nd-order convergence"
    )
