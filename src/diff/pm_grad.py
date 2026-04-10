"""
Differentiable PM simulation.

The full PM pipeline is differentiable end-to-end:

    init_pos  ──►  CIC deposit  ──►  FFT Poisson  ──►  FD gradient
                   (scatter+weights)   (ifftn)           (jnp.roll)
                       ▼
                  interpolate  ──►  KDK leapfrog  ──►  final state
                  (gather)           (lax.scan)

Every op above has a JAX-defined VJP, so jax.grad flows through the
entire chain.  Use pm_rollout as the forward function and wrap it with
jax.value_and_grad (or jax.grad) to get gradients w.r.t. initial
conditions.
"""

import jax
import jax.numpy as jnp
from functools import partial

from src.solver.state import State
from src.solver.integrator import leap_frog, step_chunk


# ---------------------------------------------------------------------------
# Core differentiable rollout
# ---------------------------------------------------------------------------

def pm_rollout(
    init_pos: jnp.ndarray,
    init_mom: jnp.ndarray,
    a_init: float,
    n_steps: int,
    dt: float,
    system,
    *,
    checkpoint: bool = False,
    return_trajectory: bool = False,
):
    """Differentiable fixed-step PM rollout via jax.lax.scan.

    Fully differentiable w.r.t. init_pos and init_mom.
    system must use solver='pm' (PP path is not differentiable).

    Parameters
    ----------
    init_pos         : (n_particles, dim) float array — initial comoving positions
    init_mom         : (n_particles, dim) float array — initial canonical momenta
    a_init           : float — initial scale factor
    n_steps          : int  — number of fixed leapfrog steps
    dt               : float — step size in scale factor a
    system           : PoissonVlasov with solver='pm'
    checkpoint       : bool — apply jax.checkpoint per step (saves memory, 2x compute)
    return_trajectory: bool — if True also return stacked intermediate states

    Returns
    -------
    final_state : State
    trajectory  : State with leading axis n_steps  (only when return_trajectory=True)
    """
    # (Removed block on p3m — now supported via safely differentiable _pp_force)
    # The short-range PP path uses Morton sorting, but JAX correctly tracks 
    # gradients through the sorted indices via standard array indexing.
    
    init_state = State(
        jnp.asarray(a_init, dtype=init_pos.dtype),
        init_pos,
        init_mom,
    )

    if return_trajectory:
        # Trajectory case: must emit one state per step, so we keep the scan
        # here and apply checkpoint per-step manually.
        def _step(state, _):
            new_state = leap_frog(dt, system, state)
            return new_state, new_state

        step_fn = jax.checkpoint(_step) if checkpoint else _step
        final_state, traj = jax.lax.scan(step_fn, init_state, xs=None, length=n_steps)
        return final_state, traj

    # Non-trajectory case: delegate to step_chunk to avoid duplicating scan logic.
    # step_chunk already handles checkpoint and lax.scan over leap_frog.
    return step_chunk(system, init_state, dt, n_steps, checkpoint=checkpoint)


# ---------------------------------------------------------------------------
# Loss function factory
# ---------------------------------------------------------------------------

def make_loss_fn(system, a_init: float, n_steps: int, dt: float,
                 observable=None, checkpoint: bool = False):
    """Return a scalar loss function differentiable w.r.t. initial conditions.

    Parameters
    ----------
    system     : PoissonVlasov (pm mode)
    a_init     : float — initial scale factor
    n_steps    : int
    dt         : float
    observable : callable(State) -> scalar, or None.
                 Defaults to mean squared final position.
    checkpoint : bool — use gradient checkpointing

    Returns
    -------
    loss_fn(init_pos, init_mom) -> scalar

    Example
    -------
        loss_fn = make_loss_fn(system, a_init=0.1, n_steps=50, dt=0.01)

        # Gradient w.r.t. both initial positions and momenta
        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
        loss, (dpos, dmom) = grad_fn(init_pos, init_mom)

        # Gradient w.r.t. initial positions only
        grad_pos = jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)
    """
    if observable is None:
        # Default: mean squared displacement from origin — replace with your own.
        observable = lambda s: jnp.mean(s.position ** 2)

    def loss_fn(init_pos, init_mom):
        final = pm_rollout(
            init_pos, init_mom, a_init, n_steps, dt, system,
            checkpoint=checkpoint,
        )
        return observable(final)

    return loss_fn


# ---------------------------------------------------------------------------
# Cosmological-parameter gradient helper
# ---------------------------------------------------------------------------

def pm_rollout_cosmo(
    init_pos: jnp.ndarray,
    init_mom: jnp.ndarray,
    OmegaM: jnp.ndarray,
    OmegaL: jnp.ndarray,
    a_init: float,
    n_steps: int,
    dt: float,
    system,
    *,
    checkpoint: bool = False,
):
    """Differentiable rollout w.r.t. cosmological parameters OmegaM, OmegaL.

    Builds a fresh Cosmology and PoissonVlasov from the JAX-array parameters
    so that jax.grad(argnums=(2, 3)) gives dL/d(OmegaM) and dL/d(OmegaL).
    The original system object is never mutated.

    The Poisson kernel depends only on box geometry, not on OmegaM/OmegaL,
    so the rebuild is numerically equivalent to the original system.

    Note: system.cosmology.H0 is treated as a fixed Python float.
    To differentiate w.r.t. H0 as well, add it as an argument in the same way.
    """
    from src.physics.cosmology import Cosmology
    from src.physics.system import PoissonVlasov

    cosmo = Cosmology(H0=system.cosmology.H0, OmegaM=OmegaM, OmegaL=OmegaL)
    override_system = PoissonVlasov(
        system.box, cosmo, system.particle_mass,
        solver=system.solver, assignment=system.assignment,
        pp_window=system.pp_window, pp_softening=system.pp_softening,
        pp_cutoff=system.pp_cutoff,
    )
    return pm_rollout(
        init_pos, init_mom, a_init, n_steps, dt, override_system,
        checkpoint=checkpoint,
    )
