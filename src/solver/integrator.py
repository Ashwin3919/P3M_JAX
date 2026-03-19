from typing import Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp
from src.solver.state import State, Vector, HamiltonianSystem

Stepper = Callable[[State], State]
HaltingCondition = Callable[[State], bool]

def leap_frog(dt: float, h: HamiltonianSystem[jnp.ndarray], s: State) -> State:
    """KDK Leap-frog integration step (immutable State)."""
    # 1. Half-step Kick
    momentum_half = s.momentum + (dt / 2) * h.momentumEquation(s)
    s_half_kicked = State(s.time, s.position, momentum_half)

    # 2. Full-step Drift
    new_position = s_half_kicked.position + dt * h.positionEquation(s_half_kicked)
    s_drifted = State(s.time + dt, new_position, momentum_half)

    # 3. Half-step Kick
    new_momentum = s_drifted.momentum + (dt / 2) * h.momentumEquation(s_drifted)

    return State(s.time + dt, new_position, new_momentum)


def leapfrog_step_scan(state: State, _, dt: float, system: HamiltonianSystem[jnp.ndarray]) -> Tuple[State, State]:
    """Scan-compatible leapfrog step."""
    new_state = leap_frog(dt, system, state)
    return new_state, new_state


def iterate_step_scan(
    system: HamiltonianSystem[jnp.ndarray],
    init: State,
    dt: float,
    n_steps: int,
    save_every: int = 1,
) -> Tuple[State, State]:
    """Iterate simulation steps using jax.lax.scan.

    Parameters
    ----------
    save_every : int
        Emit one snapshot every this many leapfrog steps.
        n_steps must be divisible by save_every.
    """
    step_fn = lambda s, x: leapfrog_step_scan(s, x, dt, system)

    if save_every == 1:
        return jax.lax.scan(step_fn, init, xs=None, length=n_steps)

    def chunk_fn(state, _):
        final, _ = jax.lax.scan(step_fn, state, xs=None, length=save_every)
        return final, final

    n_chunks = n_steps // save_every
    return jax.lax.scan(chunk_fn, init, xs=None, length=n_chunks)


def step_chunk(system: HamiltonianSystem[jnp.ndarray], state: State, dt: float, save_every: int) -> State:
    """Run exactly save_every leapfrog steps with fixed dt, return final state.

    JIT-compile with partial(step_chunk, system, dt=dt, save_every=k).
    """
    step_fn = lambda s, x: leapfrog_step_scan(s, x, dt, system)
    final, _ = jax.lax.scan(step_fn, state, xs=None, length=save_every)
    return final


# ---------------------------------------------------------------------------
# Phase 7 — Adaptive time-stepping
# ---------------------------------------------------------------------------

def compute_dt(state: State, cosmology, C_cfl: float, eps: float,
               dt_min: float, dt_max: float) -> jnp.ndarray:
    """CFL time-step estimate based on maximum particle drift rate.

    dt = C_cfl * eps / v_max   clipped to [dt_min, dt_max]

    where v_i = |p_i| / (a² H(a)) is the comoving drift speed (from
    positionEquation). eps (softening length) sets the resolution scale.
    """
    a  = state.time
    da = cosmology.da(a)
    # Drift rate: dx/da = p / (a² H(a))
    v_mag = jnp.linalg.norm(state.momentum, axis=-1) / (a ** 2 * da + 1e-20)
    v_max = jnp.max(v_mag)
    dt = C_cfl * eps / (v_max + 1e-10)
    return jnp.clip(dt, dt_min, dt_max)


def step_chunk_adaptive(
    system: HamiltonianSystem[jnp.ndarray],
    state: State,
    a_target: float,
    C_cfl: float,
    eps: float,
    dt_min: float,
    dt_max: float,
) -> State:
    """Advance state to a_target using adaptive dt via lax.while_loop.

    dt is recomputed from the CFL condition after every leapfrog step.
    The final sub-step is clamped so we land exactly on a_target.

    JIT-compile with partial(step_chunk_adaptive, system,
                             C_cfl=C_cfl, eps=eps, dt_min=dt_min, dt_max=dt_max).
    Then call with (state, a_target) in the Python chunk loop.
    """
    dt_min_arr = jnp.array(dt_min)
    a_target_arr = jnp.array(a_target)

    def cond_fn(carry):
        s, _ = carry
        return s.time < a_target_arr

    def body_fn(carry):
        s, _ = carry
        dt_cfl = compute_dt(s, system.cosmology, C_cfl, eps, dt_min, dt_max)
        # Clamp last step so we land exactly on a_target
        dt = jnp.minimum(dt_cfl, a_target_arr - s.time)
        new_s = leap_frog(dt, system, s)
        return new_s, dt

    final_state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, dt_min_arr))
    return final_state


def iterate_step(step: Stepper, halt: HaltingCondition, init: State) -> list[State]:
    """Fallback Python while-loop (Legacy)."""
    state = init
    states = []
    while not halt(state):
        states.append(state)
        state = step(state)
        if len(states) % 10 == 0:
            print(f"Time step {len(states)}, a = {state.time:.3f}")
    return states
