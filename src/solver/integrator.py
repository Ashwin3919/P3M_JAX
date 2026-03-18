from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from src.solver.state import State, Vector, HamiltonianSystem

Stepper = Callable[[State], State]
HaltingCondition = Callable[[State], bool]

def leap_frog(dt: float, h: HamiltonianSystem[jnp.ndarray], s: State) -> State:
    """Leap-frog integration step (Functional/Immutable)"""
    # 1. Half-step Kick (Momentum)
    momentum_half = s.momentum + (dt / 2) * h.momentumEquation(s)
    s_half_kicked = State(s.time, s.position, momentum_half)
    
    # 2. Full-step Drift (Position)
    new_position = s_half_kicked.position + dt * h.positionEquation(s_half_kicked)
    s_drifted = State(s.time + dt, new_position, momentum_half)
    
    # 3. Half-step Kick (Momentum)
    new_momentum = s_drifted.momentum + (dt / 2) * h.momentumEquation(s_drifted)
    
    return State(s.time + dt, new_position, new_momentum)

def leapfrog_step_scan(state: State, _, dt: float, system: HamiltonianSystem[jnp.ndarray]) -> Tuple[State, State]:
    """Scan-compatible leapfrog step"""
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
        Memory of stored trajectory scales as 1/save_every.
        Default 1 preserves original behaviour (every step saved).
    """
    step_fn = lambda s, x: leapfrog_step_scan(s, x, dt, system)

    if save_every == 1:
        return jax.lax.scan(step_fn, init, xs=None, length=n_steps)

    # Outer scan: each tick advances save_every steps, emits the final state.
    def chunk_fn(state, _):
        final, _ = jax.lax.scan(step_fn, state, xs=None, length=save_every)
        return final, final

    n_chunks = n_steps // save_every
    return jax.lax.scan(chunk_fn, init, xs=None, length=n_chunks)

def step_chunk(system: HamiltonianSystem[jnp.ndarray], state: State, dt: float, save_every: int) -> State:
    """Run exactly save_every leapfrog steps, return only the final state.

    JIT-compile this with partial(step_chunk, system, dt=dt, save_every=k)
    then call repeatedly in a Python loop for incremental I/O between chunks.
    """
    step_fn = lambda s, x: leapfrog_step_scan(s, x, dt, system)
    final, _ = jax.lax.scan(step_fn, state, xs=None, length=save_every)
    return final


def iterate_step(step: Stepper, halt: HaltingCondition, init: State) -> list[State]:
    """Fallback Python while-loop (Legacy)"""
    state = init
    states = []
    while not halt(state):
        states.append(state)
        state = step(state)
        if len(states) % 10 == 0:
            print(f"Time step {len(states)}, a = {state.time:.3f}")
    return states
