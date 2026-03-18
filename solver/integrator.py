from typing import Callable
import jax.numpy as jnp
from solver.state import State, Vector, HamiltonianSystem
from IPython.display import display, clear_output

Stepper = Callable[[State[Vector]], State[Vector]]
HaltingCondition = Callable[[State[Vector]], bool]

def leap_frog(dt: float, h: HamiltonianSystem[Vector], s: State[Vector]) -> State[Vector]:
    """Leap-frog integration step"""
    return s.kick(dt, h).wait(dt / 2).drift(dt, h).wait(dt / 2)

def iterate_step(step: Stepper, halt: HaltingCondition, init: State[Vector]) -> list[State[Vector]]:
    """Iterate simulation steps until halting condition"""
    state = init
    states = []
    while not halt(state):
        # We store a snapshot of the current state.
        # Use jnp.array to ensure we are copying the data, not just the reference.
        states.append(State(state.time, jnp.array(state.position), jnp.array(state.momentum)))
        state = step(state)
        
        # Live plot update every 10 steps
        if state.live_plot and len(states) % 10 == 0:
            clear_output(wait=True)
            display(state.fig)
        if len(states) % 10 == 0:
            print(f"Time step {len(states)}, a = {state.time:.3f}")
    return states
