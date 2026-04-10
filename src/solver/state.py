"""Core state types for the Hamiltonian N-body integrator.

State is a NamedTuple so JAX treats it as a pytree automatically:
fields are flattened/unflattened without any custom `tree_flatten` logic.
HamiltonianSystem is the abstract interface that every physics system must implement.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, NamedTuple
import jax.numpy as jnp

class VectorABC(ABC):
    @abstractmethod
    def __add__(self, other): raise NotImplementedError
    @abstractmethod
    def __rmul__(self, other): raise NotImplementedError

VectorABC.register(jnp.ndarray)
Vector = TypeVar("Vector", bound=VectorABC)

class State(NamedTuple):
    """Immutable simulation state at a single instant.

    Attributes
    ----------
    time     : float scalar  — current scale factor a
    position : (N, dim)      — comoving particle positions  [Mpc/h]
    momentum : (N, dim)      — canonical momenta  p = a² ẋ  [Mpc/h]
    """
    time: float
    position: jnp.ndarray
    momentum: jnp.ndarray

class HamiltonianSystem(ABC, Generic[Vector]):
    """Abstract interface for Hamiltonian N-body systems.

    Subclasses must implement:
      positionEquation(s)  — dq/da, the drift term
      momentumEquation(s)  — dp/da, the kick term (gravitational acceleration)
    """
    @abstractmethod
    def positionEquation(self, s: State) -> Vector: ...
    @abstractmethod
    def momentumEquation(self, s: State) -> Vector: ...
