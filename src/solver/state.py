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
    time: float
    position: jnp.ndarray
    momentum: jnp.ndarray

class HamiltonianSystem(ABC, Generic[Vector]):
    @abstractmethod
    def positionEquation(self, s: State) -> Vector: raise NotImplementedError
    @abstractmethod
    def momentumEquation(self, s: State) -> Vector: raise NotImplementedError
