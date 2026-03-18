from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import jax.numpy as jnp

class VectorABC(ABC):
    @abstractmethod
    def __add__(self, other): raise NotImplementedError
    @abstractmethod
    def __rmul__(self, other): raise NotImplementedError

VectorABC.register(jnp.ndarray)
Vector = TypeVar("Vector", bound=VectorABC)

@dataclass
class State(Generic[Vector]):
    time: float
    position: Vector
    momentum: Vector
    live_plot: bool = False
    fig: any = None
    ax: any = None

    def kick(self, dt: float, h: 'HamiltonianSystem[Vector]') -> 'State[Vector]':
        self.momentum = self.momentum + dt * h.momentumEquation(self)
        return self

    def drift(self, dt: float, h: 'HamiltonianSystem[Vector]') -> 'State[Vector]':
        self.position = self.position + dt * h.positionEquation(self)
        return self

    def wait(self, dt: float) -> 'State[Vector]':
        self.time += dt
        return self

class HamiltonianSystem(ABC, Generic[Vector]):
    @abstractmethod
    def positionEquation(self, s: State[Vector]) -> Vector: raise NotImplementedError
    @abstractmethod
    def momentumEquation(self, s: State[Vector]) -> Vector: raise NotImplementedError
