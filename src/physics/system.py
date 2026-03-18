import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.solver.state import State, HamiltonianSystem
from src.core.ops import md_cic_2d, Interp2D, gradient_2nd_order
from src.core.filters import Potential
class PoissonVlasov(HamiltonianSystem[jnp.ndarray]):
    def __init__(self, box, cosmology, particle_mass):
        self.box = box
        self.cosmology = cosmology
        self.particle_mass = particle_mass
        self.kernel = Potential()(self.box.K)

    def positionEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        return s.momentum / (s.time ** 2 * da)

    def momentumEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        x_grid = s.position / self.box.res

        # Compute density field using JAX CIC (local variable)
        delta = md_cic_2d(self.box.shape, x_grid)
        delta = delta * self.particle_mass
        delta = delta - 1.0

        # Solve Poisson equation using precomputed kernel
        delta_f = jnp.fft.fftn(delta)
        phi = jnp.fft.ifftn(delta_f * self.kernel).real * self.cosmology.G / a

        acc_x = Interp2D(gradient_2nd_order(phi, 0))
        acc_y = Interp2D(gradient_2nd_order(phi, 1))
        acc = jnp.c_[acc_x(x_grid), acc_y(x_grid)] / self.box.res

        return -acc / da
