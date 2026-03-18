import jax.numpy as jnp
from src.solver.state import State, HamiltonianSystem
from src.core.ops import md_cic_nd, InterpND, gradient_2nd_order
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
        a  = s.time
        da = self.cosmology.da(a)
        x_grid = s.position / self.box.res

        delta = md_cic_nd(self.box.shape, x_grid) * self.particle_mass - 1.0
        phi   = jnp.fft.ifftn(jnp.fft.fftn(delta) * self.kernel).real * self.cosmology.G / a

        acc_components = [
            InterpND(gradient_2nd_order(phi, i))(x_grid)
            for i in range(self.box.dim)
        ]
        acc = jnp.stack(acc_components, axis=-1) / self.box.res

        return -acc / da
