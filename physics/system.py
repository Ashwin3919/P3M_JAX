import jax.numpy as jnp
import matplotlib.pyplot as plt
from solver.state import State, HamiltonianSystem
from core.ops import md_cic_2d, Interp2D, gradient_2nd_order
from core.filters import Potential

class PoissonVlasov(HamiltonianSystem[jnp.ndarray]):
    def __init__(self, box, cosmology, particle_mass, live_plot=False):
        self.box = box
        self.cosmology = cosmology
        self.particle_mass = particle_mass
        self.delta = jnp.zeros(self.box.shape, dtype=jnp.float64)
        self.live_plot = live_plot

        # Initialize live plotting
        if live_plot:
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, box.N)
            self.ax.set_ylim(0, box.N)
            self.ax.set_title('Density Field Evolution')

    def positionEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        return s.momentum / (s.time ** 2 * da)

    def momentumEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        x_grid = s.position / self.box.res

        # Compute density field using JAX CIC
        self.delta = md_cic_2d(self.box.shape, x_grid)
        self.delta = self.delta * self.particle_mass
        self.delta = self.delta - 1.0

        # Live plotting
        if self.live_plot:
            self.ax.clear()
            rho_plot = self.delta + 1.0
            im = self.ax.imshow(jnp.log10(jnp.maximum(rho_plot, 0.1)).T,
                                extent=[0, self.box.N, 0, self.box.N],
                                origin='lower', cmap='hot', vmin=-0.5, vmax=1.0)
            self.ax.set_title(f'Log Density Field (a = {a:.3f})')

        # Solve Poisson equation
        delta_f = jnp.fft.fftn(self.delta)
        kernel = Potential()(self.box.K)
        phi = jnp.fft.ifftn(delta_f * kernel).real * self.cosmology.G / a

        # Compute acceleration
        acc_x = Interp2D(gradient_2nd_order(phi, 0))
        acc_y = Interp2D(gradient_2nd_order(phi, 1))
        acc = jnp.c_[acc_x(x_grid), acc_y(x_grid)] / self.box.res

        return -acc / da
