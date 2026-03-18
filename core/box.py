import jax.numpy as jnp
from functools import reduce

def _wave_number(s):
    """Generate wave numbers for FFT"""
    N = s[0]
    i = jnp.indices(s)
    return jnp.where(i > N / 2, i - N, i)

class Box:
    """Simulation box with periodic boundary conditions"""
    def __init__(self, dim, N, L):
        self.N = N
        self.L = L
        self.res = L / N
        self.dim = dim

        self.shape = (self.N,) * dim
        self.size = reduce(lambda x, y: x * y, self.shape)

        self.K = _wave_number(self.shape) * 2 * jnp.pi / self.L
        self.k = jnp.sqrt((self.K ** 2).sum(axis=0))

        self.k_max = N * jnp.pi / L
        self.k_min = 2 * jnp.pi / L
