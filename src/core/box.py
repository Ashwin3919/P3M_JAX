"""Simulation box geometry: grid shape, physical size, and pre-computed wave-number arrays."""
import jax.numpy as jnp
from functools import reduce

def _wave_number(s):
    """Return integer wave-number indices for an N-point FFT grid.

    Applies the standard FFT frequency convention: indices > N/2 are
    folded back to negative frequencies (i → i - N), matching the output
    of jnp.fft.fftn.

    Parameters
    ----------
    s : tuple of int — grid shape, e.g. (N, N) or (N, N, N)

    Returns
    -------
    Array of shape (dim, *s) with integer wave-number indices.
    """
    N = s[0]
    i = jnp.indices(s)
    return jnp.where(i > N / 2, i - N, i)

class Box:
    """Simulation box with periodic boundary conditions.

    Parameters
    ----------
    dim : int   — spatial dimension (2 or 3)
    N   : int   — number of grid cells per side
    L   : float — physical box side length [Mpc/h]

    Attributes
    ----------
    res   : float          — cell size Δx = L / N  [Mpc/h]
    shape : tuple of int   — grid shape  (N,)*dim
    size  : int            — total number of cells  N**dim
    K     : (dim, *shape)  — physical wave-number array [h/Mpc]
    k     : (*shape,)      — wave-number magnitude |K|  [h/Mpc]
    k_max : float          — Nyquist frequency  π N / L  [h/Mpc]
    k_min : float          — fundamental mode   2π / L   [h/Mpc]
    """
    def __init__(self, dim, N, L):
        self.N = N
        self.L = L
        self.res = L / N
        self.dim = dim

        self.shape = (self.N,) * dim
        self.size = reduce(lambda x, y: x * y, self.shape)

        self.K = _wave_number(self.shape) * 2 * jnp.pi / self.L
        self.k = jnp.sqrt((self.K ** 2).sum(axis=0))

        self.k_max = N * jnp.pi / L     # Nyquist frequency
        self.k_min = 2 * jnp.pi / L    # fundamental mode
