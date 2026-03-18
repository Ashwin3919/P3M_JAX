import itertools
import jax
import jax.numpy as jnp
from functools import partial
from math import prod as math_prod
from typing import Tuple

@partial(jax.jit, static_argnums=(0,))
def md_cic_nd(shape, pos):
    """
    CIC mass deposition for arbitrary dimensions.

    shape is static so JAX can build zero arrays and unroll the corner
    loop at compile time (2^dim iterations).

    Parameters
    ----------
    shape : tuple of int  — grid shape, e.g. (N, N) or (N, N, N)
    pos   : (n_particles, dim) float array in grid units

    Returns
    -------
    density array with the given shape
    """
    dim = len(shape)
    N = jnp.array(shape)
    idx = jnp.floor(pos).astype(jnp.int32)   # (n_particles, dim)
    f   = pos - idx                            # fractional offsets

    strides = jnp.array([math_prod(shape[d + 1:]) for d in range(dim)])
    flat_tgt = jnp.zeros(math_prod(shape), dtype=pos.dtype)

    for corner in itertools.product([0, 1], repeat=dim):
        corner_arr = jnp.array(corner, dtype=jnp.int32)           # (dim,)
        weights = jnp.prod(
            jnp.where(jnp.array(corner) == 1, f, 1.0 - f), axis=-1  # (n_particles,)
        )
        flat_idx = (((idx + corner_arr) % N) * strides).sum(-1)   # (n_particles,)
        flat_tgt = flat_tgt.at[flat_idx].add(weights)

    return flat_tgt.reshape(shape)


class InterpND:
    """N-dimensional trilinear (CIC-dual) interpolation."""

    def __init__(self, data):
        self.data = data
        self.shape = jnp.array(data.shape)
        self.dim = data.ndim

    def __call__(self, pos):
        idx = jnp.floor(pos).astype(jnp.int32)   # (n, dim)
        f   = pos - idx

        result = jnp.zeros(pos.shape[0], dtype=self.data.dtype)
        for corner in itertools.product([0, 1], repeat=self.dim):
            corner_arr = jnp.array(corner, dtype=jnp.int32)
            weights = jnp.prod(
                jnp.where(jnp.array(corner) == 1, f, 1.0 - f), axis=-1
            )
            ci = (idx + corner_arr) % self.shape
            cell_vals = self.data[tuple(ci[:, d] for d in range(self.dim))]
            result = result + weights * cell_vals
        return result


# Backward-compatible aliases so existing imports and tests are unaffected
md_cic_2d = md_cic_nd
Interp2D  = InterpND


def gradient_2nd_order(F, i):
    """Second-order finite difference gradient along axis i."""
    return (1. / 12 * jnp.roll(F, 2, axis=i) - 2. / 3 * jnp.roll(F, 1, axis=i) +
            2. / 3 * jnp.roll(F, -1, axis=i) - 1. / 12 * jnp.roll(F, -2, axis=i))


def garfield(B, P, T_filt, seed=None):
    """Generate Gaussian random field."""
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(0)
    wn = jax.random.normal(key, shape=B.shape)
    f = jnp.fft.ifftn(jnp.fft.fftn(wn) * jnp.sqrt(P(B.K))).real
    return jnp.fft.ifftn(jnp.fft.fftn(f) * T_filt(B.K)).real
