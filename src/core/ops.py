import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

@partial(jax.jit, static_argnums=(0,))
def md_cic_2d(shape: Tuple[int, int], pos: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-based cloud-in-cell mass deposition.
    shape is static so JAX can build the zero array at trace time.
    Returns the deposited density array.
    """
    N0, N1 = shape
    idx0 = jnp.floor(pos[:, 0]).astype(jnp.int32)
    idx1 = jnp.floor(pos[:, 1]).astype(jnp.int32)
    f0 = pos[:, 0] - idx0
    f1 = pos[:, 1] - idx1

    def flat(i, j):
        return (i % N0) * N1 + (j % N1)

    weights_00 = (1 - f0) * (1 - f1)
    weights_10 = f0 * (1 - f1)
    weights_01 = (1 - f0) * f1
    weights_11 = f0 * f1

    flat_tgt = jnp.zeros(N0 * N1, dtype=pos.dtype)
    flat_tgt = flat_tgt.at[flat(idx0,     idx1    )].add(weights_00)
    flat_tgt = flat_tgt.at[flat(idx0 + 1, idx1    )].add(weights_10)
    flat_tgt = flat_tgt.at[flat(idx0,     idx1 + 1)].add(weights_01)
    flat_tgt = flat_tgt.at[flat(idx0 + 1, idx1 + 1)].add(weights_11)

    return flat_tgt.reshape(shape)

class Interp2D:
    """Bilinear interpolation"""
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        N0, N1 = self.shape
        X1 = jnp.floor(x).astype(jnp.int32) % jnp.array([N0, N1])
        X2 = jnp.ceil(x).astype(jnp.int32) % jnp.array([N0, N1])
        xm = x % 1.0
        xn = 1.0 - xm

        f1 = self.data[X1[:, 0], X1[:, 1]]
        f2 = self.data[X2[:, 0], X1[:, 1]]
        f3 = self.data[X1[:, 0], X2[:, 1]]
        f4 = self.data[X2[:, 0], X2[:, 1]]

        return (f1 * xn[:, 0] * xn[:, 1] +
                f2 * xm[:, 0] * xn[:, 1] +
                f3 * xn[:, 0] * xm[:, 1] +
                f4 * xm[:, 0] * xm[:, 1])

def gradient_2nd_order(F, i):
    """Second-order finite difference gradient"""
    return (1. / 12 * jnp.roll(F, 2, axis=i) - 2. / 3 * jnp.roll(F, 1, axis=i) +
            2. / 3 * jnp.roll(F, -1, axis=i) - 1. / 12 * jnp.roll(F, -2, axis=i))

def garfield(B, P, T_filt, seed=None):
    """Generate Gaussian random field"""
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(0)
    wn = jax.random.normal(key, shape=B.shape)
    f = jnp.fft.ifftn(jnp.fft.fftn(wn) * jnp.sqrt(P(B.K))).real
    return jnp.fft.ifftn(jnp.fft.fftn(f) * T_filt(B.K)).real
