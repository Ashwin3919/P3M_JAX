import jax.numpy as jnp
from functools import reduce
from numbers import Number

def _K_pow(k, n):
    """Raise |k| to the n-th power safely"""
    return jnp.where(k == 0, 0.0, k ** n)

class Filter:
    """Base class for Fourier space filters"""
    def __init__(self, f):
        self.f = f

    def __call__(self, K):
        return self.f(K)

    def __mul__(self, other):
        if isinstance(other, Filter):
            return Filter(lambda K: self.f(K) * other.f(K))
        elif isinstance(other, Number):
            return Filter(lambda K: other * self.f(K))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Filter(lambda K: other * self.f(K))
        return NotImplemented

    def __pow__(self, n):
        return Filter(lambda K: self.f(K) ** n)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Filter(lambda K: self.f(K) / other)
        elif isinstance(other, Filter):
            return Filter(lambda K: self.f(K) / other.f(K))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Filter):
            return Filter(lambda K: self.f(K) + other.f(K))
        return NotImplemented

    def __invert__(self):
        return Filter(lambda K: self.f(K).conj())

    def abs(self, B, P):
        return jnp.sqrt(self.cc(B, P, self))

    def cc(self, B, P, other):
        """Inner product <self | other> weighted by P, with correct d-dimensional volume element."""
        return (~self * other * P)(B.K).sum().real / B.size * B.res ** B.dim

    def cf(self, B, other):
        """Cross-product of filter with a field array, with correct d-dimensional volume element."""
        return ((~self)(B.K) * other).sum().real / B.size * B.res ** B.dim

class Identity(Filter):
    def __init__(self):
        Filter.__init__(self, lambda K: 1)

class Zero(Filter):
    def __init__(self):
        Filter.__init__(self, lambda K: 0)

class Power_law(Filter):
    """Power law filter P(k) ∝ k^n"""
    def __init__(self, n):
        Filter.__init__(self, lambda K: _K_pow((K ** 2).sum(axis=0), n / 2))

def _scale_filter(B, t):
    """Discrete scale space filter"""
    def f(K):
        return reduce(
            lambda x, y: x * y,
            (jnp.exp(t / B.res ** 2 * (jnp.cos(k * B.res) - 1)) for k in K))
    return f

class Scale(Filter):
    def __init__(self, B, sigma):
        Filter.__init__(self, _scale_filter(B, sigma ** 2))

class Cutoff(Filter):
    def __init__(self, B):
        Filter.__init__(self, lambda K: jnp.where((K ** 2).sum(axis=0) <= B.k_max ** 2, 1.0, 0.0))

class Potential(Filter):
    """Gravitational potential filter: -1/k²"""
    def __init__(self):
        Filter.__init__(self, lambda K: -_K_pow((K ** 2).sum(axis=0), -1.0))
