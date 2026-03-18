import jax.numpy as jnp
from src.solver.state import State
from src.core.ops import gradient_2nd_order


def a2r(B, X):
    """Array (dim, N, ...) → particles (N^dim, dim)."""
    return X.reshape(B.dim, -1).T


def r2a(B, x):
    """Particles (N^dim, dim) → array (dim, N, ...)."""
    return x.T.reshape((B.dim,) + B.shape)


class Zeldovich:
    def __init__(self, B_mass, B_force, cosmology, phi):
        self.bm = B_mass
        self.bf = B_force
        self.cosmology = cosmology

        # Displacement field: (dim, N, ...) in grid units
        self.u = jnp.array(
            [-gradient_2nd_order(phi, i) for i in range(B_mass.dim)]
        ) / self.bm.res

    def state(self, a_init: float) -> State[jnp.ndarray]:
        """Generate initial state using Zeldovich approximation."""
        X = a2r(self.bm, jnp.indices(self.bm.shape) * self.bm.res + a_init * self.u)
        P = a2r(self.bm, a_init * self.u)
        return State(time=a_init, position=X, momentum=P)

    @property
    def particle_mass(self):
        return (self.bf.N / self.bm.N) ** self.bm.dim
