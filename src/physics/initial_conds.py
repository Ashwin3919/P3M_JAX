import jax.numpy as jnp
from src.solver.state import State
from src.core.ops import gradient_2nd_order

def a2r(B, X):
    """Convert array to particles"""
    return X.transpose([1, 2, 0]).reshape([B.N ** 2, 2])

def r2a(B, x):
    """Convert particles to array"""
    return x.reshape([B.N, B.N, 2]).transpose([2, 0, 1])

class Zeldovich:
    def __init__(self, B_mass, B_force, cosmology, phi):
        self.bm = B_mass
        self.bf = B_force
        self.cosmology = cosmology

        # Compute displacement field
        self.u = jnp.array([-gradient_2nd_order(phi, 0),
                             -gradient_2nd_order(phi, 1)]) / self.bm.res

    def state(self, a_init: float) -> State[jnp.ndarray]:
        """Generate initial state using Zeldovich approximation"""
        X = a2r(self.bm, jnp.indices(self.bm.shape) * self.bm.res + a_init * self.u)
        P = a2r(self.bm, a_init * self.u)
        return State(time=a_init, position=X, momentum=P)

    @property
    def particle_mass(self):
        return (self.bf.N / self.bm.N) ** self.bm.dim
