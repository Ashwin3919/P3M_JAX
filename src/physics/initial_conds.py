import jax.numpy as jnp
from src.solver.state import State
from src.core.ops import gradient_4th_order
from src.physics.cosmology import Cosmology as _Cosmology


def _grid_to_particles(B, X):
    """Flatten grid array (dim, N, ...) to particle array (N^dim, dim)."""
    return X.reshape(B.dim, -1).T


class Zeldovich:
    def __init__(self, B_mass, B_force, cosmology, phi):
        self.bm = B_mass
        self.bf = B_force
        self.cosmology = cosmology

        # Displacement field: (dim, N, ...) in grid units
        self.u = jnp.array(
            [-gradient_4th_order(phi, i) for i in range(B_mass.dim)]
        ) / self.bm.res

    def state(self, a_init: float) -> State[jnp.ndarray]:
        """Generate initial state using Zeldovich approximation.

        Positions:  X = q + D_ratio * a_init * u
        Momenta:    P = f(a_init) * D_ratio * a_init * u

        where:
          D_ratio = growing_mode_cosmo(a_init) / growing_mode_EdS(a_init)
                    (dimensionless ratio; = 1 for EdS, < 1 for LCDM at late times)
          f       = d ln D / d ln a  (Linder 2005 approximation)

        The ratio formulation cancels the H0-dependent normalisation in
        growing_mode, leaving a dimensionless correction factor.

        For EdS:   D_ratio = 1, f = 1  →  P = a_init * u  (identical to v1).
        For LCDM:  D_ratio < 1 at a ≳ 0.3, f < 1, correcting the initial
                   velocity field suppressed by dark energy.
        """
        # EdS reference with the SAME H0 so that H0-dependent normalisation
        # cancels exactly in the ratio D_cosmo / D_eds.
        _eds_ref = _Cosmology(H0=self.cosmology.H0, OmegaM=1.0, OmegaL=0.0)
        D_cosmo = float(self.cosmology.growing_mode(a_init))
        D_eds   = float(_eds_ref.growing_mode(a_init))
        D_ratio = D_cosmo / D_eds               # ≈ 1 at early times, < 1 late LCDM
        f       = float(self.cosmology.growth_rate(a_init))

        X = _grid_to_particles(self.bm,
                jnp.indices(self.bm.shape) * self.bm.res + D_ratio * a_init * self.u)
        P = _grid_to_particles(self.bm, f * D_ratio * a_init * self.u)
        return State(time=a_init, position=X, momentum=P)

    @property
    def particle_mass(self):
        return (self.bf.N / self.bm.N) ** self.bm.dim
