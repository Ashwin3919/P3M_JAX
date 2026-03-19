from dataclasses import dataclass
import jax.numpy as jnp
from scipy.integrate import quad

@dataclass
class Cosmology:
    H0: float
    OmegaM: float
    OmegaL: float

    @property
    def OmegaK(self):
        return 1.0 - self.OmegaM - self.OmegaL

    @property
    def G(self):
        return 3. / 2 * self.OmegaM * self.H0 ** 2

    def da(self, a):
        return self.H0 * a * jnp.sqrt(
            self.OmegaL +
            self.OmegaM * a ** -3 +
            self.OmegaK * a ** -2)

    def growing_mode(self, a):
        """Linear growth factor D(a) via the Heath (1977) integral formula.

        D(a) = (5/2) * OmegaM * H(a)/a * ∫₀ᵃ [a' H(a')]⁻³ da'

        The integral is started from eps=1e-5 (not 0) to avoid the singularity
        at a→0.  The additive 1e-5 term approximates D(eps)≈eps for early-matter
        domination, but introduces a small (H0-dependent) offset at a < 0.1.
        Use this function only as a diagnostic; the simulation does not call it.
        """
        if isinstance(a, jnp.ndarray):
            return jnp.array([self.growing_mode(float(b)) for b in a])
        elif a <= 0.001:
            return a
        else:
            factor = 5. / 2 * self.OmegaM
            # Lower integration limit eps=1e-5 avoids 1/a³ singularity.
            # The + 0.00001 seeds D(eps) ≈ eps for matter domination.
            return factor * self.da(a) / a * \
                quad(lambda b: float(self.da(b)) ** (-3), 0.00001, a)[0] + 0.00001

# Presets for internal reference
LCDM_PRESET = Cosmology(68.0, 0.31, 0.69)
EDS_PRESET = Cosmology(70.0, 1.0, 0.0)
