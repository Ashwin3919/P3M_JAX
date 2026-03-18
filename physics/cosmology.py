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
        return 1 - self.OmegaM - self.OmegaL

    @property
    def G(self):
        return 3. / 2 * self.OmegaM * self.H0 ** 2

    def da(self, a):
        return self.H0 * a * jnp.sqrt(
            self.OmegaL +
            self.OmegaM * a ** -3 +
            self.OmegaK * a ** -2)

    def growing_mode(self, a):
        if isinstance(a, jnp.ndarray):
            return jnp.array([self.growing_mode(float(b)) for b in a])
        elif a <= 0.001:
            return a
        else:
            factor = 5. / 2 * self.OmegaM
            return factor * self.da(a) / a * \
                quad(lambda b: float(self.da(b)) ** (-3), 0.00001, a)[0] + 0.00001

# Standard cosmologies
LCDM = Cosmology(68.0, 0.31, 0.69)
EdS = Cosmology(70.0, 1.0, 0.0)
