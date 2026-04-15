import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from src.solver.state import State, HamiltonianSystem
from src.core.ops import md_cic_nd, md_tsc_nd, InterpND, InterpTSC, gradient_4th_order
from src.core.filters import Potential, CICWindow, TSCWindow


class PoissonVlasov(HamiltonianSystem[jnp.ndarray]):
    def __init__(self, box, cosmology, particle_mass,
                 solver="pm", assignment="cic",
                 pp_window=2, pp_softening=0.1, pp_cutoff=2.5):
        self.box = box
        self.cosmology = cosmology
        self.particle_mass = particle_mass
        self.solver = solver
        self.assignment = assignment
        self.pp_window = pp_window
        self.pp_softening = pp_softening
        self.pp_cutoff = pp_cutoff

        # Build the deconvolved Poisson kernel -1/k² / W(k)² for both schemes.
        #
        # The mass-assignment window W(k) biases the density in Fourier space:
        #   ρ̂_mesh = W(k) · ρ̂_true   (deposition)
        #   F_particle = W(k) · F̂_mesh (interpolation)
        # Net bias: W(k)².  Dividing the kernel by W(k)² corrects both steps.
        #
        #   CIC: W(k) = ∏_d sinc²(k_d Δx / 2π)  [quadratic, sinc² per axis]
        #   TSC: W(k) = ∏_d sinc³(k_d Δx / 2π)  [cubic,     sinc³ per axis]
        #
        # Both windows are guarded against near-zero values at the Nyquist corner.
        if assignment == "tsc":
            self.kernel = (Potential() / TSCWindow(box) ** 2)(self.box.K)
        else:   # cic (default)
            self.kernel = (Potential() / CICWindow(box) ** 2)(self.box.K)

        if solver == "p3m":
            r_cut_phys = pp_cutoff * box.res
            if r_cut_phys >= box.L / 2:
                raise ValueError(
                    f"PP cutoff r_cut={r_cut_phys:.3f} Mpc/h must be < L/2={box.L/2:.3f} Mpc/h "
                    "for the minimum-image convention to be exact. "
                    "Reduce pp_cutoff or increase the box size."
                )

    def positionEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        return (s.momentum / (s.time ** 2 * da)).astype(s.position.dtype)

    def _pm_force(self, x_grid, phi, da):
        """PM long-range force: gradient of Poisson potential interpolated at particle positions.

        Uses InterpTSC when assignment='tsc' (B₂-spline weights), InterpND (CIC)
        otherwise.  The interpolation scheme must match the deposition scheme so
        that the deposit–solve–interpolate pipeline is self-adjoint.
        """
        interp_cls = InterpTSC if self.assignment == "tsc" else InterpND
        acc_components = [
            interp_cls(gradient_4th_order(phi, i))(x_grid)
            for i in range(self.box.dim)
        ]
        return jnp.stack(acc_components, axis=-1) / self.box.res

    def _morton_encode(self, x_grid):
        """Interleave bits of integer grid coordinates into a Morton (Z-curve) code.

        Loops run at trace time — XLA sees a fixed sequence of bitwise ops.
        Clips to [0, box.N - 1] before encoding; works for both 2D and 3D.
        """
        coords = jnp.floor(x_grid).astype(jnp.int64).clip(0, self.box.N - 1)
        code = jnp.zeros(coords.shape[0], dtype=jnp.int64)
        for bit in range(16):          # 16 bits per coordinate → up to 65536 grid points
            for d in range(self.box.dim):
                code = code | (((coords[:, d] >> bit) & 1) << (bit * self.box.dim + d))
        return code

    def _pp_force(self, pos, a, da):
        """Short-range PP correction with erfc force-splitting kernel.

        Uses the standard P3M split:
            F_PP(r) = G/a · m · r_hat/r² · erfc(r / alpha)
        where alpha = r_cut / 2.6 so erfc < 0.03% at the cutoff boundary.
        Particles are sorted by Morton code so the sliding window of width W
        captures spatially nearby neighbours. Minimum-image convention enforces
        periodic boundary conditions within the window.
        """
        W = self.pp_window
        eps = self.pp_softening
        L = self.box.L
        N_p = pos.shape[0]

        # Physical cutoff and splitting scale (alpha ≈ r_cut/2.6 → erfc(2.6)≈0.00024)
        r_cut = self.pp_cutoff * self.box.res
        alpha = r_cut / 2.6

        # Sort particles spatially via Morton code
        x_grid = pos / self.box.res
        codes = self._morton_encode(x_grid)
        order = jnp.argsort(codes)
        inv_order = jnp.argsort(order)
        sorted_pos = pos[order]

        offsets = jnp.arange(-W, W + 1)   # (2W+1,) — static at trace time

        def force_on_i(i):
            js_raw = i + offsets                           # (2W+1,) may be out-of-range
            js = jnp.clip(js_raw, 0, N_p - 1)             # safe gather indices
            neigh_pos = sorted_pos[js]                     # (2W+1, dim)

            r_vec = sorted_pos[i] - neigh_pos              # (2W+1, dim)
            r_vec = r_vec - L * jnp.round(r_vec / L)      # minimum image

            r_bare2 = jnp.sum(r_vec ** 2, axis=-1)        # (2W+1,) unsoftened

            # Safe distance to prevent `nan` gradients from sqrt(0) (e.g., self-interactions)
            safe_r_bare2 = jnp.where(r_bare2 > 0, r_bare2, 1.0)
            r_bare  = jnp.sqrt(safe_r_bare2)
            
            r_soft2 = r_bare2 + eps ** 2                   # softened squared
            r_soft3 = r_soft2 ** 1.5                       # |r_soft|^3

            # erfc splitting: PP correction goes to zero at r >= r_cut
            erfc_w = jss.erfc(r_bare / alpha)              # (2W+1,)

            # Exclude: self, out-of-range, and beyond physical cutoff
            valid = (js_raw >= 0) & (js_raw < N_p) & (js_raw != i) & (r_bare2 < r_cut**2)
            
            # Prevent 0 division in masked elements
            safe_r_soft3 = jnp.where(valid, r_soft3, 1.0)

            # F = G/a · m · erfc(r/alpha) / |r_soft|^3 · r_vec   (same units as PM acc)
            # particle_mass brings PP and PM forces onto the same scale:
            # PM includes mass via delta = rho * particle_mass - 1; PP must do so explicitly.
            G_eff = self.cosmology.G * self.particle_mass / a
            f = G_eff * jnp.sum(
                valid[:, None] * erfc_w[:, None] * r_vec / safe_r_soft3[:, None], axis=0
            )   # (dim,)
            return f

        sorted_acc = jax.vmap(force_on_i)(jnp.arange(N_p))
        return sorted_acc[inv_order]

    def momentumEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a  = s.time
        da = self.cosmology.da(a)
        x_grid = s.position / self.box.res

        # Python-level branch: resolved at trace time, zero runtime cost.
        if self.assignment == "tsc":
            delta = md_tsc_nd(self.box.shape, x_grid) * self.particle_mass - 1.0
        else:
            delta = md_cic_nd(self.box.shape, x_grid) * self.particle_mass - 1.0
        phi   = jnp.fft.ifftn(jnp.fft.fftn(delta) * self.kernel).real * self.cosmology.G / a

        pm_acc = self._pm_force(x_grid, phi, da)

        # Python-level branch: resolved at trace time, zero runtime cost
        if self.solver == "p3m":
            pp_acc = self._pp_force(s.position, a, da)
            return (-(pm_acc + pp_acc) / da).astype(s.position.dtype)
        return (-pm_acc / da).astype(s.position.dtype)
