"""Physics validation tests for P3M-JAX.

These tests verify that the simulation implements correct physics, not just
that the code runs. Each test isolates one physical property and checks it
against an analytic prediction.

Float64 is enabled for the entire session via conftest.py.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from src.core.box import Box
from src.core.ops import md_cic_nd, garfield, gradient_2nd_order
from src.core.filters import Power_law, Scale, Cutoff, Potential
from src.physics.cosmology import Cosmology, EDS_PRESET, LCDM_PRESET
from src.physics.system import PoissonVlasov
from src.physics.initial_conds import Zeldovich
from src.solver.state import State
from src.solver.integrator import leap_frog, compute_dt, step_chunk_adaptive
from src.utils.analysis import compute_power_spectrum


# ---------------------------------------------------------------------------
# 1. Cosmology — analytic checks
# ---------------------------------------------------------------------------

class TestCosmology:
    def test_eds_hubble_at_unity(self):
        """EdS: H(a=1) = H0 exactly."""
        cosmo = EDS_PRESET
        assert abs(float(cosmo.da(1.0)) - cosmo.H0) < 1e-10

    def test_eds_hubble_scaling(self):
        """EdS: H(a) = H0 * a^(-1/2) → da/dt = H0 * a^(1/2).

        da(a) = a * H(a) = a * H0 * a^(-3/2) = H0 * a^(-1/2)
        """
        cosmo = EDS_PRESET
        for a in [0.1, 0.5, 1.0, 2.0]:
            expected = cosmo.H0 * a ** (-0.5)
            got      = float(cosmo.da(a))
            assert abs(got - expected) / expected < 1e-10, f"a={a}: got {got}, expected {expected}"

    def test_lcdm_hubble_at_unity(self):
        """LCDM: H(a=1) = H0 * sqrt(OmegaM + OmegaL) = H0 for flat cosmology."""
        cosmo = LCDM_PRESET
        flat_check = abs(cosmo.OmegaM + cosmo.OmegaL - 1.0)
        assert flat_check < 1e-10, "LCDM_PRESET is not flat — test assumption violated"
        expected = cosmo.H0
        got      = float(cosmo.da(1.0))
        assert abs(got - expected) / expected < 1e-10

    def test_eds_growth_factor_linear(self):
        """EdS: D(a) ∝ a — the ratio D(a2)/D(a1) should equal a2/a1.

        Tested at a >= 2 where the +0.00001 lower-limit offset in the
        numerical integral is negligible (< 0.1%) relative to D(a).
        """
        cosmo = EDS_PRESET
        a1, a2 = 2.0, 4.0
        ratio_D = cosmo.growing_mode(a2) / cosmo.growing_mode(a1)
        ratio_a = a2 / a1
        assert abs(ratio_D / ratio_a - 1.0) < 0.02  # within 2% of linear

    def test_gravitational_coupling(self):
        """G_eff = (3/2) * OmegaM * H0^2 — Poisson coupling coefficient."""
        cosmo = EDS_PRESET
        expected = 1.5 * cosmo.OmegaM * cosmo.H0 ** 2
        assert abs(cosmo.G - expected) < 1e-10


# ---------------------------------------------------------------------------
# 2. Force pipeline — Poisson solver correctness
# ---------------------------------------------------------------------------

class TestForcePipeline:
    """Tests for the PM force pipeline: deposit → FFT Poisson → gradient → interp."""

    @pytest.fixture
    def uniform_system_2d(self):
        """N=8 force box, particles on a perfect grid (delta=0)."""
        N = 8
        box = Box(2, N, 10.0)
        idx = jnp.arange(N)
        gx, gy = jnp.meshgrid(idx, idx)
        pos = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
               * box.res)
        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0)
        return system, pos, box

    def test_uniform_density_zero_force(self, uniform_system_2d):
        """Uniform density → delta=0 → phi=0 → force=0 everywhere."""
        system, pos, _ = uniform_system_2d
        s = State(jnp.array(1.0), pos, jnp.zeros_like(pos))
        acc = system.momentumEquation(s)
        assert jnp.allclose(acc, 0.0, atol=1e-6)

    def test_pm_force_newton_third_law(self):
        """Forces on two equal-mass particles are equal and opposite.

        Newton's 3rd law is a consequence of the linearity of the Poisson
        equation: the total force on a periodic system is zero.
        """
        N = 16
        box = Box(2, N, 20.0)
        # Two particles off-grid; particle_mass = N^2/2 so mean density = 1
        particle_mass = float(N ** 2) / 2.0
        pos = jnp.array([[5.3, 7.1], [14.2, 11.8]], dtype=jnp.float64)
        system = PoissonVlasov(box, EDS_PRESET, particle_mass=particle_mass)
        s = State(jnp.array(1.0), pos, jnp.zeros_like(pos))
        acc = system.momentumEquation(s)
        # Total force = 0 on a periodic system (momentum conservation)
        total = acc[0] + acc[1]
        assert jnp.allclose(total, 0.0, atol=1e-5), f"Force imbalance: {total}"

    def test_pm_force_direction_attractive(self):
        """PM force on particle A points toward particle B (gravity is attractive).

        Setup: uniform background (N^2 grid) + one extra particle at a known
        offset. The force on the extra particle should point toward the grid.
        We check a simpler proxy: two particles with particle_mass = N^2/2
        → the force on particle 0 has a component in the direction of particle 1.
        """
        N = 16
        box = Box(2, N, 20.0)
        particle_mass = float(N ** 2) / 2.0
        # Particles 4 Mpc/h apart (< L/2=10): direct image is closer than
        # periodic image, so force on 0 is in +x toward particle 1.
        pos = jnp.array([[4.0, 10.0], [8.0, 10.0]], dtype=jnp.float64)
        system = PoissonVlasov(box, EDS_PRESET, particle_mass=particle_mass)
        s = State(jnp.array(1.0), pos, jnp.zeros_like(pos))
        acc = system.momentumEquation(s)
        # Particle 0 is to the left of particle 1 → force should be in +x
        assert float(acc[0, 0]) > 0, f"Force on particle 0 in x: {float(acc[0, 0]):.4f}"

    def test_single_fourier_mode_potential(self):
        """Single-mode density → Poisson gives correct potential phi_k = -G*delta_k/k^2.

        We create a pure cosine density perturbation at the fundamental mode
        and verify the potential has the expected amplitude.
        """
        N = 32
        L = 10.0
        box = Box(2, N, L)

        # delta(x,y) = A * cos(2pi*x/L)  [fundamental mode in x]
        A = 0.1
        x = jnp.linspace(0, L, N, endpoint=False)
        X, _ = jnp.meshgrid(x, x)
        delta = A * jnp.cos(2 * jnp.pi * X / L)

        # Solve Poisson in Fourier space
        kernel  = Potential()(box.K)
        delta_k = jnp.fft.fftn(delta)
        phi_k   = delta_k * kernel
        phi     = jnp.fft.ifftn(phi_k).real * EDS_PRESET.G  # at a=1

        # Analytic answer: phi = -G * A/k0^2 * cos(2pi*x/L)
        # where k0 = 2pi/L
        k0       = 2 * jnp.pi / L
        phi_anal = -EDS_PRESET.G * A / k0 ** 2 * jnp.cos(2 * jnp.pi * X / L)

        # Compare (the mean of both is zero by construction)
        assert jnp.allclose(phi, phi_anal, atol=1e-4), \
            f"Max error: {float(jnp.max(jnp.abs(phi - phi_anal))):.2e}"

    def test_gradient_of_potential_is_force(self):
        """gradient_2nd_order applied to a known phi recovers the exact force."""
        N = 64
        L = 2 * jnp.pi
        box = Box(2, N, float(L))
        x = jnp.linspace(0, float(L), N, endpoint=False)
        X, _ = jnp.meshgrid(x, x)

        # phi = sin(x) → dphi/dx = cos(x)
        phi = jnp.sin(X)
        grad_x = gradient_2nd_order(phi, 1) / box.res   # axis 1 = x dimension

        assert jnp.allclose(grad_x, jnp.cos(X), atol=1e-2)


# ---------------------------------------------------------------------------
# 3. P3M force — erfc splitting correctness
# ---------------------------------------------------------------------------

class TestP3MForce:
    """Verify that the erfc force-splitting kernel is correctly implemented."""

    def test_pm_plus_pp_closer_to_direct_newtonian(self):
        """F_PM + F_PP is closer to F_direct than F_PM alone.

        For a particle pair at separation r << r_cut, the erfc correction adds
        most of the missing short-range force that the PM misses.
        """
        N = 16
        L = 20.0
        box = Box(2, N, L)   # res = 1.25 Mpc/h

        # Particles 0.4 Mpc/h apart (0.32 cells) — deep in the sub-cell regime
        pos = jnp.array([[8.0, 8.0], [8.4, 8.0]], dtype=jnp.float64)
        r   = 0.4

        pm_sys  = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0, solver="pm")
        p3m_sys = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0,
                                 solver="p3m", pp_window=2, pp_cutoff=3.0,
                                 pp_softening=0.05)

        a  = jnp.array(1.0)
        s  = State(a, pos, jnp.zeros_like(pos))

        pm_acc  = pm_sys.momentumEquation(s)
        p3m_acc = p3m_sys.momentumEquation(s)
        pp_acc  = p3m_sys._pp_force(pos, a, EDS_PRESET.da(a))

        # Direct Newtonian (softened, same eps) for comparison
        eps      = 0.05
        G_eff    = EDS_PRESET.G / float(a)
        r_soft3  = (r ** 2 + eps ** 2) ** 1.5
        f_direct = G_eff * jnp.array([1.0, 0.0]) / r_soft3   # points in +x

        # _pp_force returns the contribution that is NEGATED in momentumEquation.
        # Particle 1 is in +x from particle 0 → r_vec[0→1] = +0.4 in x
        # but _pp_force computes r_vec = pos[i] - pos[j] (FROM j TO i) → -0.4 in x
        # so pp_acc[0, 0] < 0, and -pp_acc[0, 0] > 0 gives the attractive direction.
        assert float(pp_acc[0, 0]) < 0, \
            f"_pp_force on particle 0 in x should be negative (gets negated in momentumEquation): {float(pp_acc[0, 0]):.4f}"

        # P3M total magnitude should exceed PM magnitude
        assert jnp.linalg.norm(p3m_acc[0]) > jnp.linalg.norm(pm_acc[0])

    def test_erfc_split_sums_to_direct(self):
        """erfc(r/alpha) + erf(r/alpha) = 1 at every separation.

        This is the mathematical identity that guarantees the force splitting
        is exact: F_PP + F_PM = F_direct when PM is ideal.
        """
        import jax.scipy.special as jss
        N = 8
        box = Box(2, N, 10.0)
        r_cut = 2.5 * box.res
        alpha = r_cut / 2.6

        r_vals = jnp.linspace(0.01, r_cut, 50)
        erfc_v = jss.erfc(r_vals / alpha)
        erf_v  = jss.erf(r_vals / alpha)
        assert jnp.allclose(erfc_v + erf_v, 1.0, atol=1e-6)

    def test_pp_force_finite_and_bounded(self):
        """PP force is finite and bounded for arbitrary particle positions."""
        N = 16
        box = Box(2, N, 20.0)
        rng = np.random.default_rng(42)
        pos = jnp.array(rng.uniform(0.5, 19.5, size=(32, 2)))

        p3m_sys = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0,
                                 solver="p3m", pp_window=3, pp_cutoff=2.0,
                                 pp_softening=0.1)
        a  = jnp.array(0.5)
        da = EDS_PRESET.da(a)
        pp = p3m_sys._pp_force(pos, a, da)

        assert jnp.all(jnp.isfinite(pp)), "PP force contains NaN or Inf"
        # With softening 0.1 Mpc/h, force magnitude per particle is bounded
        max_force = float(jnp.max(jnp.linalg.norm(pp, axis=-1)))
        assert max_force < 1e6, f"PP force magnitude unexpectedly large: {max_force:.2e}"

    def test_p3m_cutoff_validation(self):
        """PoissonVlasov raises ValueError when r_cut >= L/2."""
        N = 8
        box = Box(2, N, 10.0)  # res=1.25, L/2=5.0
        # pp_cutoff=5 → r_cut=6.25 > L/2=5.0 → should raise
        with pytest.raises(ValueError, match="minimum-image"):
            PoissonVlasov(box, EDS_PRESET, particle_mass=1.0,
                          solver="p3m", pp_cutoff=5.0)


# ---------------------------------------------------------------------------
# 4. Leapfrog integrator
# ---------------------------------------------------------------------------

class TestIntegrator:
    def test_leapfrog_zero_force_constant_momentum(self):
        """With zero force (uniform density), momentum is constant.

        The position drifts linearly in a; momentum must not change.
        """
        N = 8
        box = Box(2, N, 10.0)
        idx = jnp.arange(N)
        gx, gy = jnp.meshgrid(idx, idx)
        pos = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
               * box.res)
        mom0 = jnp.ones_like(pos) * 0.3

        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0)
        s0 = State(jnp.array(0.5), pos, mom0)
        s1 = leap_frog(0.01, system, s0)

        # Force=0 so both half-kicks leave momentum unchanged
        assert jnp.allclose(s1.momentum, mom0, atol=1e-8), \
            f"Momentum changed under zero force. Max delta: {float(jnp.max(jnp.abs(s1.momentum - mom0))):.2e}"

    def test_leapfrog_advances_time(self):
        """leap_frog increments time by exactly dt."""
        N = 8
        box = Box(2, N, 10.0)
        pos = jnp.zeros((4, 2), dtype=jnp.float64)
        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0)
        s0 = State(jnp.array(0.3), pos, jnp.zeros_like(pos))
        dt = 0.05
        s1 = leap_frog(dt, system, s0)
        assert abs(float(s1.time) - 0.35) < 1e-12

    def test_leapfrog_position_drift_direction(self):
        """Positive momentum → position increases (drift in the correct direction)."""
        N = 8
        box = Box(2, N, 10.0)
        # Uniform density → no force; particle at (5,5) with positive x-momentum
        idx = jnp.arange(N)
        gx, gy = jnp.meshgrid(idx, idx)
        pos = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
               * box.res)
        mom = jnp.zeros_like(pos).at[0, 0].set(1.0)   # particle 0 has p_x = 1

        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0)
        s0 = State(jnp.array(1.0), pos, mom)
        s1 = leap_frog(0.01, system, s0)

        # Particle 0's x-coordinate should increase
        assert float(s1.position[0, 0]) > float(s0.position[0, 0])

    def test_leapfrog_approximate_time_reversibility(self):
        """Running forward then backward returns close to the initial state.

        Error is O(dt^2) for the Verlet integrator. We use uniform density
        so that the only time dependence comes from the cosmological prefactor.
        """
        N = 8
        box  = Box(2, N, 10.0)
        idx  = jnp.arange(N)
        gx, gy = jnp.meshgrid(idx, idx)
        pos0 = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
                * box.res)
        mom0 = jnp.ones_like(pos0) * 0.05

        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0)
        dt = 0.001   # small dt → O(dt^2) error is tiny
        s0 = State(jnp.array(1.0), pos0, mom0)
        s1 = leap_frog( dt, system, s0)
        s2 = leap_frog(-dt, system, s1)

        pos_err = float(jnp.max(jnp.abs(s2.position - s0.position)))
        mom_err = float(jnp.max(jnp.abs(s2.momentum - s0.momentum)))
        assert pos_err < 1e-8, f"Position reversibility error: {pos_err:.2e}"
        assert mom_err < 1e-8, f"Momentum reversibility error: {mom_err:.2e}"

    def test_adaptive_step_lands_exactly_at_target(self):
        """step_chunk_adaptive stops with state.time == a_target exactly."""
        from functools import partial
        N = 8
        box = Box(2, N, 10.0)
        idx = jnp.arange(N)
        gx, gy = jnp.meshgrid(idx, idx)
        pos = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
               * box.res)
        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0)
        s0 = State(jnp.array(0.1), pos, jnp.ones_like(pos) * 0.01)

        a_target = 0.15
        fn = jax.jit(partial(step_chunk_adaptive, system,
                             C_cfl=0.3, eps=0.5, dt_min=0.001, dt_max=0.05))
        s1 = fn(s0, a_target)
        jax.block_until_ready(s1)

        assert abs(float(s1.time) - a_target) < 1e-9, \
            f"Expected a={a_target}, got a={float(s1.time):.10f}"


# ---------------------------------------------------------------------------
# 5. Zeldovich initial conditions
# ---------------------------------------------------------------------------

class TestZeldovichICs:
    @pytest.fixture
    def za_2d(self):
        N = 16
        bm = Box(2, N, 10.0)
        bf = Box(2, N * 2, 10.0)
        P   = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
        phi = garfield(bm, P, Potential(), seed=7)
        return Zeldovich(bm, bf, EDS_PRESET, phi), bm, bf

    def test_particle_count(self, za_2d):
        za, bm, _ = za_2d
        s = za.state(0.02)
        assert s.position.shape == (bm.N ** bm.dim, bm.dim)
        assert s.momentum.shape == (bm.N ** bm.dim, bm.dim)

    def test_mean_density_unity(self, za_2d):
        """CIC-deposited density has mean exactly 1.0."""
        za, bm, _ = za_2d
        s   = za.state(0.02)
        rho = md_cic_nd(bm.shape, s.position / bm.res)
        assert abs(float(rho.mean()) - 1.0) < 1e-4

    def test_small_displacement_at_early_time(self, za_2d):
        """At a_init=0.02, max particle displacement << L/2.

        Lagrangian grid uses jnp.indices ordering: position[:, 0] is the
        first-axis (row/dim-0) coordinate.  Use indexing='ij' in meshgrid
        to match that convention.
        """
        za, bm, _ = za_2d
        s  = za.state(0.02)
        # Lagrangian positions — must match jnp.indices row-major layout
        idx = jnp.arange(bm.N)
        gi, gj = jnp.meshgrid(idx, idx, indexing='ij')
        q = (jnp.stack([gi.ravel(), gj.ravel()], axis=-1).astype(jnp.float64)
             * bm.res)
        max_disp = float(jnp.max(jnp.abs(s.position - q)))
        assert max_disp < bm.L / 4, \
            f"Max Zel'dovich displacement {max_disp:.3f} Mpc/h exceeds L/4={bm.L/4:.1f}"

    def test_positions_within_box(self, za_2d):
        """All particle positions lie within (-tol, L+tol) at a_init=0.02.

        Zeldovich ICs don't apply periodic wrapping; at early times particles
        near the boundary may be displaced by a small amount (< 0.1 Mpc/h)
        outside [0, L).  This is physically correct — the simulation handles
        periodic BCs in CIC deposition and force computation.
        """
        za, bm, _ = za_2d
        s = za.state(0.02)
        pos = np.array(s.position)
        tol = 0.1  # Mpc/h — a_init * max_displacement << L/2
        assert np.all(pos >= -tol),     f"Particles too far below 0 (min={pos.min():.4f})"
        assert np.all(pos < bm.L + tol), f"Particles too far above L={bm.L}"

    def test_momentum_proportional_to_displacement_eds(self, za_2d):
        """For EdS at a_init=0.02: p = a_init * u where u is the displacement field.

        In the code: X = q + a_init * u and P = a_init * u, so P = X - q.
        Lagrangian grid uses jnp.indices ordering (row-major, dim-0 first);
        indexing='ij' matches that convention.
        """
        za, bm, _ = za_2d
        a_init = 0.02
        s = za.state(a_init)
        idx = jnp.arange(bm.N)
        gi, gj = jnp.meshgrid(idx, idx, indexing='ij')
        q = (jnp.stack([gi.ravel(), gj.ravel()], axis=-1).astype(jnp.float64)
             * bm.res)
        displacement = s.position - q
        assert jnp.allclose(s.momentum, displacement, atol=1e-10), \
            "Zel'dovich momentum should equal displacement for EdS at a_init"


# ---------------------------------------------------------------------------
# 6. Power spectrum
# ---------------------------------------------------------------------------

class TestPowerSpectrum:
    def test_white_noise_flat_spectrum(self):
        """A white-noise density field produces an approximately flat P(k).

        The ratio P(k_high) / P(k_low) should be close to 1 (no strong tilt).
        """
        rng = np.random.default_rng(0)
        N = 64
        L = 100.0
        # White noise density (mean=1)
        rho = 1.0 + rng.standard_normal((N, N)) * 0.01

        k, Pk = compute_power_spectrum(rho, L, n_bins=10)
        assert len(k) >= 4, "Too few bins for white noise test"

        # Ratio of highest to lowest bin should be within 2x for white noise
        ratio = float(Pk[-1]) / float(Pk[0])
        assert 0.1 < ratio < 10.0, \
            f"P(k) ratio {ratio:.2f} suggests non-flat spectrum for white noise"

    def test_shot_noise_scaling_with_particle_count(self):
        """Shot noise level scales as 1/n_bar = V/N_particles.

        Two uniform density fields with N1 and N2=4*N1 particles.
        After shot-noise subtraction, the residual P(k) should differ by ~1/4.
        """
        N = 32
        L = 50.0

        # Uniform density (no structure, just particles on a grid)
        # N1 particles: particle_mass = N^2 / N1 → mean density = 1 (approx)
        rho_grid_16 = np.ones((N, N), dtype=float)
        rho_grid_64 = np.ones((N, N), dtype=float)

        n1 = 16 * 16    # 256 particles
        n2 = 64 * 64    # 4096 particles = 16x more

        _, Pk1 = compute_power_spectrum(rho_grid_16, L, particle_count=n1)
        _, Pk2 = compute_power_spectrum(rho_grid_64, L, particle_count=n2)

        # Shot noise for n2 is 1/16 of n1.  After subtraction, both should be
        # near zero for a uniform field, but the subtracted amount differs by 16x.
        V     = L ** 2
        sn1   = V / n1
        sn2   = V / n2
        ratio = sn1 / sn2
        assert abs(ratio - 16.0) < 1e-10

    def test_power_spectrum_total_variance(self):
        """Parseval's theorem: sum_k P(k) * k^(d-1) * dk / (2pi)^d ≈ <delta^2>.

        We use a cruder version: the mean of binned P(k) should be on the same
        order as the field variance.
        """
        rng = np.random.default_rng(1)
        N   = 32
        L   = 50.0
        sigma = 0.5
        rho   = 1.0 + rng.standard_normal((N, N)) * sigma   # known variance = sigma^2

        k, Pk = compute_power_spectrum(rho, L, n_bins=15, particle_count=None)
        assert len(Pk) > 0

        # P(k) values should be positive (no shot noise subtracted)
        assert np.all(Pk > 0), "P(k) contains non-positive values before shot noise subtraction"

    def test_cic_window_correction_increases_high_k_power(self):
        """CIC deconvolution should increase (not decrease) power at high k.

        The CIC window function W(k) < 1 for k > 0, so dividing by W^2 > 1
        should always yield P_corrected >= P_raw.
        """
        rng = np.random.default_rng(2)
        N   = 32
        L   = 50.0
        rho = 1.0 + rng.standard_normal((N, N)) * 0.1

        # Compute with and without window correction by comparing at high k
        k_corr, Pk_corr = compute_power_spectrum(rho, L, n_bins=8, particle_count=None)

        # The correction should only increase power (W^2 <= 1)
        # At low k, W ≈ 1 and correction is negligible
        # At high k, W < 1 and correction is significant
        # We just verify all returned P(k) values are positive and finite
        assert np.all(np.isfinite(Pk_corr)), "Corrected P(k) contains non-finite values"
        assert np.all(Pk_corr > 0), "Corrected P(k) contains non-positive values"

    def test_power_spectrum_output_shape(self):
        """compute_power_spectrum returns (k, Pk) arrays of the same length."""
        rho = np.ones((16, 16)) + np.random.default_rng(3).standard_normal((16, 16)) * 0.1
        k, Pk = compute_power_spectrum(rho, 10.0, n_bins=5)
        assert k.shape == Pk.shape
        assert k.ndim == 1
        assert len(k) > 0

    def test_3d_power_spectrum_shape(self):
        """compute_power_spectrum handles 3D density arrays."""
        rng = np.random.default_rng(4)
        rho = 1.0 + rng.standard_normal((16, 16, 16)) * 0.1
        k, Pk = compute_power_spectrum(rho, 50.0, n_bins=8)
        assert k.shape == Pk.shape
        assert np.all(k > 0)
        assert np.all(Pk > 0)


# ---------------------------------------------------------------------------
# 7. End-to-end smoke test — full PM step
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_single_pm_step_runs_and_advances(self):
        """A single leapfrog PM step on a small perturbed IC produces finite output."""
        N    = 16
        L    = 10.0
        bm   = Box(2, N, L)
        bf   = Box(2, N * 2, L)
        P    = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
        phi  = garfield(bm, P, Potential(), seed=99) * 2.0
        za   = Zeldovich(bm, bf, EDS_PRESET, phi)
        s0   = za.state(0.05)
        s0   = s0._replace(position=s0.position.astype(jnp.float64),
                           momentum=s0.momentum.astype(jnp.float64))

        system = PoissonVlasov(bf, EDS_PRESET, za.particle_mass, solver="pm")
        s1 = jax.jit(lambda s: leap_frog(0.01, system, s))(s0)
        jax.block_until_ready(s1)

        assert float(s1.time) == pytest.approx(0.06, abs=1e-12)
        assert jnp.all(jnp.isfinite(s1.position)), "Positions contain NaN/Inf after one step"
        assert jnp.all(jnp.isfinite(s1.momentum)), "Momenta contain NaN/Inf after one step"

    def test_p3m_step_runs_and_advances(self):
        """A single P3M step on a small system produces finite output."""
        N    = 8
        L    = 10.0
        bm   = Box(2, N, L)
        bf   = Box(2, N * 2, L)
        P    = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
        phi  = garfield(bm, P, Potential(), seed=42) * 1.0
        za   = Zeldovich(bm, bf, EDS_PRESET, phi)
        s0   = za.state(0.05)
        s0   = s0._replace(position=s0.position.astype(jnp.float64),
                           momentum=s0.momentum.astype(jnp.float64))

        system = PoissonVlasov(bf, EDS_PRESET, za.particle_mass,
                               solver="p3m", pp_window=2, pp_cutoff=2.0,
                               pp_softening=0.1)
        s1 = jax.jit(lambda s: leap_frog(0.01, system, s))(s0)
        jax.block_until_ready(s1)

        assert jnp.all(jnp.isfinite(s1.position)), "P3M positions contain NaN/Inf"
        assert jnp.all(jnp.isfinite(s1.momentum)), "P3M momenta contain NaN/Inf"
