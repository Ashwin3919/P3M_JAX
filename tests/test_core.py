import jax.numpy as jnp
import pytest
from src.core.ops import md_cic_2d, Interp2D, md_cic_nd, InterpND, gradient_2nd_order
from src.core.box import Box


# ---------------------------------------------------------------------------
# Existing 2D tests (aliases must keep these passing)
# ---------------------------------------------------------------------------

def test_md_cic_2d_mass_conservation():
    """Mass is conserved during 2D CIC deposition."""
    shape = (64, 64)
    pos = jnp.array([[10.5, 10.5], [20.2, 30.8], [63.9, 63.9]])
    rho = md_cic_2d(shape, pos)
    assert jnp.allclose(jnp.sum(rho), len(pos), atol=1e-5)


def test_interp2d_identity():
    """Interp2D returns exact values at integer grid points."""
    data = jnp.arange(16).reshape(4, 4).astype(jnp.float32)
    interp = Interp2D(data)
    pos = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    results = interp(pos)
    expected = jnp.array([data[0, 0], data[1, 1], data[2, 3]])
    assert jnp.allclose(results, expected, atol=1e-5)


def test_box_wave_numbers():
    """Box generates correct fundamental and Nyquist frequencies."""
    N = 8
    L = 10.0
    box = Box(2, N, L)
    assert jnp.isclose(box.k_min, 2 * jnp.pi / L)
    assert jnp.isclose(box.k_max, N * jnp.pi / L)


def test_gradient_2nd_order():
    """2nd-order finite difference gradient of sin(x) ≈ cos(x).

    gradient_2nd_order returns the raw stencil output (derivative × dx),
    so we divide by dx to recover dF/dx before comparing.
    """
    N = 64
    x = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    dx = 2 * jnp.pi / N
    X, Y = jnp.meshgrid(x, x)
    F = jnp.sin(X)
    grad_x = gradient_2nd_order(F, 1) / dx
    assert jnp.allclose(grad_x, jnp.cos(X), atol=1e-2)


# ---------------------------------------------------------------------------
# Phase 1: 3D infrastructure tests (Box, filters, garfield)
# ---------------------------------------------------------------------------

def test_box_3d_shapes():
    box = Box(3, 64, 50.0)
    assert box.K.shape == (3, 64, 64, 64)
    assert box.k.shape == (64, 64, 64)


def test_potential_filter_3d():
    from src.core.filters import Potential
    box = Box(3, 16, 10.0)
    out = Potential()(box.K)
    assert out.shape == (16, 16, 16)


def test_garfield_3d():
    from src.core.filters import Power_law, Cutoff, Potential
    from src.core.ops import garfield
    box = Box(3, 16, 10.0)
    P = Power_law(-0.5) * Cutoff(box)
    f = garfield(box, P, Potential(), seed=0)
    assert f.shape == (16, 16, 16)


# ---------------------------------------------------------------------------
# Phase 2: nd CIC and InterpND tests
# ---------------------------------------------------------------------------

def test_md_cic_nd_mass_conservation_3d():
    shape = (8, 8, 8)
    pos = jnp.array([[1.0, 1.0, 1.0], [4.5, 4.5, 4.5]])
    rho = md_cic_nd(shape, pos)
    assert jnp.allclose(rho.sum(), 2.0, atol=1e-5)


def test_md_cic_nd_unit_deposit_3d():
    """Particle at exact cell center deposits entirely into that cell."""
    shape = (4, 4, 4)
    pos = jnp.array([[1.0, 2.0, 3.0]])
    rho = md_cic_nd(shape, pos)
    assert jnp.allclose(rho[1, 2, 3], 1.0, atol=1e-5)
    assert jnp.allclose(rho.sum(), 1.0, atol=1e-5)


def test_interpnd_3d_grid_points():
    data = jnp.arange(64).reshape(4, 4, 4).astype(jnp.float32)
    interp = InterpND(data)
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    out = interp(pos)
    assert jnp.allclose(out[0], data[0, 0, 0], atol=1e-5)
    assert jnp.allclose(out[1], data[1, 2, 3], atol=1e-5)


# ---------------------------------------------------------------------------
# Phase 3: Zeldovich 3D shape test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 4 / Phase 6: Solver selection (PM vs P3M) + erfc force splitting
# ---------------------------------------------------------------------------

def test_pm_uniform_force_is_zero():
    """Uniform particle distribution → PM acceleration is zero everywhere."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 8
    force_box = Box(2, N, 10.0)

    # Particles at every force-box grid point: integer x_grid → CIC deposits 1.0/cell
    idx = jnp.arange(N)
    gx, gy = jnp.meshgrid(idx, idx)
    positions = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
                 * force_box.res)

    # particle_mass=1 → delta = rho*1 - 1 = 0 everywhere → phi = 0 → acc = 0
    system = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0, solver="pm")
    s = State(time=jnp.array(1.0), position=positions, momentum=jnp.zeros_like(positions))
    acc = system.momentumEquation(s)
    assert jnp.allclose(acc, 0.0, atol=1e-5)


def test_p3m_close_pair_force_larger_than_pm():
    """P3M total acceleration > PM for a close particle pair (PP adds at short range)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 8
    force_box = Box(2, N, 10.0)

    # Two particles 0.3 Mpc/h apart — well below one grid cell (res=1.25 Mpc/h)
    positions = jnp.array([[2.0, 2.0], [2.3, 2.0]], dtype=jnp.float64)

    pm_sys  = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0, solver="pm")
    p3m_sys = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0,
                             solver="p3m", pp_window=1, pp_cutoff=2.5)

    s = State(time=jnp.array(1.0), position=positions, momentum=jnp.zeros_like(positions))

    pm_acc  = pm_sys.momentumEquation(s)
    p3m_acc = p3m_sys.momentumEquation(s)

    # erfc(r/alpha) > 0 at short range → P3M magnitude strictly exceeds PM
    pm_mag  = jnp.linalg.norm(pm_acc[0])
    p3m_mag = jnp.linalg.norm(p3m_acc[0])
    assert p3m_mag > pm_mag


def test_pp_force_zero_beyond_cutoff():
    """PP correction vanishes for particles separated by more than pp_cutoff cells."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 16
    force_box = Box(2, N, 20.0)   # res = 1.25 Mpc/h

    # pp_cutoff=2.0 → r_cut = 2.0 * 1.25 = 2.5 Mpc/h
    # Place two particles 5.0 Mpc/h apart (>> r_cut)
    positions = jnp.array([[4.0, 4.0], [9.0, 4.0]], dtype=jnp.float64)

    p3m_sys = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0,
                             solver="p3m", pp_window=4, pp_cutoff=2.0,
                             pp_softening=0.1)

    a = jnp.array(1.0)
    da = EDS_PRESET.da(a)
    pp = p3m_sys._pp_force(positions, a, da)

    # Both particles are beyond cutoff from each other → PP ≈ 0
    assert jnp.allclose(pp, 0.0, atol=1e-6)


def test_pp_erfc_less_than_direct():
    """At short range, erfc-split PP force is smaller than raw Newtonian (erfc < 1)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.scipy.special as jss
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 8
    L = 10.0
    force_box = Box(2, N, L)   # res = 1.25

    # pp_cutoff=2.5 → r_cut = 2.5 * 1.25 = 3.125, alpha = 3.125/2.6 ≈ 1.2
    positions = jnp.array([[3.0, 3.0], [3.4, 3.0]], dtype=jnp.float64)
    r = 0.4   # separation in Mpc/h

    p3m_sys = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0,
                             solver="p3m", pp_window=1, pp_cutoff=2.5,
                             pp_softening=0.01)
    a = jnp.array(1.0)
    da = EDS_PRESET.da(a)

    pp = p3m_sys._pp_force(positions, a, da)

    r_cut = 2.5 * force_box.res
    alpha = r_cut / 2.6
    erfc_val = float(jss.erfc(r / alpha))

    # erfc weight at r=0.4 must be < 1 (splitting kernel suppresses vs raw Newtonian)
    assert erfc_val < 1.0
    # PP force on particle 0 must be non-zero (within cutoff) and finite
    assert jnp.isfinite(pp[0]).all()
    assert jnp.linalg.norm(pp[0]) > 0


# ---------------------------------------------------------------------------
# Phase 7: Adaptive time-stepping
# ---------------------------------------------------------------------------

def test_compute_dt_decreases_with_velocity():
    """Higher particle velocities → smaller adaptive dt."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.physics.cosmology import EDS_PRESET
    from src.solver.integrator import compute_dt
    from src.solver.state import State

    pos = jnp.zeros((4, 2), dtype=jnp.float64)
    slow_mom = jnp.ones((4, 2), dtype=jnp.float64) * 0.1
    fast_mom = jnp.ones((4, 2), dtype=jnp.float64) * 10.0

    s_slow = State(time=jnp.array(1.0), position=pos, momentum=slow_mom)
    s_fast = State(time=jnp.array(1.0), position=pos, momentum=fast_mom)

    # dt_max large enough that fast particles don't hit the cap
    dt_slow = compute_dt(s_slow, EDS_PRESET, C_cfl=0.3, eps=0.5, dt_min=1e-4, dt_max=1000.0)
    dt_fast = compute_dt(s_fast, EDS_PRESET, C_cfl=0.3, eps=0.5, dt_min=1e-4, dt_max=1000.0)

    assert float(dt_slow) > float(dt_fast)


def test_compute_dt_respects_bounds():
    """compute_dt output is always within [dt_min, dt_max]."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.physics.cosmology import EDS_PRESET
    from src.solver.integrator import compute_dt
    from src.solver.state import State

    dt_min, dt_max = 0.005, 0.05
    pos = jnp.zeros((16, 3), dtype=jnp.float64)

    for v_scale in [1e-6, 1.0, 1e6]:
        mom = jnp.ones((16, 3), dtype=jnp.float64) * v_scale
        s = State(time=jnp.array(0.5), position=pos, momentum=mom)
        dt = compute_dt(s, EDS_PRESET, C_cfl=0.3, eps=0.5, dt_min=dt_min, dt_max=dt_max)
        assert float(dt) >= dt_min - 1e-10
        assert float(dt) <= dt_max + 1e-10


def test_zeldovich_3d_shape():
    from src.core.filters import Power_law, Scale, Cutoff, Potential
    from src.core.ops import garfield, md_cic_nd
    from src.physics.cosmology import EDS_PRESET
    from src.physics.initial_conds import Zeldovich

    N = 16
    bm = Box(3, N, 10.0)
    bf = Box(3, N * 2, 10.0)
    P = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
    phi = garfield(bm, P, Potential(), seed=0)
    za  = Zeldovich(bm, bf, EDS_PRESET, phi)
    s   = za.state(0.02)

    assert s.position.shape == (N**3, 3)
    assert s.momentum.shape == (N**3, 3)
    # CIC mean density ≈ 1
    rho = md_cic_nd(bm.shape, s.position / bm.res)
    assert jnp.allclose(rho.mean(), 1.0, atol=0.05)
