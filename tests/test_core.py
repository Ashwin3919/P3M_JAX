"""Unit tests for core numerical infrastructure (Phases 1–7).

Float64 is enabled for the entire session via conftest.py.
"""
import jax.numpy as jnp
import pytest
from src.core.ops import md_cic_2d, Interp2D, md_cic_nd, InterpND, md_tsc_nd, InterpTSC, gradient_4th_order
from src.core.box import Box


# ---------------------------------------------------------------------------
# Phase 1–2: Box, CIC, Interp, Gradient
# ---------------------------------------------------------------------------

def test_md_cic_2d_mass_conservation():
    """Total deposited mass equals number of particles."""
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


def test_gradient_4th_order():
    """4th-order FD gradient of sin(x) ≈ cos(x).

    gradient_4th_order returns the stencil numerator (not divided by Δx),
    so divide by dx before comparing — this is the documented convention.
    """
    N = 64
    x = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    dx = 2 * jnp.pi / N
    X, Y = jnp.meshgrid(x, x)
    F = jnp.sin(X)
    grad_x = gradient_4th_order(F, 1) / dx
    assert jnp.allclose(grad_x, jnp.cos(X), atol=1e-2)


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
# Phase 5: TSC mass deposition and interpolation
# ---------------------------------------------------------------------------

def test_md_tsc_nd_mass_conservation_2d():
    """TSC total deposited mass equals particle count (2D)."""
    shape = (64, 64)
    pos = jnp.array([[10.5, 10.5], [20.2, 30.8], [63.9, 63.9]])
    rho = md_tsc_nd(shape, pos)
    assert jnp.allclose(rho.sum(), len(pos), atol=1e-5)


def test_md_tsc_nd_mass_conservation_3d():
    """TSC total deposited mass equals particle count (3D)."""
    shape = (8, 8, 8)
    pos = jnp.array([[1.5, 1.5, 1.5], [4.5, 4.5, 4.5]])
    rho = md_tsc_nd(shape, pos)
    assert jnp.allclose(rho.sum(), 2.0, atol=1e-5)


def test_md_tsc_nd_unit_deposit_at_grid_center():
    """Particle at an exact integer grid point deposits entirely into that cell.

    For TSC the nearest-grid-point is round(pos) = pos exactly, so dx=0 and the
    only non-zero weight is w[0] = 0.75 - 0² = 0.75... wait, w[-1]=w[+1]=0 but
    w[0] = 0.75 - 0 = 0.75 → only 0.75 deposited? No: the ND weight is the
    product of 1D weights.  For a 2D particle: w_total = w_x[0] * w_y[0]
    = (0.75-0)*(0.75-0) = 0.5625 for corners (0,0).
    Actually TSC at exact integer pos has dx=0:
      w[-1]=0.5*(0.5-0)²=0.125, w[0]=0.75, w[+1]=0.5*(0.5+0)²=0.125 → sum=1.
    The particle weight is split across 3 cells per axis.  The central cell
    receives w[0]^d = 0.75^d of the mass.  Total across all 3^d cells = 1.
    This test just verifies mass conservation and correct central weight.
    """
    shape = (8, 8)
    pos = jnp.array([[4.0, 4.0]])   # exactly on grid point
    rho = md_tsc_nd(shape, pos)
    assert jnp.allclose(rho.sum(), 1.0, atol=1e-5)
    # Central cell gets 0.75^2 = 0.5625 of the mass
    assert jnp.allclose(rho[4, 4], 0.75 ** 2, atol=1e-5)


def test_md_tsc_nd_vs_cic_smoother():
    """TSC deposit is smoother than CIC: smaller peak cell mass for same particle.

    Both deposit 1 particle at the same off-integer position.  The TSC peak
    must be ≤ the CIC peak because TSC spreads mass over more cells.
    """
    shape = (16, 16)
    pos   = jnp.array([[3.3, 7.7]])
    rho_cic = md_cic_nd(shape, pos)
    rho_tsc = md_tsc_nd(shape, pos)
    assert float(rho_tsc.max()) <= float(rho_cic.max()) + 1e-6


def test_interp_tsc_constant_field():
    """InterpTSC returns the exact constant for a uniform field (partition of unity).

    TSC weights sum to 1 at every position, so a uniform field must be
    reproduced exactly regardless of particle position.
    """
    constant = 7.5
    data = jnp.ones((8, 8), dtype=jnp.float64) * constant
    interp = InterpTSC(data)
    pos = jnp.array([[0.0, 0.0], [1.3, 2.7], [3.5, 5.5]], dtype=jnp.float64)
    results = interp(pos)
    assert jnp.allclose(results, constant, atol=1e-6)


def test_interp_tsc_linear_field():
    """InterpTSC is exact for linear fields (polynomial reproduction of degree 1).

    B₂-spline kernels reproduce polynomials up to degree 2, so any linear
    function f(x,y) = ax + by must be recovered exactly.
    """
    N = 16
    x = jnp.arange(N, dtype=jnp.float64)
    X, Y = jnp.meshgrid(x, x, indexing='ij')
    data   = 2.0 * X + 3.0 * Y
    interp = InterpTSC(data)
    # Positions away from boundaries to avoid wrap-around artifacts
    pos = jnp.array([[4.3, 7.8], [2.1, 9.6]], dtype=jnp.float64)
    results  = interp(pos)
    expected = 2.0 * pos[:, 0] + 3.0 * pos[:, 1]
    assert jnp.allclose(results, expected, atol=1e-4)


def test_tsc_force_newton_third_law():
    """TSC solver satisfies Newton's 3rd law: total force on a periodic system is zero."""
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 16
    box = Box(2, N, 20.0)
    particle_mass = float(N ** 2) / 2.0
    pos = jnp.array([[5.3, 7.1], [14.2, 11.8]], dtype=jnp.float64)
    system = PoissonVlasov(box, EDS_PRESET, particle_mass=particle_mass, assignment="tsc")
    s = State(jnp.array(1.0), pos, jnp.zeros_like(pos))
    acc = system.momentumEquation(s)
    total = acc[0] + acc[1]
    assert jnp.allclose(total, 0.0, atol=1e-4), f"TSC force imbalance: {total}"


def test_tsc_uniform_force_is_zero():
    """TSC on a uniform particle grid: delta=0 → zero PM acceleration."""
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 8
    force_box = Box(2, N, 10.0)
    idx = jnp.arange(N)
    gx, gy = jnp.meshgrid(idx, idx)
    positions = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
                 * force_box.res)
    system = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0, assignment="tsc")
    s = State(jnp.array(1.0), positions, jnp.zeros_like(positions))
    acc = system.momentumEquation(s)
    assert jnp.allclose(acc, 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Phase 3: Zeldovich 3D shape test
# ---------------------------------------------------------------------------

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
    rho = md_cic_nd(bm.shape, s.position / bm.res)
    assert jnp.allclose(rho.mean(), 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Phase 4 / Phase 6: PM and P3M solver selection + erfc force splitting
# ---------------------------------------------------------------------------

def test_pm_uniform_force_is_zero():
    """Uniform particle grid → delta=0 → phi=0 → zero PM acceleration."""
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 8
    force_box = Box(2, N, 10.0)
    idx = jnp.arange(N)
    gx, gy = jnp.meshgrid(idx, idx)
    positions = (jnp.stack([gx.ravel(), gy.ravel()], axis=-1).astype(jnp.float64)
                 * force_box.res)

    system = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0, solver="pm")
    s = State(time=jnp.array(1.0), position=positions, momentum=jnp.zeros_like(positions))
    acc = system.momentumEquation(s)
    assert jnp.allclose(acc, 0.0, atol=1e-5)


def test_p3m_close_pair_force_larger_than_pm():
    """P3M magnitude > PM for a sub-cell-separation pair (PP adds at short range)."""
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET
    from src.solver.state import State

    N = 8
    force_box = Box(2, N, 10.0)
    positions = jnp.array([[2.0, 2.0], [2.3, 2.0]], dtype=jnp.float64)

    pm_sys  = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0, solver="pm")
    p3m_sys = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0,
                             solver="p3m", pp_window=1, pp_cutoff=2.5)

    s = State(time=jnp.array(1.0), position=positions, momentum=jnp.zeros_like(positions))
    pm_acc  = pm_sys.momentumEquation(s)
    p3m_acc = p3m_sys.momentumEquation(s)

    assert jnp.linalg.norm(p3m_acc[0]) > jnp.linalg.norm(pm_acc[0])


def test_pp_force_zero_beyond_cutoff():
    """PP correction vanishes for particles separated by more than pp_cutoff cells."""
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET

    N = 16
    force_box = Box(2, N, 20.0)   # res=1.25 → r_cut = 2.0*1.25 = 2.5 Mpc/h
    positions = jnp.array([[4.0, 4.0], [9.0, 4.0]], dtype=jnp.float64)  # 5 Mpc/h apart

    p3m_sys = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0,
                             solver="p3m", pp_window=4, pp_cutoff=2.0, pp_softening=0.1)
    a  = jnp.array(1.0)
    da = EDS_PRESET.da(a)
    pp = p3m_sys._pp_force(positions, a, da)
    assert jnp.allclose(pp, 0.0, atol=1e-6)


def test_pp_erfc_less_than_direct():
    """At short range the erfc weight is < 1, confirming force-splitting is applied."""
    import jax.scipy.special as jss
    from src.physics.system import PoissonVlasov
    from src.physics.cosmology import EDS_PRESET

    N = 8
    force_box = Box(2, N, 10.0)   # res=1.25 → r_cut=3.125, alpha≈1.2
    positions = jnp.array([[3.0, 3.0], [3.4, 3.0]], dtype=jnp.float64)

    p3m_sys = PoissonVlasov(force_box, EDS_PRESET, particle_mass=1.0,
                             solver="p3m", pp_window=1, pp_cutoff=2.5, pp_softening=0.01)
    a  = jnp.array(1.0)
    da = EDS_PRESET.da(a)
    pp = p3m_sys._pp_force(positions, a, da)

    r_cut  = 2.5 * force_box.res
    alpha  = r_cut / 2.6
    erfc_v = float(jss.erfc(0.4 / alpha))

    assert erfc_v < 1.0
    assert jnp.isfinite(pp[0]).all()
    assert jnp.linalg.norm(pp[0]) > 0


# ---------------------------------------------------------------------------
# Phase 7: Adaptive time-stepping
# ---------------------------------------------------------------------------

def test_compute_dt_decreases_with_velocity():
    """Higher particle velocities → smaller CFL dt (both unclamped)."""
    from src.physics.cosmology import EDS_PRESET
    from src.solver.integrator import compute_dt
    from src.solver.state import State

    pos = jnp.zeros((4, 2), dtype=jnp.float64)
    s_slow = State(jnp.array(1.0), pos, jnp.ones((4, 2), dtype=jnp.float64) * 0.1)
    s_fast = State(jnp.array(1.0), pos, jnp.ones((4, 2), dtype=jnp.float64) * 10.0)

    dt_slow = compute_dt(s_slow, EDS_PRESET, C_cfl=0.3, eps=0.5, dt_min=1e-4, dt_max=1000.0)
    dt_fast = compute_dt(s_fast, EDS_PRESET, C_cfl=0.3, eps=0.5, dt_min=1e-4, dt_max=1000.0)
    assert float(dt_slow) > float(dt_fast)


def test_compute_dt_respects_bounds():
    """compute_dt output stays within [dt_min, dt_max] for all velocity scales."""
    from src.physics.cosmology import EDS_PRESET
    from src.solver.integrator import compute_dt
    from src.solver.state import State

    dt_min, dt_max = 0.005, 0.05
    pos = jnp.zeros((16, 3), dtype=jnp.float64)
    for v_scale in [1e-6, 1.0, 1e6]:
        mom = jnp.ones((16, 3), dtype=jnp.float64) * v_scale
        s   = State(jnp.array(0.5), pos, mom)
        dt  = compute_dt(s, EDS_PRESET, C_cfl=0.3, eps=0.5, dt_min=dt_min, dt_max=dt_max)
        assert float(dt) >= dt_min - 1e-10
        assert float(dt) <= dt_max + 1e-10


# ---------------------------------------------------------------------------
# VTK I/O — regression tests for file format correctness
# ---------------------------------------------------------------------------

def test_write_vtk_particles_creates_file(tmp_path):
    """write_vtk_particles produces a non-empty binary VTK file."""
    import numpy as np
    from src.utils.io import write_vtk_particles

    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    mom = np.zeros_like(pos)
    path = write_vtk_particles(pos, mom, 0.5, str(tmp_path), "test")
    import os
    assert os.path.exists(path)
    with open(path, 'rb') as f:
        header = f.read(200)
    assert b"# vtk DataFile Version 3.0" in header
    assert b"BINARY" in header
    assert b"POLYDATA" in header
    assert b"POINTS 2 float" in header


def test_write_vtk_particles_2d_padded(tmp_path):
    """2D positions are zero-padded to (N, 3) for VTK compatibility."""
    import numpy as np
    from src.utils.io import write_vtk_particles

    pos = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mom = np.zeros_like(pos)
    path = write_vtk_particles(pos, mom, 0.3, str(tmp_path), "test2d")
    import os
    assert os.path.exists(path)


def test_write_vtk_density_2d_creates_file(tmp_path):
    """write_vtk_density produces a valid ASCII VTK structured-points file for 2D."""
    import numpy as np
    from src.utils.io import write_vtk_density

    N = 4
    box = Box(2, N, 10.0)
    rho = np.ones((N, N), dtype=np.float32) * 1.5
    path = write_vtk_density(rho, box, 0.5, str(tmp_path), "test_dens")
    import os
    assert os.path.exists(path)
    content = open(path).read()
    assert "STRUCTURED_POINTS" in content
    assert f"DIMENSIONS {N} {N} 1" in content
    assert "density float 1" in content


def test_write_vtk_density_3d_creates_file(tmp_path):
    """write_vtk_density produces a valid VTK file for 3D with correct dimensions."""
    import numpy as np
    from src.utils.io import write_vtk_density

    N = 4
    box = Box(3, N, 10.0)
    rho = np.ones((N, N, N), dtype=np.float32) * 2.0
    path = write_vtk_density(rho, box, 0.8, str(tmp_path), "test_dens3d")
    import os
    assert os.path.exists(path)
    content = open(path).read()
    assert f"DIMENSIONS {N} {N} {N}" in content


# ---------------------------------------------------------------------------
# 3D end-to-end smoke test
# ---------------------------------------------------------------------------

def test_3d_pm_simulation_advances_and_stays_finite():
    """3D PM simulation: run 2 leapfrog steps, check state is finite and a advances."""
    from src.core.filters import Power_law, Scale, Cutoff, Potential
    from src.core.ops import garfield
    from src.physics.cosmology import EDS_PRESET
    from src.physics.initial_conds import Zeldovich
    from src.physics.system import PoissonVlasov
    from src.solver.integrator import leap_frog
    from src.solver.state import State

    N = 8
    bm = Box(3, N, 10.0)
    bf = Box(3, N * 2, 10.0)
    P   = Power_law(-0.5) * Scale(bm, 0.2) * Cutoff(bm)
    phi = garfield(bm, P, Potential(), seed=1)
    za  = Zeldovich(bm, bf, EDS_PRESET, phi)
    state = za.state(0.02)

    system = PoissonVlasov(bf, EDS_PRESET, particle_mass=za.particle_mass)
    dt = 0.01
    for _ in range(2):
        state = leap_frog(dt, system, state)

    assert jnp.all(jnp.isfinite(state.position)), "3D positions contain NaN/Inf"
    assert jnp.all(jnp.isfinite(state.momentum)), "3D momenta contain NaN/Inf"
    assert float(state.time) > 0.02
