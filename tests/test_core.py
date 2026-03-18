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
