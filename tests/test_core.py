import jax.numpy as jnp
import pytest
from src.core.ops import md_cic_2d, Interp2D, gradient_2nd_order
from src.core.box import Box

def test_md_cic_2d_mass_conservation():
    """Test if mass is conserved during CIC deposition"""
    shape = (64, 64)
    # Random positions within the box
    pos = jnp.array([[10.5, 10.5], [20.2, 30.8], [63.9, 63.9]])
    
    rho = md_cic_2d(shape, pos)
    
    # Total mass should equal number of particles (each particle has weight 1.0)
    assert jnp.allclose(jnp.sum(rho), len(pos), atol=1e-5)

def test_interp2d_identity():
    """Test if Interp2D correctly interpolates at grid points"""
    data = jnp.arange(16).reshape(4, 4).astype(jnp.float32)
    interp = Interp2D(data)
    
    # At exact grid points (integers), it should return the exact value
    pos = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    results = interp(pos)
    
    expected = jnp.array([data[0,0], data[1,1], data[2,3]])
    assert jnp.allclose(results, expected, atol=1e-5)

def test_box_wave_numbers():
    """Test if Box correctly generates wave numbers"""
    N = 8
    L = 10.0
    box = Box(2, N, L)
    
    # Fundamental frequency should be 2*pi/L
    k_min_expected = 2 * jnp.pi / L
    assert jnp.isclose(box.k_min, k_min_expected)
    
    # Nyquist frequency should be N*pi/L
    k_max_expected = N * jnp.pi / L
    assert jnp.isclose(box.k_max, k_max_expected)

def test_gradient_2nd_order():
    """Test finite difference gradient on a simple sine wave"""
    N = 64
    x = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x)
    F = jnp.sin(X)
    
    # Gradient of sin(x) is cos(x)
    grad_x = gradient_2nd_order(F, 1) # Axis 1 corresponds to X in meshgrid
    expected = jnp.cos(X)
    
    # Note: 2nd order finite diff isn't perfect, but should be close
    assert jnp.allclose(grad_x, expected, atol=1e-2)
