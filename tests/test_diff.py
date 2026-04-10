"""Differentiability tests for the PM pipeline.

Verifies that jax.grad flows through every stage of the PM force pipeline:
    init_pos → CIC deposit → FFT Poisson solve → FD gradient → interpolate
              → KDK leapfrog (lax.scan) → loss

Three levels of checks are performed:
  1. jax.grad does not raise (the pipeline is traceable end-to-end)
  2. Gradients are finite and non-zero (AD actually propagated signal)
  3. AD gradients match finite-difference estimates to within 1%
     (the VJP is numerically correct, not just structurally valid)

Float64 is required for accurate finite-difference baselines and is
enabled for the session via conftest.py.
"""
import jax
import jax.numpy as jnp
import pytest

from src.core.box import Box
from src.physics.cosmology import EDS_PRESET, LCDM_PRESET
from src.physics.system import PoissonVlasov
from src.solver.integrator import step_chunk
from src.solver.state import State
from src.diff import pm_rollout, make_loss_fn


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_pm_system():
    """16-cell 2D PM system — small enough for fast tests."""
    box = Box(N=16, L=100.0, dim=2)
    return PoissonVlasov(box, EDS_PRESET, particle_mass=1.0 / 16 ** 2, solver="pm")


@pytest.fixture(scope="module")
def small_pm_system_3d():
    """8-cell 3D PM system."""
    box = Box(N=8, L=50.0, dim=3)
    return PoissonVlasov(box, EDS_PRESET, particle_mass=1.0 / 8 ** 3, solver="pm")


@pytest.fixture(scope="module")
def init_state_2d():
    key = jax.random.PRNGKey(0)
    n_p = 32
    pos = jax.random.uniform(key, (n_p, 2), dtype=jnp.float64) * 90.0
    mom = jax.random.normal(key,  (n_p, 2), dtype=jnp.float64) * 0.01
    return pos, mom


@pytest.fixture(scope="module")
def init_state_3d():
    key = jax.random.PRNGKey(1)
    n_p = 16
    pos = jax.random.uniform(key, (n_p, 3), dtype=jnp.float64) * 40.0
    mom = jax.random.normal(key,  (n_p, 3), dtype=jnp.float64) * 0.01
    return pos, mom


# ---------------------------------------------------------------------------
# 1. Structural differentiability — jax.grad does not raise
# ---------------------------------------------------------------------------

class TestPMGradStructural:
    """Verify the AD graph can be built without errors."""

    def test_grad_wrt_positions_does_not_raise(self, small_pm_system, init_state_2d):
        """jax.grad w.r.t. init_pos traces the PM pipeline without error."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=2, dt=0.01)
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))
        dpos = grad_fn(init_pos, init_mom)
        assert dpos.shape == init_pos.shape

    def test_grad_wrt_momenta_does_not_raise(self, small_pm_system, init_state_2d):
        """jax.grad w.r.t. init_mom traces the PM pipeline without error."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=2, dt=0.01)
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=1))
        dmom = grad_fn(init_pos, init_mom)
        assert dmom.shape == init_mom.shape

    def test_value_and_grad_joint(self, small_pm_system, init_state_2d):
        """jax.value_and_grad returns a scalar loss and matching gradient shapes."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=2, dt=0.01)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
        loss, (dpos, dmom) = grad_fn(init_pos, init_mom)
        assert loss.shape == ()
        assert dpos.shape == init_pos.shape
        assert dmom.shape == init_mom.shape

    def test_grad_3d_does_not_raise(self, small_pm_system_3d, init_state_3d):
        """Gradient traces through 3D PM pipeline."""
        init_pos, init_mom = init_state_3d
        loss_fn = make_loss_fn(small_pm_system_3d, a_init=0.1, n_steps=2, dt=0.01)
        dpos = jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)
        assert dpos.shape == init_pos.shape

    def test_grad_lcdm_does_not_raise(self, init_state_2d):
        """Gradient traces through LCDM cosmology (non-EdS Hubble parameter)."""
        box = Box(N=16, L=100.0, dim=2)
        system = PoissonVlasov(box, LCDM_PRESET, particle_mass=1.0 / 16 ** 2, solver="pm")
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(system, a_init=0.1, n_steps=2, dt=0.01)
        dpos = jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)
        assert dpos.shape == init_pos.shape


# ---------------------------------------------------------------------------
# 2. Gradient signal — AD propagates non-zero, finite values
# ---------------------------------------------------------------------------

class TestPMGradSignal:
    """Verify that gradients carry actual signal (not all-zero or NaN/Inf)."""

    def test_grad_pos_finite(self, small_pm_system, init_state_2d):
        """Gradient w.r.t. positions is finite everywhere."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)
        dpos = jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)
        assert jnp.all(jnp.isfinite(dpos)), f"Non-finite gradient: {dpos}"

    def test_grad_pos_nonzero(self, small_pm_system, init_state_2d):
        """Gradient w.r.t. positions is not identically zero."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)
        dpos = jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)
        assert jnp.linalg.norm(dpos) > 1e-12, "Gradient is all-zero — AD did not propagate"

    def test_grad_mom_finite(self, small_pm_system, init_state_2d):
        """Gradient w.r.t. momenta is finite everywhere."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)
        dmom = jax.jit(jax.grad(loss_fn, argnums=1))(init_pos, init_mom)
        assert jnp.all(jnp.isfinite(dmom)), f"Non-finite gradient: {dmom}"

    def test_grad_mom_nonzero(self, small_pm_system, init_state_2d):
        """Gradient w.r.t. momenta is not identically zero."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)
        dmom = jax.jit(jax.grad(loss_fn, argnums=1))(init_pos, init_mom)
        assert jnp.linalg.norm(dmom) > 1e-12, "Gradient is all-zero — AD did not propagate"

    def test_different_positions_give_different_grads(self, small_pm_system):
        """Two distinct initial conditions produce distinct gradients."""
        key1 = jax.random.PRNGKey(10)
        key2 = jax.random.PRNGKey(11)
        n_p = 32
        pos1 = jax.random.uniform(key1, (n_p, 2), dtype=jnp.float64) * 90.0
        pos2 = jax.random.uniform(key2, (n_p, 2), dtype=jnp.float64) * 90.0
        mom  = jnp.zeros((n_p, 2), dtype=jnp.float64)

        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=2, dt=0.01)
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

        dpos1 = grad_fn(pos1, mom)
        dpos2 = grad_fn(pos2, mom)
        assert not jnp.allclose(dpos1, dpos2), "Different ICs gave identical gradients"


# ---------------------------------------------------------------------------
# 3. Numerical correctness — AD vs finite differences
# ---------------------------------------------------------------------------

class TestPMGradNumerical:
    """Check that AD gradients match finite-difference estimates.

    Tolerance: relative error < 1% for a single component.
    We test one component each for pos and mom to keep runtime short.
    """

    @pytest.fixture(scope="class")
    def precomputed(self, small_pm_system, init_state_2d):
        """Pre-compute AD gradients once for all numerical tests in this class."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))
        loss, (dpos, dmom) = grad_fn(init_pos, init_mom)
        return loss_fn, init_pos, init_mom, dpos, dmom

    def _fd_grad(self, f, x, y, particle, dim, eps=1e-3):
        """Central-difference gradient of f(x, y) w.r.t. x[particle, dim]."""
        xp = x.at[particle, dim].add( eps)
        xm = x.at[particle, dim].add(-eps)
        return (f(xp, y) - f(xm, y)) / (2.0 * eps)

    def test_fd_check_pos_component(self, small_pm_system, init_state_2d):
        """AD gradient of loss w.r.t. pos[0,0] matches FD to within 1%."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)

        ad_grad  = float(jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)[0, 0])
        fd_grad  = float(self._fd_grad(loss_fn, init_pos, init_mom, particle=0, dim=0))

        rel_err = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-12)
        assert rel_err < 0.01, (
            f"AD={ad_grad:.6f}, FD={fd_grad:.6f}, rel_err={rel_err:.2e} — "
            "AD gradient does not match finite differences"
        )

    def test_fd_check_pos_second_component(self, small_pm_system, init_state_2d):
        """AD gradient of loss w.r.t. pos[3,1] matches FD to within 1%."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)

        ad_grad = float(jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)[3, 1])
        fd_grad = float(self._fd_grad(loss_fn, init_pos, init_mom, particle=3, dim=1))

        rel_err = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-12)
        assert rel_err < 0.01, (
            f"AD={ad_grad:.6f}, FD={fd_grad:.6f}, rel_err={rel_err:.2e}"
        )

    def test_fd_check_mom_component(self, small_pm_system, init_state_2d):
        """AD gradient of loss w.r.t. mom[0,0] matches FD to within 1%."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=3, dt=0.01)

        ad_grad = float(jax.jit(jax.grad(loss_fn, argnums=1))(init_pos, init_mom)[0, 0])
        fd_grad = float(self._fd_grad(
            lambda m, p: loss_fn(p, m), init_mom, init_pos, particle=0, dim=0
        ))

        rel_err = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-12)
        assert rel_err < 0.01, (
            f"AD={ad_grad:.6f}, FD={fd_grad:.6f}, rel_err={rel_err:.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Multi-step gradient consistency
# ---------------------------------------------------------------------------

class TestPMGradMultiStep:
    """Verify gradients remain valid as n_steps increases."""

    @pytest.mark.parametrize("n_steps", [1, 5, 10])
    def test_grad_finite_for_n_steps(self, small_pm_system, init_state_2d, n_steps):
        """Gradient is finite for 1, 5, and 10 leapfrog steps."""
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(small_pm_system, a_init=0.1, n_steps=n_steps, dt=0.01)
        dpos = jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)
        assert jnp.all(jnp.isfinite(dpos)), f"Non-finite grad at n_steps={n_steps}"

    def test_checkpoint_gives_same_grad_as_no_checkpoint(self, small_pm_system, init_state_2d):
        """jax.checkpoint does not change the gradient values, only memory usage."""
        init_pos, init_mom = init_state_2d

        def loss(pos, mom, ckpt):
            final = pm_rollout(pos, mom, 0.1, 5, 0.01, small_pm_system,
                               checkpoint=ckpt)
            return jnp.mean(final.position ** 2)

        grad_no_ckpt = jax.jit(jax.grad(lambda p, m: loss(p, m, False), argnums=0))(init_pos, init_mom)
        grad_ckpt    = jax.jit(jax.grad(lambda p, m: loss(p, m, True),  argnums=0))(init_pos, init_mom)

        assert jnp.allclose(grad_no_ckpt, grad_ckpt, rtol=1e-5, atol=1e-8), (
            "Checkpoint changed gradient values — rematerialisation is incorrect"
        )


# ---------------------------------------------------------------------------
# 5. step_chunk checkpoint integration
# ---------------------------------------------------------------------------

class TestStepChunkCheckpoint:
    """step_chunk(checkpoint=True) must give the same result as checkpoint=False."""

    def test_checkpoint_matches_no_checkpoint_forward(self, small_pm_system, init_state_2d):
        """Forward pass: step_chunk with/without checkpoint gives identical states."""
        init_pos, init_mom = init_state_2d
        s0 = State(jnp.asarray(0.1), init_pos, init_mom)

        s_no_ckpt = step_chunk(small_pm_system, s0, dt=0.01, save_every=5, checkpoint=False)
        s_ckpt    = step_chunk(small_pm_system, s0, dt=0.01, save_every=5, checkpoint=True)

        assert jnp.allclose(s_no_ckpt.position, s_ckpt.position, atol=1e-10)
        assert jnp.allclose(s_no_ckpt.momentum, s_ckpt.momentum, atol=1e-10)

    def test_checkpoint_grad_matches_no_checkpoint(self, small_pm_system, init_state_2d):
        """Backward pass: gradients with/without checkpoint are numerically identical."""
        init_pos, init_mom = init_state_2d

        def loss(pos, mom, ckpt):
            s0    = State(jnp.asarray(0.1, dtype=jnp.float64), pos, mom)
            final = step_chunk(small_pm_system, s0, dt=0.01, save_every=3, checkpoint=ckpt)
            return jnp.mean(final.position ** 2)

        grad_no_ckpt = jax.jit(jax.grad(lambda p, m: loss(p, m, False), argnums=0))(init_pos, init_mom)
        grad_ckpt    = jax.jit(jax.grad(lambda p, m: loss(p, m, True),  argnums=0))(init_pos, init_mom)

        assert jnp.allclose(grad_no_ckpt, grad_ckpt, rtol=1e-5, atol=1e-8), \
            "step_chunk checkpoint changed gradient values"


# ---------------------------------------------------------------------------
# 6. pm_rollout delegates to step_chunk (no duplicate scan logic)
# ---------------------------------------------------------------------------

class TestPMRolloutDelegation:
    """pm_rollout (non-trajectory) must give the same result as step_chunk directly."""

    def test_pm_rollout_matches_step_chunk(self, small_pm_system, init_state_2d):
        """pm_rollout without return_trajectory produces the same state as step_chunk."""
        init_pos, init_mom = init_state_2d
        a_init  = 0.1
        n_steps = 5
        dt      = 0.01

        # pm_rollout path
        final_rollout = pm_rollout(init_pos, init_mom, a_init, n_steps, dt, small_pm_system)

        # step_chunk path
        s0    = State(jnp.asarray(a_init, dtype=jnp.float64), init_pos, init_mom)
        final_chunk = step_chunk(small_pm_system, s0, dt=dt, save_every=n_steps)

        assert jnp.allclose(final_rollout.position, final_chunk.position, atol=1e-10)
        assert jnp.allclose(final_rollout.momentum, final_chunk.momentum, atol=1e-10)
        assert jnp.allclose(final_rollout.time,     final_chunk.time,     atol=1e-12)


# ---------------------------------------------------------------------------
# 7. P3M Path Differentiability
# ---------------------------------------------------------------------------

class TestP3MDifferentiability:
    """pm_rollout gracefully traces the P3M short-range solver."""

    def test_p3m_grad_does_not_raise_and_is_finite(self, init_state_2d):
        """Gradient traces through P3M pipeline without NaN/errors."""
        box = Box(N=16, L=100.0, dim=2)
        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0 / 16 ** 2,
                               solver="p3m", pp_cutoff=2.5)
        init_pos, init_mom = init_state_2d
        
        loss_fn = make_loss_fn(system, a_init=0.1, n_steps=1, dt=0.01)
        dpos = jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)
        
        assert dpos.shape == init_pos.shape
        assert jnp.all(jnp.isfinite(dpos)), f"Non-finite grad in P3M rollout: {dpos}"
        
class TestP3MGradNumerical:
    """Check that AD gradients for P3M match finite-difference estimates."""
    
    def _fd_grad(self, f, x, y, particle, dim, eps=1e-3):
        xp = x.at[particle, dim].add( eps)
        xm = x.at[particle, dim].add(-eps)
        return (f(xp, y) - f(xm, y)) / (2.0 * eps)

    def test_p3m_fd_check_pos_component(self, init_state_2d):
        """AD gradient of P3M loss w.r.t. pos[0,0] matches FD to within 1%."""
        box = Box(N=16, L=100.0, dim=2)
        system = PoissonVlasov(box, EDS_PRESET, particle_mass=1.0 / 16 ** 2,
                               solver="p3m", pp_cutoff=2.5)
        init_pos, init_mom = init_state_2d
        loss_fn = make_loss_fn(system, a_init=0.1, n_steps=1, dt=0.01)

        ad_grad  = float(jax.jit(jax.grad(loss_fn, argnums=0))(init_pos, init_mom)[0, 0])
        fd_grad  = float(self._fd_grad(loss_fn, init_pos, init_mom, particle=0, dim=0))

        rel_err = abs(ad_grad - fd_grad) / (abs(fd_grad) + 1e-12)
        assert rel_err < 0.01, (
            f"P3M AD={ad_grad:.6f}, FD={fd_grad:.6f}, rel_err={rel_err:.2e} — "
            "AD gradient does not match finite differences"
        )

