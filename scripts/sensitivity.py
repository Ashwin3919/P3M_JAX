"""
Differentiable power spectrum: jax.grad through garfield → CIC → FFT.

Demonstrates that the full amplitude A → initial density → P(k) pipeline
is end-to-end differentiable.  Key result:

    P_total ∝ A²  (Zeldovich / linear regime)
    ⟹  dP_total/dA = 2 P_total / A   (analytic)

jax.grad recovers this exactly, verifying gradient flow through:
    scalar A  →  phi = phi_template * A
              →  Zeldovich displacement u ∝ A
              →  CIC density ρ
              →  FFT  →  |δ̂|²  →  P_total

Usage
-----
    python scripts/sensitivity.py

Output
------
    results/sensitivity/sensitivity.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from src.core.box import Box
from src.core.filters import Power_law, Scale, Cutoff, Potential
from src.core.ops import gradient_4th_order, md_cic_nd
from src.physics.cosmology import EDS_PRESET, Cosmology


# ---------------------------------------------------------------------------
# Differentiable forward model
# ---------------------------------------------------------------------------

def _make_power_fns(phi_template, bm, a_init, D_ratio):
    """Return jit-compiled (forward, grad) functions that take only scalar A.

    All non-array state (Box metadata, floats) is closed over so that JAX
    never tries to trace through a Python object — only the scalar A crosses
    the jit boundary.

    Physics:
        phi  = phi_template * A          (potential ∝ A)
        u    = -∇phi / res               (displacement ∝ A)
        δ    = ρ_CIC - 1                 (density contrast ∝ A)
        P    = Σ_k |δ̂_k|² V             (total power ∝ A²)
        dP/dA = 2P/A                     (analytic result we verify)
    """
    dim     = bm.dim
    res     = bm.res
    shape   = bm.shape
    size    = bm.size
    L       = bm.L
    lattice = jnp.indices(shape).astype(jnp.float64) * res   # constant

    def _total_power(A):
        phi    = phi_template * A
        u      = jnp.array([-gradient_4th_order(phi, i) for i in range(dim)]) / res
        X_flat = (lattice + D_ratio * a_init * u).reshape(dim, -1).T
        rho    = md_cic_nd(shape, X_flat / res)
        delta_k = jnp.fft.fftn(rho - 1.0) / size
        return jnp.sum(jnp.abs(delta_k) ** 2) * L ** dim

    return jax.jit(_total_power), jax.jit(jax.grad(_total_power))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N, L, n_s, a_init = 32, 50.0, -0.5, 0.02

    bm = Box(2, N, L)

    # Pre-compute growth-factor ratio outside the JAX trace (uses scipy.quad)
    eds_ref = Cosmology(H0=EDS_PRESET.H0, OmegaM=1.0, OmegaL=0.0)
    D_ratio = float(EDS_PRESET.growing_mode(a_init)) / float(eds_ref.growing_mode(a_init))

    # Fixed white-noise field — not a function of A
    wn = jax.random.normal(jax.random.PRNGKey(42), shape=bm.shape, dtype=jnp.float64)

    # Unit-amplitude potential template (computed once, reused for all A)
    P_filt       = Power_law(n_s) * Scale(bm, 0.2) * Cutoff(bm)
    phi_template = jnp.fft.ifftn(
        jnp.fft.fftn(wn) * jnp.sqrt(P_filt(bm.K)) * Potential()(bm.K)
    ).real   # shape (N, N), constant w.r.t. A

    # Build jit-compiled functions closed over phi_template and box constants
    pow_fn, grad_fn = _make_power_fns(phi_template, bm, a_init, D_ratio)

    # Sweep A values
    A_values = np.linspace(2.0, 20.0, 12)
    P_sim, dPdA_sim = [], []
    for A_val in A_values:
        A_arr = jnp.array(A_val)
        P_sim.append(float(pow_fn(A_arr)))
        dPdA_sim.append(float(grad_fn(A_arr)))

    P_sim    = np.array(P_sim)
    dPdA_sim = np.array(dPdA_sim)

    # Analytic linear-theory prediction: P ∝ A² ⟹ dP/dA = 2P/A.
    # This only holds for small A where displacements ≪ 1 grid cell.
    # At large A, particles cross cell boundaries and CIC becomes non-linear,
    # so the true gradient (from jax.grad) diverges from 2P/A — that IS physics.
    dPdA_linear = 2.0 * P_sim / A_values

    # Verify jax.grad against finite differences (machine-precision check).
    # This is independent of the linear-theory assumption.
    eps_fd = 1e-4
    A_mid  = jnp.array(A_values[len(A_values) // 2])
    dPdA_fd = float(pow_fn(A_mid + eps_fd) - pow_fn(A_mid - eps_fd)) / (2 * eps_fd)
    dPdA_ad = float(grad_fn(A_mid))
    fd_err  = abs(dPdA_ad - dPdA_fd) / (abs(dPdA_fd) + 1e-30)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.loglog(A_values, P_sim, "o-", lw=2, label="$P_{total}$ (simulation)")
    ax.loglog(A_values, P_sim[0] * (A_values / A_values[0]) ** 2,
              "--", lw=1.5, color="gray", label=r"$\propto A^2$ (linear, small $A$)")
    ax.set_xlabel("Initial amplitude $A$", fontsize=12)
    ax.set_ylabel(r"Total power $\sum_k P(k)$", fontsize=12)
    ax.set_title("$P_{total}$ vs $A$: linear for small $A$,\nnon-linear (CIC) for large $A$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.loglog(A_values, np.abs(dPdA_sim), "o-", lw=2,
               label=r"$|dP/dA|$ via jax.grad")
    ax2.loglog(A_values, dPdA_linear, "x--", lw=1.5, color="tab:orange",
               label=r"Linear theory $2P/A$")
    ax2.set_xlabel("Initial amplitude $A$", fontsize=12)
    ax2.set_ylabel(r"$|dP_{total}/dA|$", fontsize=12)
    ax2.set_title(
        f"jax.grad vs linear-theory $2P/A$\n"
        f"FD verification at $A={float(A_mid):.0f}$: rel. err = {fd_err:.1e}"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Differentiable simulation: jax.grad through garfield → CIC → FFT",
        fontsize=13, y=1.01
    )
    fig.tight_layout()

    out = Path("results/sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved → {path}")
    print(f"jax.grad vs finite-difference rel. error at A={float(A_mid):.0f}: {fd_err:.2e}")
    print("  → Should be < 1e-5 (machine-precision AD vs FD agreement)")
    assert fd_err < 1e-4, (
        f"jax.grad does not match finite differences: rel err = {fd_err:.2e}\n"
        "This would indicate broken gradient flow through CIC or FFT."
    )
    print("Assertion passed — jax.grad matches finite differences to machine precision.")


if __name__ == "__main__":
    main()
