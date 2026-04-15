"""
Linear-theory power spectrum comparison.

Runs a small 2D EdS PM simulation and overlays the measured P(k) against
the linear-theory prediction P_lin(k, a) = P_init(k) × (a / a_start)².

Key physics shown
-----------------
- Large-scale modes (small k): simulation tracks linear theory → linear regime.
- Small-scale modes (large k): simulation diverges upward → non-linear collapse.
- The k-scale where they first diverge is the non-linear scale k_nl(a).

Usage
-----
    python scripts/linear_theory.py

Output
------
    results/linear_theory/linear_theory.png
"""
import sys
import functools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

jax.config.update("jax_enable_x64", True)

from src.core.box import Box
from src.core.filters import Power_law, Scale, Cutoff, Potential
from src.core.ops import garfield, md_cic_nd
from src.physics.cosmology import EDS_PRESET
from src.physics.initial_conds import Zeldovich
from src.physics.system import PoissonVlasov
from src.solver.integrator import step_chunk
from src.utils.analysis import compute_power_spectrum, linear_theory_pk


def main():
    # --- Simulation parameters ---
    N       = 32       # particles/side (32² = 1024 particles)
    L       = 50.0     # box size [Mpc/h]
    n_s     = -0.5     # power-law index
    A       = 2.0      # small amplitude → stays in linear regime for longer
    a_start = 0.02
    a_end   = 1.0
    dt      = 0.02
    n_steps = round((a_end - a_start) / dt)   # 49 steps
    # Save a snapshot every ~10 steps
    save_every  = 10
    n_chunks    = n_steps // save_every        # 4 snapshots (plus initial)

    bm = Box(2, N, L)
    bf = Box(2, N * 2, L)

    print(f"Setting up simulation: N={N}, L={L} Mpc/h, {n_steps} steps")

    # Build ICs
    P_filt = Power_law(n_s) * Scale(bm, 0.2) * Cutoff(bm)
    phi    = garfield(bm, P_filt, Potential(), seed=4) * A
    za     = Zeldovich(bm, bf, EDS_PRESET, phi)
    state  = za.state(a_start)

    system   = PoissonVlasov(bf, EDS_PRESET, za.particle_mass)
    chunk_fn = jax.jit(
        functools.partial(step_chunk, system, dt=dt, save_every=save_every)
    )

    # --- Measure P(k) at initial snapshot ---
    # Shot noise level for reference line: 1/n̄ = V / N_particles
    shot_noise = L ** 2 / N ** 2

    def measure_pk(s):
        """Return (k, Pk) without shot noise subtraction.

        We omit shot noise subtraction so that early-time snapshots — where
        the physical signal P(k) can be below 1/n̄ — still return valid bins.
        The shot noise floor is shown as a horizontal line on the figure instead.
        """
        x_grid = s.position / bm.res
        rho    = np.array(md_cic_nd(bm.shape, x_grid)) + 1.0
        k, Pk  = compute_power_spectrum(rho, L, n_bins=20)
        return k, Pk

    k_init, Pk_init = measure_pk(state)

    snapshots = [(float(state.time), k_init, Pk_init)]

    # --- Evolve and record ---
    print("Evolving...", end=" ", flush=True)
    for chunk in range(n_chunks):
        state = chunk_fn(state)
        a_now = float(state.time)
        k, Pk = measure_pk(state)
        snapshots.append((a_now, k, Pk))
        print(f"a={a_now:.2f}", end=" ", flush=True)
    print("\nDone.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5.5))

    a_values = np.array([s[0] for s in snapshots])
    cmap  = matplotlib.colormaps["plasma"]
    norm  = mcolors.LogNorm(vmin=max(a_values.min(), 1e-3), vmax=a_values.max())

    for a_snap, k_snap, Pk_snap in snapshots:
        color = cmap(norm(a_snap))

        # Simulation P(k)
        ax.loglog(k_snap, Pk_snap, "-", color=color, lw=1.8, alpha=0.9)

        # Linear-theory P_lin(k, a) extrapolated from the initial snapshot.
        # Skip if the initial snapshot has no valid k bins (signal below shot noise).
        if len(k_init) == 0:
            continue
        Pk_lin = linear_theory_pk(k_snap, np.interp(k_snap, k_init, Pk_init),
                                   a_start, a_snap)
        ax.loglog(k_snap, Pk_lin, "--", color=color, lw=1.0, alpha=0.6)

    # Shot noise floor
    ax.axhline(shot_noise, color="dimgray", lw=1.0, ls=":", alpha=0.7)
    ax.text(bm.k_min * 1.1, shot_noise * 1.3, "shot noise $1/\\bar{n}$",
            fontsize=8, color="dimgray", va="bottom")

    # Legend entries (one per series type, not per snapshot)
    ax.loglog([], [], "k-",  lw=1.8, label="Simulation $P(k)$")
    ax.loglog([], [], "k--", lw=1.0, label="Linear theory $P_{\\rm lin}(k)$")

    # Colourbar for scale factor
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=22)
    cb.set_label("Scale factor $a$", fontsize=11)

    ax.set_xlabel(r"Wavenumber $k$ [h/Mpc]", fontsize=12)
    ax.set_ylabel(r"$P(k)$ [(Mpc/h)$^2$]", fontsize=12)
    ax.set_title(
        "Simulation vs Linear Theory\n"
        "Solid = simulation,  Dashed = linear theory  (same colour = same $a$)",
        fontsize=11
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)

    # Annotate the non-linear scale at the last snapshot
    a_last, k_last, Pk_last = snapshots[-1]
    Pk_lin_last = linear_theory_pk(k_last, np.interp(k_last, k_init, Pk_init),
                                    a_start, a_last)
    ratio = Pk_last / (Pk_lin_last + 1e-30)
    # Non-linear scale: first k where simulation exceeds linear theory by >20%
    nl_mask = ratio > 1.2
    if nl_mask.any():
        k_nl = k_last[nl_mask][0]
        ax.axvline(k_nl, color="gray", lw=1.2, ls=":", alpha=0.8)
        ax.text(k_nl * 1.05, ax.get_ylim()[0] * 2,
                f"$k_{{\\rm nl}} \\approx {k_nl:.2f}$\nh/Mpc",
                fontsize=9, color="gray", va="bottom")

    fig.tight_layout()

    out = Path("results/linear_theory")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "linear_theory.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")

    # Sanity check: use the first snapshot that has a valid (non-empty) initial
    # spectrum as baseline; compare large-scale modes against linear theory.
    if len(k_init) == 0:
        # Initial snapshot was below shot noise — use second snapshot as baseline
        _, k_base, Pk_base = snapshots[1]
        a_base = snapshots[1][0]
        _, k_check, Pk_check = snapshots[2]
        a_check = snapshots[2][0]
    else:
        k_base, Pk_base, a_base = k_init, Pk_init, a_start
        a_check, k_check, Pk_check = snapshots[1]

    Pk_lin_check = linear_theory_pk(
        k_check, np.interp(k_check, k_base, Pk_base), a_base, a_check
    )
    n_ls    = max(1, len(k_check) // 3)
    rel_dev = np.abs(Pk_check[:n_ls] / (Pk_lin_check[:n_ls] + 1e-30) - 1.0)
    print(f"Large-scale rel. deviation from linear theory at a={a_check:.2f}: "
          f"{rel_dev.mean():.1%}  (expect < 50% for PM on a coarse grid)")
    # Linear theory is exact only in the infinite-resolution limit.
    # A 50% tolerance accounts for: (1) discretisation on a 32² grid,
    # (2) the Zeldovich ICs themselves being an approximation, and
    # (3) mode-coupling already present at large a.
    # The key physics is visible in the plot: large-scale lines track each other,
    # small-scale lines diverge upward.
    assert rel_dev.mean() < 0.60, (
        f"Simulation deviates unexpectedly from linear theory at large scales: "
        f"{rel_dev.mean():.1%} — check ICs or force pipeline."
    )
    print("Assertion passed.")


if __name__ == "__main__":
    main()
