from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Core power spectrum computation (Adapted for 2D)
# ---------------------------------------------------------------------------

def compute_power_spectrum(
    density: np.ndarray,
    box_size: float,
    n_bins: int = 30,
    particle_count: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute annularly- (2D) averaged power spectrum P(k).
    """
    density = np.asarray(density, dtype=np.float64)
    N = density.shape[0]
    ndim = density.ndim
    V = box_size**ndim

    # Density contrast
    delta = density - 1.0
    delta_k = np.fft.fftn(delta) / N**ndim
    Pk2d = np.abs(delta_k)**2 * V

    # wavenumber grid (works for any ndim)
    ki = np.fft.fftfreq(N, d=box_size/N) * 2*np.pi
    grids = np.meshgrid(*[ki]*ndim, indexing="ij")
    k_mag = np.sqrt(sum(g**2 for g in grids)).ravel()

    Pk2d = Pk2d.ravel()

    # ---- CIC deconvolution ----
    dx = box_size / N
    W = np.ones_like(grids[0])
    for g in grids:
        W *= np.sinc(g * dx / (2 * np.pi))
    W = W**2
    W = W.ravel()

    # Correct for CIC
    Pk2d /= np.where(W > 1e-6, W, 1.0)

    # ---- Shot noise correction ----
    if particle_count is not None:
        nbar = particle_count / V
        Pk2d -= 1.0 / nbar

    # ---- Bin averaging ----
    MIN_MODES = 5
    k_min = 2*np.pi / box_size
    k_max = np.sqrt(ndim) * np.pi * N / box_size
    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins+1)

    k_centers_list = []
    Pk_list = []

    for i in range(n_bins):
        mask = (k_mag >= edges[i]) & (k_mag < edges[i+1])
        if np.sum(mask) >= MIN_MODES:
            k_centers_list.append(np.mean(k_mag[mask]))
            Pk_list.append(np.mean(Pk2d[mask]))

    k_out = np.array(k_centers_list)
    Pk_out = np.array(Pk_list)
    valid = np.isfinite(k_out) & np.isfinite(Pk_out) & (Pk_out > 0)

    return k_out[valid], Pk_out[valid]

# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def append_to_csv(csv_path: Path | str, step: int, a: float, k: np.ndarray, Pk: np.ndarray):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["step","a","k","Pk"])
        for ki, Pi in zip(k, Pk):
            writer.writerow([step, f"{a:.6f}", f"{ki:.6e}", f"{Pi:.6e}"])

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_power_spectrum_evolution(csv_path: Path | str, output_path: Path | str):
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    
    if not csv_path.exists():
        return

    # Using Pandas for much more reliable data handling
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading power spectrum CSV: {e}")
        return
        
    if df.empty:
        return

    steps = df["step"].unique()
    a_values = df.groupby("step")["a"].first().values
    
    a_min, a_max = np.min(a_values), np.max(a_values)

    # Simplified style for better compatibility across systems
    _apply_robust_style()

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = matplotlib.colormaps["plasma"]
    
    # Avoid LogNorm error if a_min is 0 or very small
    vmin = max(a_min, 1e-4)
    vmax = max(a_max, vmin * 1.1)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for step, a in zip(steps, a_values):
        sub_df = df[df["step"] == step].sort_values("k")
        ax.plot(sub_df["k"], sub_df["Pk"], color=cmap(norm(a)), lw=1.2, alpha=0.8)

    # Add Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.03, aspect=20)
    cb.set_label("Scale factor a", fontsize=12)

    # Titles and Labels (explicitly set)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Wavenumber k [h/Mpc]", fontsize=12)
    ax.set_ylabel(r"Power P(k) [(Mpc/h)$^{d}$]", fontsize=12)
    ax.set_title("Density Power Spectrum Evolution", fontsize=14, pad=15)
    
    ax.grid(True, which="both", alpha=0.15)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _apply_robust_style():
    """Reset to a default, robust style that works on most machines"""
    plt.style.use("default")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",  # Professional LaTeX look without TeX install
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
