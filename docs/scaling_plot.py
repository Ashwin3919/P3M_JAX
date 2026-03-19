import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── CONFIG ──────────────────────────────────────────────────────────────
DARK_MODE = False  # Set to True for slides, False for papers

if DARK_MODE:
    BG = "#0d1117"
    CARD_BG = "#0d1117"
    TEXT_CLR = "#c9d1d9"
    GRID_CLR = "#30363d"
else:
    BG = "#ffffff"
    CARD_BG = "#ffffff"
    TEXT_CLR = "#111111"
    GRID_CLR = "#dddddd"

# Okabe-Ito Palette
C_NUMPY = "#D55E00"   # Vermilion for NumPy reference
C_JAX   = "#0072B2"   # Blue for P3M-JAX
C_OOM   = "#CC9900"   # Amber for OOM warning

# ─── BENCHMARK DATA ──────────────────────────────────────────────────────
N_vals    = np.array([64, 128, 256, 512, 1024])
particles = N_vals ** 2

# Experiment 1: Δa = 0.005 (Low Res)
ref_time_1 = np.array([0.55,  1.30,  4.72,  21.36, 101.17])
jax_time_1 = np.array([0.50,  1.00,  2.02,   8.12,  43.13])
speedup_1  = [1.1, 1.3, 2.3, 2.6, 2.3]

# Experiment 2: Δa = 0.001 (High Res — N=1024 OOM)
ref_time_2 = np.array([1.84,  5.83, 23.13, 107.59,    np.nan])
jax_time_2 = np.array([1.12,  2.62,  8.60,  39.71,    np.nan])
speedup_2  = [1.6, 2.2, 2.7, 2.7, np.nan]

particle_labels = ["4K", "16K", "65K", "262K", "1M"]

# ─── PUBLICATION FONT SETTINGS ───────────────────────────────────────────
# Uses Matplotlib's mathtext (no external LaTeX needed).
# "stix" gives a Times New Roman-like serif face — standard for journals.
plt.rcParams.update({
    "text.usetex":       False,
    "mathtext.fontset":  "stix",       # journal-standard math font
    "font.family":       "STIXGeneral",
    "font.size":         11,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    TEXT_CLR,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.dpi":        300,
})

# ─── FIGURE ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 5.0))   # ~single-column journal width
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD_BG)

fig.subplots_adjust(bottom=0.20, top=0.91, left=0.12, right=0.97)

# ─── CURVES ──────────────────────────────────────────────────────────────
MS = 5
LW = 1.6

# Exp 1 — solid lines
ax.plot(particles, ref_time_1,
        color=C_NUMPY, marker='s', markersize=MS, linewidth=LW,
        linestyle='-',  label=r"NumPy, $\Delta a=0.005$",  zorder=3)
ax.plot(particles, jax_time_1,
        color=C_JAX,   marker='o', markersize=MS, linewidth=LW,
        linestyle='-',  label=r"P3M-JAX, $\Delta a=0.005$", zorder=3)

# Speedup labels below JAX points (Exp 1)
for px, py, sp in zip(particles, jax_time_1, speedup_1):
    ax.annotate(f"{sp}×", (px, py),
                textcoords="offset points", xytext=(0, -14),
                ha='center', va='top',
                color=C_JAX, fontweight='bold', fontsize=9, zorder=4)

# Exp 2 — dashed lines
ax.plot(particles, ref_time_2,
        color=C_NUMPY, marker='s', markersize=MS, linewidth=LW,
        linestyle='--', label=r"NumPy, $\Delta a=0.001$",  zorder=3)
ax.plot(particles, jax_time_2,
        color=C_JAX,   marker='o', markersize=MS, linewidth=LW,
        linestyle='--', label=r"P3M-JAX, $\Delta a=0.001$", zorder=3)

# Speedup labels above JAX points (Exp 2)
for px, py, sp in zip(particles, jax_time_2, speedup_2):
    if np.isnan(py):
        continue
    ax.annotate(f"{sp}×", (px, py),
                textcoords="offset points", xytext=(0, 8),
                ha='center', va='bottom',
                color=C_JAX, fontweight='bold', fontsize=9, zorder=4)

# ─── OOM ANNOTATION ──────────────────────────────────────────────────────
oom_x = particles[-1]
ax.axvline(oom_x, color=C_OOM, linestyle=':', linewidth=1.2, zorder=1)

ax.annotate(
    r"OOM ($\Delta a=0.001$)" + "\nJAX: 94%,  NumPy: 41%",
    xy=(oom_x, 90),
    xytext=(oom_x * 0.38, 18),
    color=C_OOM, ha='right', va='center', fontsize=8.5,
    arrowprops=dict(arrowstyle='->', color=C_OOM, shrinkA=0, shrinkB=4, lw=0.9)
)

# ─── AXES ────────────────────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_yscale('log')

ax.grid(True, which='major', color=GRID_CLR, linestyle='--', linewidth=0.5, zorder=0)

ax.set_xticks(particles)
ax.set_xticklabels(particle_labels, fontsize=10)
ax.tick_params(colors=TEXT_CLR, labelsize=10)

ax.set_xlabel(r"Number of Particles ($N^2$)",
              color=TEXT_CLR, fontsize=11, labelpad=6)
ax.set_ylabel("Wall-clock Time (s)",
              color=TEXT_CLR, fontsize=11, labelpad=6)
ax.set_title("Runtime Scaling: NumPy vs. P3M-JAX",
             color=TEXT_CLR, fontsize=12, pad=10)

ax.set_ylim(0.2, 500)

# ─── LEGEND ──────────────────────────────────────────────────────────────
legend = ax.legend(
    loc='upper left', frameon=True, facecolor=CARD_BG,
    edgecolor=GRID_CLR, fancybox=False, labelcolor=TEXT_CLR,
    fontsize=9.5, ncol=2, handlelength=2.2, columnspacing=1.0
)
legend.get_frame().set_linewidth(0.6)

# ─── FOOTNOTE ────────────────────────────────────────────────────────────
footnote = (
    r"Hardware: Apple M2 (CPU)  $|$  "
    r"2D PM Solver, $\Lambda$CDM ($L=50\,\mathrm{Mpc}/h$)"
)
fig.text(0.5, 0.01, footnote,
         ha='center', va='bottom', fontsize=8.5, color=TEXT_CLR)

# ─── SAVE ────────────────────────────────────────────────────────────────
suffix = "dark" if DARK_MODE else "light"
plt.savefig(f"benchmark_scaling_combined_{suffix}.pdf",
            facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.savefig(f"benchmark_scaling_combined_{suffix}.png",
            facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
plt.show()

print(f"\n✓ Saved → benchmark_scaling_combined_{suffix}.pdf / .png")