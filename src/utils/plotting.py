import matplotlib.pyplot as plt
import numpy as np
from src.core.ops import md_cic_nd


def plot_density_evolution(all_states, box, times=None, save_path=None):
    """Plot density field evolution at three scale factors.

    For 3D boxes a z-projection (mean along last axis) is shown.

    Parameters
    ----------
    all_states : State  — stacked trajectory
    box        : Box
    times      : list of float | None — target scale factors; defaults to
                 [a_start, mid, a_end] inferred from trajectory
    save_path  : str | None
    """
    if times is None:
        t = np.asarray(all_states.time)
        times = [float(t[0]), float(t[len(t) // 2]), float(t[-1])]

    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(2, len(times), figsize=(5 * len(times), 10))
        if len(times) == 1:
            axes = axes.reshape(-1, 1)

        for i, t_target in enumerate(times):
            idx = np.argmin(np.abs(np.asarray(all_states.time) - t_target))
            pos = all_states.position[idx]
            t_actual = float(all_states.time[idx])

            x_grid = pos / box.res
            rho = np.array(md_cic_nd(box.shape, x_grid)) + 1.0

            rho_2d = rho.mean(axis=2) if rho.ndim == 3 else rho

            # Determine data-driven colour range
            log_rho = np.log10(np.maximum(rho_2d, 0.1))
            vmin_log = np.percentile(log_rho, 2)
            vmax_log = np.percentile(log_rho, 98)

            ax_top = axes[0, i]
            im1 = ax_top.imshow(log_rho.T, extent=[0, box.L, 0, box.L],
                                origin='lower', cmap='hot',
                                vmin=vmin_log, vmax=vmax_log)
            ax_top.set_title(f'a = {t_actual:.2f}', color='white', fontsize=14)
            if i == 0:
                ax_top.set_ylabel('y [Mpc/h]', color='white')
            plt.colorbar(im1, ax=ax_top, shrink=0.8, label='log₁₀(ρ/ρ̄)')

            ax_bottom = axes[1, i]
            im2 = ax_bottom.imshow(rho_2d.T, extent=[0, box.L, 0, box.L],
                                   origin='lower', cmap='viridis',
                                   vmin=0.5, vmax=3.0)
            ax_bottom.set_xlabel('x [Mpc/h]', color='white')
            if i == 0:
                ax_bottom.set_ylabel('y [Mpc/h]', color='white')
            plt.colorbar(im2, ax=ax_bottom, shrink=0.8, label='ρ/ρ̄')

        proj_label = " (z-projection)" if box.dim == 3 else ""
        fig.suptitle(f'Density Field Evolution{proj_label}', color='white', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_particles(all_states, box, times=None, n_particles=2000, save_path=None):
    """Plot particle positions (x-y projection). Works for 2D and 3D.

    Parameters
    ----------
    all_states  : State  — stacked trajectory
    box         : Box
    times       : list of float | None — target scale factors
    n_particles : int   — max particles to display (random subsample)
    save_path   : str | None
    """
    if times is None:
        t = np.asarray(all_states.time)
        times = [float(t[0]), float(t[len(t) // 2]), float(t[-1])]

    rng = np.random.default_rng(seed=0)   # reproducible subsample
    n_total = all_states.position.shape[1]
    particle_indices = rng.choice(n_total, size=min(n_particles, n_total), replace=False)

    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(1, len(times), figsize=(6 * len(times), 6))
        if len(times) == 1:
            axes = [axes]

        for i, t_target in enumerate(times):
            idx = np.argmin(np.abs(np.asarray(all_states.time) - t_target))
            pos = np.array(all_states.position[idx, particle_indices]) % box.L
            t_actual = float(all_states.time[idx])

            ax = axes[i]
            ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.8, c='cyan', edgecolors='none')
            ax.set_xlim(0, box.L)
            ax.set_ylim(0, box.L)
            ax.set_title(f'a = {t_actual:.2f}', color='white', fontsize=14)
            ax.set_xlabel('x [Mpc/h]', color='white')
            if i == 0:
                ax.set_ylabel('y [Mpc/h]', color='white')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        proj_label = " (x-y projection)" if box.dim == 3 else ""
        fig.suptitle(f'Particle Evolution{proj_label}', color='white', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
