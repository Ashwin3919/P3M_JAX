import matplotlib.pyplot as plt
import numpy as np
from core.ops import md_cic_2d

def plot_density_evolution(states, box, times=[0.02, 0.5, 2.0], save_path=None):
    """Plot density field evolution"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, len(times), figsize=(5 * len(times), 10))
    if len(times) == 1:
        axes = axes.reshape(-1, 1)

    state_indices = []
    for t in times:
        idx = min(range(len(states)), key=lambda i: abs(states[i].time - t))
        state_indices.append(idx)

    for i, idx in enumerate(state_indices):
        state = states[idx]
        x_grid = state.position / box.res
        rho = md_cic_2d(box.shape, x_grid)
        rho = rho + 1.0
        rho_np = np.array(rho)

        ax_top = axes[0, i]
        log_rho = np.log10(np.maximum(rho_np, 0.1))
        im1 = ax_top.imshow(log_rho.T, extent=[0, box.L, 0, box.L],
                            origin='lower', cmap='hot', vmin=-0.3, vmax=0.8)
        ax_top.set_title(f'a = {state.time:.2f}', color='white', fontsize=14)
        if i == 0: ax_top.set_ylabel('y [Mpc/h]', color='white')
        plt.colorbar(im1, ax=ax_top, shrink=0.8, label='log₁₀(ρ/ρ̄)')

        ax_bottom = axes[1, i]
        im2 = ax_bottom.imshow(rho_np.T, extent=[0, box.L, 0, box.L],
                               origin='lower', cmap='viridis', vmin=0.5, vmax=3.0)
        ax_bottom.set_xlabel('x [Mpc/h]', color='white')
        if i == 0: ax_bottom.set_ylabel('y [Mpc/h]', color='white')
        plt.colorbar(im2, ax=ax_bottom, shrink=0.8, label='ρ/ρ̄')

    fig.suptitle('Density Field Evolution', color='white', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_particles(states, box, times=[0.02, 0.5, 2.0], n_particles=2000, save_path=None):
    """Plot particle positions"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, len(times), figsize=(6 * len(times), 6))
    if len(times) == 1:
        axes = [axes]

    state_indices = []
    for t in times:
        idx = min(range(len(states)), key=lambda i: abs(states[i].time - t))
        state_indices.append(idx)

    n_total = len(states[0].position)
    particle_indices = np.random.choice(n_total, size=min(n_particles, n_total), replace=False)

    for i, idx in enumerate(state_indices):
        state = states[idx]
        ax = axes[i]
        pos = np.array(state.position[particle_indices] % box.L)
        ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.8, c='cyan', edgecolors='none')
        ax.set_xlim(0, box.L)
        ax.set_ylim(0, box.L)
        ax.set_title(f'a = {state.time:.2f}', color='white', fontsize=14)
        ax.set_xlabel('x [Mpc/h]', color='white')
        if i == 0: ax.set_ylabel('y [Mpc/h]', color='white')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Particle Evolution', color='white', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
