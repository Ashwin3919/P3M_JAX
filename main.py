#!/usr/bin/env python3
import argparse
import os
import time
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

from src.core.box import Box
from src.core.filters import Power_law, Scale, Cutoff, Potential
from src.core.ops import garfield, md_cic_nd
from src.physics.cosmology import Cosmology
from src.physics.system import PoissonVlasov
from src.physics.initial_conds import Zeldovich
from src.solver.integrator import iterate_step_scan, step_chunk, step_chunk_adaptive, compute_dt
from src.solver.state import State
from src.utils.plotting import plot_density_evolution, plot_particles
from src.utils.config_parser import load_config, get_results_dir
from src.utils.io import write_vtk_particles, write_vtk_density
from src.utils.analysis import compute_power_spectrum, append_to_csv, plot_power_spectrum_evolution

_DTYPE_MAP = {
    'float16': jnp.float16,
    'float32': jnp.float32,
    'float64': jnp.float64,
}

def run_simulation(config_path):
    # 1. Load configuration
    config = load_config(config_path)
    results_dir = get_results_dir(config['name'])

    # Resolve precision
    precision_str = config.get('precision', 'float64')
    if precision_str not in _DTYPE_MAP:
        raise ValueError(f"precision must be one of {list(_DTYPE_MAP)}, got '{precision_str}'")
    dtype = _DTYPE_MAP[precision_str]
    if precision_str == 'float64':
        jax.config.update("jax_enable_x64", True)
    if precision_str == 'float16':
        print("WARNING: float16 precision is numerically unstable for N-body simulations. Use float32 or float64.")

    print(f"=== Starting N-Body Simulation: {config['name']} ===")
    print(f"Precision: {precision_str}")
    print(f"Results will be saved to: {results_dir}")

    # 2. Setup cosmology
    cosmo = Cosmology(
        H0=config.get('H0', 70.0),
        OmegaM=config.get('OmegaM', 1.0),
        OmegaL=config.get('OmegaL', 0.0)
    )
    print(f"Cosmology: H0={cosmo.H0}, OmegaM={cosmo.OmegaM}, OmegaL={cosmo.OmegaL}")

    # 3. Setup boxes
    dim = config.get('dim', 2)
    B_mass = Box(dim, config['N'], config['L'])
    force_box = Box(dim, config['N'] * 2, B_mass.L)

    # 4. Generate Initial Conditions
    print("Generating initial conditions...")
    Power_spectrum = (Power_law(config['power_index']) * Scale(B_mass, 0.2) * Cutoff(B_mass))
    phi = (garfield(B_mass, Power_spectrum, Potential(), config['seed']) * config['A']).astype(dtype)

    za = Zeldovich(B_mass, force_box, cosmo, phi)
    state = za.state(config['a_start'])
    state = state._replace(
        position=state.position.astype(dtype),
        momentum=state.momentum.astype(dtype),
    )

    # 5. Setup system (solver selection)
    solver     = config.get('solver', 'pm')
    pp_window  = config.get('pp_window', 2)
    pp_soft    = config.get('pp_softening', 0.1)
    pp_cutoff  = config.get('pp_cutoff', 2.5)
    system = PoissonVlasov(force_box, cosmo, za.particle_mass,
                           solver=solver, pp_window=pp_window,
                           pp_softening=pp_soft, pp_cutoff=pp_cutoff)
    print(f"Solver: {solver.upper()}")

    # 6. Setup integrator (fixed or adaptive)
    timestepping = config.get('timestepping', 'fixed')
    save_every   = config.get('save_every', 1)
    vtk_freq     = config.get('vtk_freq', 1)   # save VTK every N chunks

    if timestepping == 'adaptive':
        C_cfl   = config.get('C_cfl', 0.3)
        dt_min  = config.get('dt_min', 0.001)
        dt_max  = config.get('dt_max', 0.05)
        eps     = pp_soft   # use PP softening as the resolution scale for CFL
        n_chunks = config.get('n_chunks', 50)   # number of output checkpoints
        chunk_da = (config['a_end'] - config['a_start']) / n_chunks

        chunk_fn = jax.jit(
            partial(step_chunk_adaptive, system,
                    C_cfl=C_cfl, eps=eps, dt_min=dt_min, dt_max=dt_max)
        )
        print(f"Timestepping: ADAPTIVE  C_cfl={C_cfl}  dt=[{dt_min}, {dt_max}]  "
              f"n_chunks={n_chunks}")

    else:   # fixed dt
        dt = config['dt']
        # Use round() to avoid float-division truncation (e.g. 0.98/0.02 → 48 not 49)
        n_steps_raw = round((config['a_end'] - config['a_start']) / dt)
        n_steps  = (n_steps_raw // save_every) * save_every
        n_chunks = n_steps // save_every
        chunk_da = save_every * dt

        chunk_fn = jax.jit(partial(step_chunk, system, dt=dt, save_every=save_every))
        print(f"Timestepping: FIXED  dt={dt}  steps={n_steps}  chunks={n_chunks}")

    # 7. Run simulation — save VTK + CSV after every chunk
    ps_csv_path = os.path.join(results_dir, "power_spectrum.csv")
    saved_times, saved_pos, saved_mom = [], [], []

    print(f"Running {n_chunks} chunks...")
    start_time = time.time()

    for chunk_idx in range(n_chunks):
        if timestepping == 'adaptive':
            a_target = config['a_start'] + (chunk_idx + 1) * chunk_da
            state = chunk_fn(state, a_target)
        else:
            state = chunk_fn(state)
        jax.block_until_ready(state)

        a_val  = float(state.time)
        pos_np = np.array(state.position)
        mom_np = np.array(state.momentum)

        # NaN/Inf guard — catches numerical blow-ups immediately
        if not np.all(np.isfinite(pos_np)):
            raise RuntimeError(
                f"NaN/Inf detected in particle positions at chunk {chunk_idx + 1}, "
                f"a = {a_val:.4f}. Consider reducing dt or the amplitude A."
            )

        saved_times.append(a_val)
        saved_pos.append(pos_np)
        saved_mom.append(mom_np)

        # Density on mass grid for I/O
        x_grid = state.position / B_mass.res
        rho = np.array(md_cic_nd(B_mass.shape, x_grid) + 1.0)

        if config.get('save_vtk') and chunk_idx % vtk_freq == 0:
            write_vtk_particles(pos_np, mom_np, a_val, results_dir, config['name'])
            write_vtk_density(rho, B_mass, a_val, results_dir, config['name'])

        if config.get('save_power_spectrum'):
            k, Pk = compute_power_spectrum(rho, B_mass.L, particle_count=len(pos_np))
            append_to_csv(ps_csv_path, chunk_idx, a_val, k, Pk)

        # Adaptive: print current dt alongside progress
        if timestepping == 'adaptive':
            dt_est = float(compute_dt(state, cosmo, C_cfl, eps, dt_min, dt_max))
            print(f"  [{chunk_idx + 1}/{n_chunks}] a = {a_val:.3f}  dt_est = {dt_est:.4f}")
        else:
            print(f"  [{chunk_idx + 1}/{n_chunks}] a = {a_val:.3f}")

    runtime = time.time() - start_time
    print(f"Simulation completed in {runtime:.2f} seconds!")

    # Reconstruct all_states for end-of-run plots.
    # Guard against OOM on large 3D runs: stacking all chunks into a single array
    # can exceed available RAM (e.g. N=128^3, 50 chunks ≈ 25 GB). Skip plotting
    # in that case and rely on the per-chunk VTK files instead.
    _bytes_per_elem = 8 if precision_str == 'float64' else 4
    _traj_bytes = len(saved_pos) * len(saved_pos[0]) * dim * 2 * _bytes_per_elem  # pos+mom
    oom_threshold_gb = config.get('oom_threshold_gb', 4.0)
    if _traj_bytes > oom_threshold_gb * 1024**3:
        print(f"Trajectory too large to stack ({_traj_bytes / 1024**3:.1f} GB > "
              f"{oom_threshold_gb} GB). Skipping end-of-run plots.")
        if config.get('save_power_spectrum'):
            _save_ps_plot(ps_csv_path, results_dir)
    else:
        all_states = State(
            time=jnp.array(saved_times),
            position=jnp.array(saved_pos),
            momentum=jnp.array(saved_mom),
        )

        # 8. End-of-run plots
        print("Saving analysis results...")
        key_times = [config['a_start'], (config['a_start'] + config['a_end']) / 2, config['a_end']]

        plot_density_evolution(all_states, B_mass, key_times,
                               save_path=os.path.join(results_dir, 'density_evolution.png'))
        plot_particles(all_states, B_mass, key_times,
                       save_path=os.path.join(results_dir, 'particle_evolution.png'))

        if config.get('save_power_spectrum'):
            _save_ps_plot(ps_csv_path, results_dir)

    print(f"Results saved in {results_dir}")


def _save_ps_plot(ps_csv_path, results_dir):
    """Write the power spectrum evolution plot from the accumulated CSV."""
    print("Generating power spectrum evolution plot...")
    plot_power_spectrum_evolution(ps_csv_path, os.path.join(results_dir, "power_spectrum.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()
    run_simulation(args.config)
