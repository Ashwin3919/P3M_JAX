#!/usr/bin/env python3
import argparse
import os
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

# Import custom modules (updated for src/ structure)
from src.core.box import Box
from src.core.filters import Power_law, Scale, Cutoff, Potential
from src.core.ops import garfield, md_cic_nd
from src.physics.cosmology import Cosmology
from src.physics.system import PoissonVlasov
from src.physics.initial_conds import Zeldovich
from src.solver.integrator import iterate_step_scan, step_chunk
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
    B_m = Box(dim, config['N'], config['L'])
    force_box = Box(dim, config['N'] * 2, B_m.L)
    
    # 4. Generate Initial Conditions
    print("Generating initial conditions...")
    Power_spectrum = (Power_law(config['power_index']) * Scale(B_m, 0.2) * Cutoff(B_m))
    phi = (garfield(B_m, Power_spectrum, Potential(), config['seed']) * config['A']).astype(dtype)

    za = Zeldovich(B_m, force_box, cosmo, phi)
    state = za.state(config['a_start'])
    state = state._replace(
        position=state.position.astype(dtype),
        momentum=state.momentum.astype(dtype),
    )
    
    # 5. Setup System and Integrator
    system = PoissonVlasov(force_box, cosmo, za.particle_mass)
    save_every = config.get('save_every', 1)
    n_steps_raw = int((config['a_end'] - config['a_start']) / config['dt'])
    n_steps  = (n_steps_raw // save_every) * save_every   # round to exact multiple
    n_chunks = n_steps // save_every

    chunk_fn = jax.jit(partial(step_chunk, system, dt=config['dt'], save_every=save_every))

    # 6. Run simulation — save VTK + CSV after every chunk so progress is never lost
    ps_csv_path = os.path.join(results_dir, "power_spectrum.csv")
    saved_times, saved_pos, saved_mom = [], [], []

    print(f"Running simulation: {n_steps} steps, snapshot every {save_every} ({n_chunks} chunks)...")
    start_time = time.time()

    for chunk_idx in range(n_chunks):
        state = chunk_fn(state)
        jax.block_until_ready(state)

        a_val   = float(state.time)
        pos_np  = np.array(state.position)
        mom_np  = np.array(state.momentum)

        saved_times.append(a_val)
        saved_pos.append(pos_np)
        saved_mom.append(mom_np)

        # --- incremental I/O ---
        x_grid = state.position / B_m.res
        rho = np.array(md_cic_nd(B_m.shape, x_grid) + 1.0)

        if config.get('save_vtk'):
            write_vtk_particles(pos_np, mom_np, a_val, results_dir, config['name'])
            write_vtk_density(rho, B_m, a_val, results_dir, config['name'])

        if config.get('save_power_spectrum'):
            k, Pk = compute_power_spectrum(rho, B_m.L, particle_count=len(pos_np))
            append_to_csv(ps_csv_path, chunk_idx, a_val, k, Pk)

        print(f"  [{chunk_idx + 1}/{n_chunks}] a = {a_val:.3f}")

    runtime = time.time() - start_time
    print(f"Simulation completed in {runtime:.2f} seconds!")

    # Reconstruct all_states for the end-of-run plots
    all_states = State(
        time=jnp.array(saved_times),
        position=jnp.array(saved_pos),
        momentum=jnp.array(saved_mom),
    )

    # 7. End-of-run plots (PNGs only, data already saved above)
    print("Saving analysis results...")
    key_times = [config['a_start'], (config['a_start'] + config['a_end']) / 2, config['a_end']]

    plot_density_evolution(all_states, B_m, key_times,
                           save_path=os.path.join(results_dir, 'density_evolution.png'))
    plot_particles(all_states, B_m, key_times,
                   save_path=os.path.join(results_dir, 'particle_evolution.png'))

    if config.get('save_power_spectrum'):
        print("Generating power spectrum evolution plot...")
        plot_power_spectrum_evolution(ps_csv_path, os.path.join(results_dir, "power_spectrum.png"))
    
    print(f"Results saved in {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()
    run_simulation(args.config)
