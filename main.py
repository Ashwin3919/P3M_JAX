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
from src.core.ops import garfield, md_cic_2d
from src.physics.cosmology import Cosmology
from src.physics.system import PoissonVlasov
from src.physics.initial_conds import Zeldovich
from src.solver.integrator import iterate_step_scan
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
    B_m = Box(2, config['N'], config['L'])
    force_box = Box(2, config['N'] * 2, B_m.L)
    
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
    n_steps = int((config['a_end'] - config['a_start']) / config['dt'])
    
    # 6. Run JIT-compiled Simulation
    run_sim_jit = jax.jit(partial(iterate_step_scan, system, dt=config['dt'], n_steps=n_steps))
    
    print(f"Running simulation for {n_steps} steps...")
    start_time = time.time()
    final_state, all_states = run_sim_jit(state)
    jax.block_until_ready(final_state)
    
    runtime = time.time() - start_time
    print(f"Simulation completed in {runtime:.2f} seconds!")
    
    # 7. Visualization & Storage
    print("Saving analysis results...")
    key_times = [config['a_start'], (config['a_start'] + config['a_end']) / 2, config['a_end']]
    
    # Standard plots
    plot_density_evolution(all_states, B_m, key_times, 
                           save_path=os.path.join(results_dir, 'density_evolution.png'))
    plot_particles(all_states, B_m, key_times, 
                   save_path=os.path.join(results_dir, 'particle_evolution.png'))
    
    # Analysis & VTK
    ps_csv_path = os.path.join(results_dir, "power_spectrum.csv")
    
    # Save snapshots based on frequency or key times
    if config.get('save_vtk') or config.get('save_power_spectrum'):
        vtk_freq = config.get('vtk_freq', 10)
        indices = np.unique(np.append(np.arange(0, n_steps, vtk_freq), n_steps - 1))
        
        for idx in indices:
            a_val = float(all_states.time[idx])
            pos_val = all_states.position[idx]
            mom_val = all_states.momentum[idx]
            
            # Density for PS and VTK
            x_grid = pos_val / B_m.res
            rho = np.array(md_cic_2d(B_m.shape, x_grid) + 1.0)
            
            if config.get('save_vtk'):
                write_vtk_particles(pos_val, mom_val, a_val, results_dir, config['name'])
                write_vtk_density(rho, B_m, a_val, results_dir, config['name'])
            
            if config.get('save_power_spectrum'):
                k, Pk = compute_power_spectrum(rho, B_m.L, particle_count=len(pos_val))
                append_to_csv(ps_csv_path, int(idx), a_val, k, Pk)

    if config.get('save_power_spectrum'):
        print("Generating power spectrum evolution plot...")
        plot_power_spectrum_evolution(ps_csv_path, os.path.join(results_dir, "power_spectrum.png"))
    
    print(f"Results saved in {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()
    run_simulation(args.config)
