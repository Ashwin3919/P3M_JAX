#!/usr/bin/env python3
import argparse
import os
import time
import jax
import matplotlib.pyplot as plt
from functools import partial

# Import custom modules
from core.box import Box
from core.filters import Power_law, Scale, Cutoff, Potential
from core.ops import garfield
from physics.cosmology import LCDM, EdS
from physics.system import PoissonVlasov
from physics.initial_conds import Zeldovich
from solver.integrator import leap_frog, iterate_step
from utils.plotting import plot_density_evolution, plot_particles
from utils.config_parser import load_config, get_results_dir

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

def run_simulation(config_path):
    # 1. Load configuration
    config = load_config(config_path)
    results_dir = get_results_dir(config['name'])
    
    print(f"=== Starting N-Body Simulation: {config['name']} ===")
    print(f"Results will be saved to: {results_dir}")
    
    # 2. Setup cosmology
    cosmo_map = {"LCDM": LCDM, "EdS": EdS}
    cosmo = cosmo_map.get(config.get('cosmology', 'EdS'), EdS)
    
    # 3. Setup boxes
    B_m = Box(2, config['N'], config['L'])
    force_box = Box(2, config['N'] * 2, B_m.L)
    
    # 4. Generate Initial Conditions
    print("Generating initial conditions using Zeldovich approximation...")
    Power_spectrum = (Power_law(config['power_index']) * 
                      Scale(B_m, 0.2) * 
                      Cutoff(B_m))
    phi = garfield(B_m, Power_spectrum, Potential(), config['seed']) * config['A']
    
    za = Zeldovich(B_m, force_box, cosmo, phi)
    state = za.state(config['a_start'])
    
    # 5. Setup System and Integrator
    system = PoissonVlasov(force_box, cosmo, za.particle_mass, 
                           live_plot=config.get('live_plot', False))
    state.live_plot = config.get('live_plot', False)
    if state.live_plot:
        state.fig = system.fig

    stepper = partial(leap_frog, config['dt'], system)
    
    # 6. Run Simulation
    print(f"Starting simulation with {len(state.position)} particles...")
    start_time = time.time()
    
    states = iterate_step(stepper, lambda s: s.time > config['a_end'], state)
    
    runtime = (time.time() - start_time) / 60
    print(f"\nSimulation completed in {runtime:.2f} minutes!")
    
    # 7. Visualization & Storage
    print("Saving final analysis plots...")
    key_times = [config['a_start'], (config['a_start'] + config['a_end']) / 2, config['a_end']]
    
    plot_density_evolution(states, B_m, key_times, 
                           save_path=os.path.join(results_dir, 'density_evolution.png'))
    plot_particles(states, B_m, key_times, 
                   save_path=os.path.join(results_dir, 'particle_evolution.png'))
    
    print(f"Results saved successfully in {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run modular P2M N-Body simulation.")
    parser.add_argument("--config", type=str, default="configs/default.json",
                        help="Path to the simulation configuration file.")
    args = parser.parse_args()
    
    run_simulation(args.config)
