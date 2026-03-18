# P2M-JAX N-Body Simulation

A high-performance, modular 2D N-Body simulation using **JAX**. This project refactors Johan Hidding's implementation into a modular, config-driven architecture following Google's internal engineering standards for scientific computing.

## 🏗 Project Structure

- **`main.py`**: The entry point. Loads config, initializes ICs, and runs the simulation.
- **`core/`**: JAX-optimized kernels (CIC deposition, Force interpolation, Gaussian Random Fields).
- **`physics/`**: Cosmological background (LCDM/EdS) and the Poisson-Vlasov system.
- **`solver/`**: Generic Hamiltonian integration framework (Leap-frog).
- **`configs/`**: JSON configuration files for different resolutions and seeds.
- **`results/`**: Outputs (plots and data) are automatically stored here by config name.
- **`utils/`**: Plotting and configuration parsing logic.

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Running Simulations
Run the default simulation (128 res, EdS cosmology):
```bash
python main.py --config configs/default.json
```

Run a high-resolution simulation (256 res, LCDM cosmology):
```bash
python main.py --config configs/high_res.json
```

### 3. Creating New Configs
To run a custom simulation, create a new JSON file in `configs/`:
```json
{
    "N": 64,
    "L": 50.0,
    "A": 8.0,
    "seed": 123,
    "a_start": 0.02,
    "a_end": 1.0,
    "dt": 0.02,
    "power_index": -0.5,
    "cosmology": "EdS"
}
```
Then run it with:
```bash
python main.py --config configs/my_custom_sim.json
```

## 📊 Results
Plots for density evolution and particle positions are saved automatically in:
`results/<config_name>/density_evolution.png`
`results/<config_name>/particle_evolution.png`

---
*Developed for P2M/P3M research workflows.*
