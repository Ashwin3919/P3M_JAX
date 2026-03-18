# P2M-JAX N-Body Simulation

A high-performance, modular 2D N-Body simulation using **JAX**. This project uses a modern `src/` layout and a JIT-compiled, config-driven architecture for professional scientific computing.

## 🏗 Project Structure

- **`main.py`**: Entry point. Loads config and runs the JIT-compiled simulation loop.
- **`src/`**: Core library code.
  - **`core/`**: JAX kernels (CIC deposition, Force interpolation, GRFs).
  - **`physics/`**: Cosmology parameters, Poisson-Vlasov Hamiltonian, and ICs.
  - **`solver/`**: Immutable State management and `lax.scan` integrators.
  - **`utils/`**: Plotting, Power Spectrum Analysis, and VTK IO.
- **`configs/`**: JSON files for simulation parameters.
- **`results/`**: Outputs (plots and VTK data) categorized by config name.
- **`tests/`**: Unit tests for core simulation kernels.

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running Simulations
Run with a specific configuration:
```bash
python main.py --config configs/default.json
```

### 3. Running Tests
Verify the core operations:
```bash
pytest tests/
```

## 📊 Outputs & Analysis

### VTK Visualization
If enabled in config, binary VTK files (including momenta) are saved in:
- `results/<config>/vtk/particles/`
- `results/<config>/vtk/density/`

### Power Spectrum
The simulation computes the matter power spectrum evolution, saving a CSV and a publication-quality plot using LaTeX-style fonts (`STIX`).

---
*Developed for high-performance cosmological P2M/P3M research.*
