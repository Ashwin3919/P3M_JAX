# P3M-JAX: Cosmological N-Body Simulation

A JAX-accelerated, dimension-agnostic Particle-Particle-Particle-Mesh (P3M) N-body code for cosmological simulations. Supports 2D and 3D domains, pure PM and full P3M force computation, fixed and adaptive time-stepping — all controlled by a single JSON config file.

Runs without modification on CPU, NVIDIA GPU, and Apple Silicon (MPS).

---

## Project Structure

```
P3M_JAX/
├── main.py                     # Entry point
├── configs/                    # JSON simulation configs
│   ├── default.json            # 2D EdS, N=128, float64, PM, fixed dt
│   ├── high_res.json           # 2D LCDM, N=256, float32, PM, fixed dt
│   ├── 3d_default.json         # 3D EdS, N=64, float32, PM, fixed dt
│   ├── 3d_heigh_res.json       # 3D LCDM, N=128, float32, PM, fixed dt
│   └── p3m_adaptive.json       # 2D EdS, N=64, P3M + adaptive dt (demo)
├── src/
│   ├── core/                   # JAX kernels: CIC, interpolation, GRFs, filters
│   ├── physics/                # Cosmology, PoissonVlasov system, Zeldovich ICs
│   ├── solver/                 # State, leapfrog integrator, lax.scan / while_loop
│   └── utils/                  # Power spectrum, VTK I/O, plotting
├── tests/
│   └── test_core.py            # 17 unit tests
└── results/                    # Auto-created; outputs organised by config name
```

---

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

---

## Running Simulations

```bash
python main.py --config configs/default.json
```

All configs use `"solver": "pm"` by default. To use the P3M solver or adaptive stepping, either edit an existing config or use `p3m_adaptive.json`:

```bash
python main.py --config configs/p3m_adaptive.json
```

### Available Configurations

| Config | dim | N | Solver | Stepping | Precision | Notes |
|--------|-----|---|--------|----------|-----------|-------|
| `default.json` | 2 | 128 | PM | fixed dt=0.02 | float64 | Recommended starting point |
| `high_res.json` | 2 | 256 | PM | fixed dt=0.015 | float32 | Power spectrum convergence |
| `3d_default.json` | 3 | 64 | PM | fixed dt=0.02 | float32 | 3D validation |
| `3d_heigh_res.json` | 3 | 128 | PM | fixed dt=0.02 | float32 | Production 3D run |
| `p3m_adaptive.json` | 2 | 64 | P3M | adaptive | float64 | P3M + adaptive dt demo |

---

## Config Parameters

```json
{
  "dim": 2,              // spatial dimension: 2 or 3
  "N": 128,              // particles per side  (N^dim total particles)
  "L": 50.0,             // box side length [Mpc/h]
  "A": 10.0,             // initial displacement field amplitude
  "seed": 4,             // random seed
  "a_start": 0.02,       // initial scale factor
  "a_end": 1.0,          // final scale factor
  "power_index": -0.5,   // primordial spectral index n_s
  "H0": 70.0,            // Hubble constant [km/s/Mpc]
  "OmegaM": 1.0,         // matter density
  "OmegaL": 0.0,         // cosmological constant
  "precision": "float64", // "float16" | "float32" | "float64"

  // --- Force solver ---
  "solver": "pm",        // "pm" (default) | "p3m"
  "pp_window": 4,        // P3M: sliding window half-width W (neighbours = 2W+1)
  "pp_softening": 0.2,   // P3M: gravitational softening length [Mpc/h]
  "pp_cutoff": 2.5,      // P3M: PP cutoff in units of force grid cell size

  // --- Time-stepping (fixed) ---
  "dt": 0.02,            // fixed Δa step  (used when timestepping="fixed")
  "save_every": 1,       // leapfrog steps between saved snapshots

  // --- Time-stepping (adaptive) ---
  "timestepping": "fixed", // "fixed" | "adaptive"
  "C_cfl": 0.3,           // CFL safety factor
  "dt_min": 0.001,        // minimum Δa
  "dt_max": 0.05,         // maximum Δa
  "n_chunks": 50,         // number of output checkpoints (adaptive only)

  // --- Output ---
  "save_vtk": true,
  "vtk_freq": 1,         // write VTK every N chunks  (1 = every chunk)
  "save_power_spectrum": true
}
```

---

## Output

All output is written to `results/<config_name>/`:

```
results/default/
├── density_evolution.png       # 3-panel density map at early/mid/late times
├── particle_evolution.png      # 3-panel particle scatter plot
├── power_spectrum.csv          # P(k) at every saved timestep
├── power_spectrum.png          # P(k) evolution coloured by scale factor
└── vtk/
    ├── particles/              # Binary VTK PolyData (positions + momenta)
    └── density/                # ASCII VTK StructuredPoints (density field)
```

VTK files are compatible with ParaView 5.x+. Load all particle snapshots as a time series via **File → Open** on the `vtk/particles/` directory.

---

## Running Tests

```bash
pytest tests/
```

17 unit tests cover CIC mass conservation, interpolation, Box wave numbers, gradient accuracy, 3D infrastructure, PM/P3M force correctness, erfc force splitting, and adaptive time-step bounds.

---

## Generalisation Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dimension-agnostic Box, filters, FFT | Done |
| 2 | ND CIC deposition and interpolation | Done |
| 3 | Zeldovich ICs for 3D | Done |
| 4 | PoissonVlasov 3D PM solver | Done |
| 5 | TSC mass assignment + deconvolved Green's function | Not done |
| 6 | Short-range PP forces via Morton Z-curve + erfc splitting | Done |
| 7 | Adaptive time-stepping via lax.while_loop + CFL condition | Done |
