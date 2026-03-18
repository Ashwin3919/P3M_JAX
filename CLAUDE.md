# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation (specify config)
python main.py --config configs/default.json
python main.py --config configs/high_res.json

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_core.py::test_md_cic_2d_mass_conservation
```

## Architecture

This is a 2D P3M (Particle-Particle-Particle-Mesh) cosmological N-body simulation using JAX for GPU/CPU acceleration.

### Data Flow

```
Config JSON → Cosmology + Box setup → Zeldovich ICs → PoissonVlasov system → Leap-frog integrator → Output
```

### Key Design Patterns

**Immutable State**: `State(NamedTuple)` holds `(time, position, momentum)`. All integrator steps return new State objects — never mutated in-place.

**JAX compilation boundary**: `main.py` JIT-compiles `iterate_step_scan` once before the run loop. All physics code inside must be JAX-traceable (no Python control flow on array values, no side effects).

**lax.scan for time integration**: The leap-frog loop uses `jax.lax.scan` to avoid Python-level loops and enable XLA compilation. This returns the full trajectory `(final_state, all_states)` where `all_states` has shape `(n_steps, ...)`.

**Dual-resolution boxes**: Two `Box` instances are created — `B_m` (N×N for mass deposition) and `B_force` (2N×2N for force computation). Force resolution is 2× the mass grid to reduce aliasing.

### Module Responsibilities

| Module | Role |
|--------|------|
| `src/core/box.py` | `Box` class: periodic domain, FFT wavenumber grids (`K`, `k`), Nyquist freq |
| `src/core/ops.py` | CIC mass deposition `md_cic_2d()`, bilinear interpolation `Interp2D`, 2nd-order gradients, Gaussian random field `garfield()` |
| `src/core/filters.py` | Composable Fourier-space filters: `Power_law`, `Scale`, `Cutoff`, `Potential` (-1/k²). Support operator overloading (`*`, `+`, `^`, `~`) |
| `src/physics/cosmology.py` | `Cosmology` dataclass: H(a), linear growth factor D(a). Presets: `LCDM_PRESET` (Planck 2018), `EDS_PRESET` |
| `src/physics/system.py` | `PoissonVlasov`: equations of motion in comoving coordinates. Solves Poisson via FFT with precomputed potential kernel |
| `src/physics/initial_conds.py` | `Zeldovich` approximation: Gaussian random potential → displacement field → initial positions/momenta |
| `src/solver/state.py` | `State` NamedTuple + `HamiltonianSystem` ABC |
| `src/solver/integrator.py` | KDK leap-frog: `leap_frog()` (single step), `iterate_step_scan()` (scan over n_steps) |
| `src/utils/analysis.py` | `compute_power_spectrum()` with CIC deconvolution and shot noise subtraction |
| `src/utils/io.py` | VTK output (binary PolyData for particles, ASCII StructuredPoints for density) |

### Force Calculation Pipeline (inside `PoissonVlasov`)

1. CIC deposition: particle positions → density grid ρ(x)
2. Density contrast: δ = ρ/ρ̄ - 1
3. FFT: δ̂ = FFT(δ)
4. Gravitational potential: φ̂ = δ̂ × kernel (precomputed -1/k² filter)
5. IFFT: φ = IFFT(φ̂)
6. Gradient: ∇φ via 2nd-order finite differences with periodic BC
7. Interpolation: ∇φ(xᵢ) via bilinear interpolation at particle positions

### Equations of Motion (comoving coordinates)

- Position: dx/dt = p / (a² H(a))
- Momentum: dp/dt = -∇φ / H(a)

where `a` is the cosmological scale factor and `H(a)` is the Hubble parameter.

### Power Spectrum Output

`compute_power_spectrum()` applies:
- Annular k-bin averaging
- CIC window correction: P_corr = P / W² where W = sinc(k Δx / 2π)²
- Shot noise subtraction: P_true = P_meas - 1/n̄

Results appended to `results/{config_name}/power_spectrum.csv` with scale factor `a` column.

### Config Parameters

```json
{
  "N": 128,              // particles per side (N² total)
  "L": 50,               // box size [Mpc/h]
  "a_start": 0.02,       // initial scale factor
  "a_end": 1.0,          // final scale factor
  "dt": 0.02,            // timestep in scale factor
  "seed": 4,             // random seed for ICs
  "n_s": -0.5,           // primordial power spectrum index
  "precision": "float64", // float16 | float32 | float64
  "H0": 100, "OmegaM": 1.0, "OmegaL": 0.0  // cosmology
}
```

### Precision

Set via `"precision"` in the config. Affects particle positions, momenta, density arrays, and the initial potential field.

| Value | x64 enabled | Use case |
|-------|-------------|----------|
| `"float64"` | yes | Default; required for accurate gravitational potentials |
| `"float32"` | no | Faster on GPU/MPS; ~2× memory reduction; acceptable for large N |
| `"float16"` | no | Experimental; numerically unstable — emits a warning at startup |

`jax_enable_x64` is enabled automatically only when `precision = "float64"`. `md_cic_2d` infers its output dtype from the input position array, so it naturally follows the chosen precision.
