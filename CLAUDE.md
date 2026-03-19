# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation (specify config)
python main.py --config configs/default.json          # 2D PM, fixed dt
python main.py --config configs/high_res.json         # 2D PM, higher N
python main.py --config configs/3d_default.json       # 3D PM, fixed dt
python main.py --config configs/3d_heigh_res.json     # 3D PM, higher N
python main.py --config configs/p3m_adaptive.json     # 2D P3M + adaptive dt
python main.py --config configs/3d_heigh_res_p3m.json # 3D P3M

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_core.py::test_md_cic_2d_mass_conservation
```

## Generalisation Roadmap — Status

Phase 1 (dimension-agnostic infrastructure): DONE
Phase 2 (ND CIC deposition and interpolation): DONE
Phase 3 (Zeldovich ICs for 3D): DONE
Phase 4 (PoissonVlasov 3D PM solver): DONE
Phase 5 (TSC mass assignment + deconvolved Green's function): NOT DONE
Phase 6 (short-range PP forces via Morton Z-curve + sliding window): DONE
Phase 7 (adaptive timestepping via lax.while_loop): DONE

## Architecture

This is an N-dimensional (2D and 3D) P3M (Particle-Particle-Particle-Mesh) cosmological N-body simulation using JAX for GPU/CPU acceleration. The `dim` parameter in the config controls dimensionality throughout.

### Data Flow

```
Config JSON → Cosmology + Box setup → Zeldovich ICs → PoissonVlasov system → Leap-frog integrator → Output
```

### Key Design Patterns

**Immutable State**: `State(NamedTuple)` holds `(time, position, momentum)`. All integrator steps return new State objects — never mutated in-place.

**JAX compilation boundary**: `main.py` JIT-compiles `step_chunk` or `step_chunk_adaptive` once before the chunk loop. All physics code inside must be JAX-traceable (no Python control flow on array values, no side effects). The solver branch (`pm` vs `p3m`) is a Python-level `if` resolved at trace time — zero runtime cost.

**lax.scan for time integration**: The leap-frog loop inside each chunk uses `jax.lax.scan`. `iterate_step_scan` runs the full trajectory in a single scan; `step_chunk` runs exactly `save_every` steps per call and is what `main.py` actually uses in the chunk loop.

**Dual-resolution boxes**: Two `Box` instances are created — `B_mass` (N×N[×N] for mass deposition and output) and `force_box` (2N×2N[×2N] for force computation). Force resolution is 2× the mass grid to reduce aliasing.

**Chunk loop in main.py**: Rather than one big scan, `main.py` runs a Python `for chunk_idx in range(n_chunks)` loop, calling the JIT-compiled chunk function each iteration. This allows per-chunk I/O (VTK, CSV, plots) without holding the full trajectory in memory.

### Module Responsibilities

| Module | Role |
|--------|------|
| `src/core/box.py` | `Box` class: periodic domain, FFT wavenumber grids (`K`, `k`), Nyquist/min freq, grid resolution |
| `src/core/ops.py` | ND CIC mass deposition `md_cic_nd()` (alias: `md_cic_2d`), ND interpolation `InterpND` (alias: `Interp2D`), 2nd-order finite-difference gradient `gradient_2nd_order()`, Gaussian random field `garfield()` |
| `src/core/filters.py` | Composable Fourier-space filters: `Power_law`, `Scale`, `Cutoff`, `Potential` (-1/k²), `Identity`, `Zero`. Operator overloading: `*` (product), `+` (sum), `**` (power), `~` (conjugate), `/` (ratio). Inner product `cc()` and cross-product `cf()` methods. |
| `src/physics/cosmology.py` | `Cosmology` dataclass: `da(a)` = aḢ(a), `G` = (3/2)ΩM H0², `growing_mode(a)` (diagnostic only, not called by solver). Presets: `LCDM_PRESET` (Planck 2018: H0=68, ΩM=0.31, ΩL=0.69), `EDS_PRESET` (H0=70, ΩM=1, ΩL=0) |
| `src/physics/system.py` | `PoissonVlasov`: PM force (`_pm_force`), PP correction (`_pp_force`), Morton encoding (`_morton_encode`). `solver='pm'` uses PM only; `solver='p3m'` adds PP short-range correction. |
| `src/physics/initial_conds.py` | `Zeldovich`: Gaussian random potential → displacement field `u` → initial positions/momenta. `particle_mass` property returns `(N_force/N_mass)^dim`. |
| `src/solver/state.py` | `State(NamedTuple)`: `(time, position, momentum)`. `HamiltonianSystem` ABC with `positionEquation` / `momentumEquation`. |
| `src/solver/integrator.py` | KDK leap-frog `leap_frog()` (single step). `iterate_step_scan()` (full scan). `step_chunk()` (fixed dt, JIT target). `step_chunk_adaptive()` (CFL dt, lax.while_loop). `compute_dt()` (CFL estimate). |
| `src/utils/analysis.py` | `compute_power_spectrum()` with CIC deconvolution and shot noise subtraction. `append_to_csv()`. `plot_power_spectrum_evolution()`. |
| `src/utils/io.py` | `write_vtk_particles()` (binary Legacy VTK PolyData, float32 big-endian). `write_vtk_density()` (ASCII VTK StructuredPoints, Fortran ordering). 2D positions padded to (N,3) with z=0. |
| `src/utils/config_parser.py` | `load_config()`: JSON load + required-key validation + physical-range checks. `get_results_dir()`. |
| `src/utils/plotting.py` | `plot_density_evolution()`, `plot_particles()` — end-of-run matplotlib figures. |

### Force Calculation Pipeline (inside `PoissonVlasov.momentumEquation`)

**PM step (always active):**
1. CIC deposition: particle positions → density grid ρ(x) via `md_cic_nd`
2. Density contrast: δ = ρ × particle_mass - 1
3. FFT: δ̂ = FFT(δ)
4. Gravitational potential: φ̂ = δ̂ × kernel (precomputed `-1/k²` filter, set at `__init__`)
5. IFFT + scale: φ = IFFT(φ̂).real × G / a
6. Gradient: ∇φ via `gradient_2nd_order` (4-point stencil, periodic) along each axis
7. Interpolation: ∇φ(xᵢ) via `InterpND` at particle positions → PM acceleration

**PP step (only when `solver='p3m'`):**
1. Convert positions to grid units; Morton-encode to Z-curve integer codes
2. Sort particles by Morton code (`jnp.argsort`); compute inverse permutation
3. For each particle `i` (via `jax.vmap`): gather `±W` neighbours from sorted array
4. Apply minimum-image convention for periodic BC
5. Compute `erfc(r/alpha)` splitting kernel (alpha = r_cut/2.6)
6. Mask: exclude self, out-of-range indices, and pairs beyond r_cut
7. Accumulate PP acceleration; un-sort back to original particle order

**Total acceleration:** `-(pm_acc + pp_acc) / da` for P3M, `-pm_acc / da` for PM.

### Equations of Motion (comoving coordinates)

- Position: `dx/da = p / (a² H(a))`  [`positionEquation`]
- Momentum: `dp/da = -(∇φ_PM + ∇φ_PP) / H(a)`  [`momentumEquation`]

where `a` is the cosmological scale factor, `da = a·H(a)` (from `cosmology.da`), and `G = (3/2) ΩM H0²`.

### KDK Leap-frog

Each `leap_frog` call does: half-kick → drift → half-kick. Time `s.time` advances by `dt` (the scale factor step). The `step_chunk` function wraps this in a `lax.scan` of length `save_every` for efficiency.

### Adaptive Timestepping (`timestepping: "adaptive"`)

`step_chunk_adaptive` uses `lax.while_loop` to advance from the current `a` to `a_target`. The step size is recomputed each iteration via `compute_dt`:

```
dt = C_cfl × eps / v_max   clipped to [dt_min, dt_max]
```

where `v_i = |p_i| / (a² H(a))` is the comoving drift speed and `eps` = pp_softening. The final sub-step is clamped to land exactly on `a_target`.

### Initial Conditions (`Zeldovich`)

The potential `phi` is generated by `garfield()` with filters `Power_law(n_s) * Scale(B, 0.2) * Cutoff(B)`, scaled by amplitude `A`. The Zeldovich displacement is `u = -∇phi / res` (grid units). Initial state:

- Positions: `X = grid_positions + a_start × u`
- Momenta: `P = a_start × u`

Note: `P = a_start × u` is exact for EdS (f=1) but omits the `f(a)·H(a)·D(a)` prefactor for LCDM.

### Power Spectrum Output

`compute_power_spectrum()` applies:
- Density contrast: δ = ρ - 1 (expects normalised density ρ = 1 + δ)
- FFT, `P(k) = |δ̂|² × V / N^(2d)`
- CIC deconvolution: `P → P / W²` where `W = ∏ sinc(kᵢ Δx / 2π)`
- Shot noise subtraction: `P → P - 1/n̄`  (if `particle_count` provided)
- Logarithmic k-binning, minimum 5 modes per bin, invalid bins dropped

Results appended to `results/{config_name}/power_spectrum.csv` with columns `[step, a, k, Pk]`.

### VTK Output

- **Particles** (`vtk/particles/`): Legacy binary PolyData, big-endian float32. POINT_DATA contains momentum vectors. 2D positions padded to (N,3) with z=0.
- **Density** (`vtk/density/`): ASCII StructuredPoints. Values written in Fortran (x-fastest) order. Both 2D and 3D supported.
- Filenames: `{config_name}_particles_a{a_str}.vtk`, `{config_name}_density_a{a_str}.vtk`

### Config Parameters — Full Reference

```json
{
  "dim": 2,                  // spatial dimension: 2 or 3
  "N": 128,                  // particles per side (N^dim total); mass grid is N^dim, force grid is (2N)^dim
  "L": 50.0,                 // box size [Mpc/h]
  "A": 10.0,                 // amplitude of initial potential field
  "a_start": 0.02,           // initial scale factor
  "a_end": 1.0,              // final scale factor
  "power_index": -0.5,       // primordial power spectrum index n_s (used as Power_law(n_s))
  "seed": 4,                 // PRNG seed for garfield() white noise

  // --- Cosmology (all optional, defaults shown) ---
  "H0": 70.0,                // Hubble constant [km/s/Mpc]
  "OmegaM": 1.0,             // matter density parameter
  "OmegaL": 0.0,             // cosmological constant density parameter

  // --- Precision ---
  "precision": "float64",    // "float16" | "float32" | "float64"

  // --- Solver ---
  "solver": "pm",            // "pm" (long-range only) | "p3m" (PM + PP short-range)
  "pp_window": 2,            // half-width of Morton sliding window (p3m only)
  "pp_softening": 0.1,       // gravitational softening length [Mpc/h] (p3m only)
  "pp_cutoff": 2.5,          // PP cutoff in units of force grid resolution (p3m only); must satisfy r_cut < L/2

  // --- Fixed timestepping (timestepping omitted or "fixed") ---
  "dt": 0.02,                // timestep in scale factor units
  "save_every": 1,           // emit one snapshot per this many leapfrog steps; n_steps must be divisible

  // --- Adaptive timestepping ---
  "timestepping": "adaptive",
  "C_cfl": 0.3,              // CFL safety factor
  "dt_min": 0.001,           // minimum allowed dt
  "dt_max": 0.05,            // maximum allowed dt
  "n_chunks": 50,            // number of output checkpoints

  // --- Output ---
  "save_vtk": true,          // write VTK particle + density files
  "vtk_freq": 1,             // write VTK every N chunks (1 = every chunk)
  "save_power_spectrum": true, // compute and save P(k) each chunk
  "oom_threshold_gb": 4.0    // skip end-of-run trajectory stack if estimated RAM > this
}
```

### Precision

Set via `"precision"` in the config. Affects particle positions, momenta, density arrays, and the initial potential field.

| Value | x64 enabled | Use case |
|-------|-------------|----------|
| `"float64"` | yes | Default; required for accurate gravitational potentials |
| `"float32"` | no | Faster on GPU/MPS; ~2× memory reduction; acceptable for large N |
| `"float16"` | no | Experimental; numerically unstable — emits a warning at startup |

`jax_enable_x64` is enabled automatically only when `precision = "float64"`. `md_cic_nd` infers its output dtype from the input position array dtype.

### Known Limitations

- **TSC mass assignment (Phase 5)**: Not implemented. CIC is used throughout.
- **Zeldovich momentum for LCDM**: `P = a_start × u` is exact for EdS but omits the `f(a)·H(a)·D(a)` prefactor for LCDM. Small error in initial velocities only; positions at `a_start` are unaffected.
- **PP force-split accuracy**: The erfc splitting kernel is applied but the PM force contribution at short separations is not analytically subtracted. Equivalent to assuming PM force is negligible below `r_cut` — holds when `r_cut ≪ Δx_force`.
- **Large-N PP scaling**: The Morton sliding window is O(Nₚ·W). Spatial hashing would give O(Nₚ) average-case lookup but requires variable-length neighbour lists incompatible with XLA's static-shape model. Fixed-width padding would be needed, partially negating the advantage.
- **`growing_mode` not used by solver**: `Cosmology.growing_mode(a)` is a diagnostic only (uses scipy quadrature). The simulation never calls it.

### Adding a New Config

1. Copy an existing JSON from `configs/`.
2. Set `solver`, `timestepping`, and `dim` as needed.
3. If `solver="p3m"`, ensure `pp_cutoff × (L / (2N)) < L / 2` (validated by `load_config`).
4. If `timestepping="adaptive"`, omit `dt` and add `C_cfl`, `dt_min`, `dt_max`, `n_chunks`.
5. Run `python main.py --config configs/your_new_config.json`.
