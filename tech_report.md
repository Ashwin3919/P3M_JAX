# P3M-JAX: A Dimension-Agnostic Particle-Particle-Particle-Mesh Cosmological N-Body Simulation Framework

**Ashwin Shirke**

---

## Abstract

We present P3M-JAX, a cosmological N-body simulation framework implementing the Particle-Particle-Particle-Mesh (P3M) algorithm in the JAX numerical computing library. The code solves the collisionless Boltzmann equation coupled to Poisson's equation in an expanding Friedmann-Lemaître-Robertson-Walker (FLRW) universe, using comoving coordinates and the Kick-Drift-Kick (KDK) leapfrog integrator. The implementation is fully dimension-agnostic, supporting both two-dimensional and three-dimensional domains via a single `dim` parameter. The Zeldovich approximation generates self-consistent initial conditions from a specified primordial power spectrum. Force computation supports two modes selectable at runtime: a pure Particle-Mesh (PM) mode using CIC deposition, FFT-based Poisson solution, second-order finite-difference gradient, and bilinear/trilinear interpolation; and a full P3M mode that augments the PM long-range force with a short-range direct particle-particle correction using a Morton Z-curve sorted sliding window and an error-function force-splitting kernel. Time integration supports both fixed $\Delta a$ stepping via `jax.lax.scan` and a CFL-adaptive scheme via `jax.lax.while_loop`. JAX's XLA compilation delivers hardware-accelerated performance on CPU, GPU, and Apple Silicon without platform-specific code. Outputs include particle and density snapshots in Legacy VTK format and the matter power spectrum $P(k)$, corrected for CIC window function aliasing and Poisson shot noise.

---

## 1. Introduction

Cosmological N-body simulations are the primary theoretical tool for studying the formation of large-scale structure in the universe. Beginning with the work of Hockney & Eastwood (1988) and Efstathiou et al. (1985), successive generations of codes — from GADGET (Springel 2005) to AREPO (Springel 2010) and beyond — have grown in complexity to incorporate hydrodynamics, star formation, feedback, and multi-scale force decomposition. For pedagogical, methodological, and rapid-prototyping purposes, however, a clean, minimal, and hardware-portable implementation remains valuable.

P3M-JAX addresses this need. It is written entirely in Python and JAX, requires no compiled extensions, and runs without modification on CPUs, NVIDIA GPUs, and Apple Silicon accelerators. The design emphasises three properties: correctness of the physics, transparency of the numerics, and performance through compilation rather than low-level optimisation.

The code solves the self-gravitating equations of motion in comoving coordinates for $N_p = N^d$ equal-mass particles in a periodic domain of side $L$, where $d \in \{2, 3\}$ is the spatial dimension. This document describes the physical model, numerical algorithms, design architecture, JAX-specific implementation choices, validation tests, and the full space of configurations supported by the code.

---

## 2. Physical Model

### 2.1 Comoving Equations of Motion

Let $\mathbf{x}$ denote comoving particle positions in units of Mpc/h and $\mathbf{p}$ the comoving momenta conjugate to $\mathbf{x}$. In an expanding background described by scale factor $a(t)$ and Hubble parameter $H(a) \equiv \dot{a}/a$, the equations of motion take the form

$$\frac{d\mathbf{x}}{da} = \frac{\mathbf{p}}{a^2 H(a)}, \tag{1}$$

$$\frac{d\mathbf{p}}{da} = -\frac{\nabla_x \phi}{H(a)}, \tag{2}$$

where $\phi(\mathbf{x}, a)$ is the comoving gravitational potential satisfying

$$\nabla^2 \phi = \frac{3}{2} \Omega_M H_0^2 \, \delta(\mathbf{x}, a) / a, \tag{3}$$

and $\delta \equiv \rho / \bar{\rho} - 1$ is the density contrast. The factor $G_\text{eff} \equiv \frac{3}{2} \Omega_M H_0^2$ is precomputed as `cosmology.G`. Scale factor $a$ serves as the time variable; the system is advanced in increments $\Delta a$.

### 2.2 Hubble Parameter

For a spatially flat or curved FLRW universe with matter density $\Omega_M$, cosmological constant $\Omega_\Lambda$, and curvature $\Omega_K = 1 - \Omega_M - \Omega_\Lambda$,

$$H(a) = H_0 \, a \sqrt{\Omega_\Lambda + \Omega_M a^{-3} + \Omega_K a^{-2}}. \tag{4}$$

Two cosmological presets are provided: `EDS_PRESET` (Einstein-de Sitter: $\Omega_M = 1$, $\Omega_\Lambda = 0$) and `LCDM_PRESET` ($H_0 = 68$, $\Omega_M = 0.31$, $\Omega_\Lambda = 0.69$). Any combination can be specified via the configuration file.

### 2.3 Linear Growth Factor

The linear growth factor $D(a)$ is

$$D(a) = \frac{5}{2} \Omega_M H(a) / a \int_0^a \frac{da'}{[a' H(a')]^3}, \tag{5}$$

evaluated numerically via `scipy.integrate.quad` and available as `cosmology.growing_mode(a)`.

---

## 3. Initial Conditions

### 3.1 Primordial Power Spectrum

The initial density field is a Gaussian random field with power spectrum $P_\text{prim}(k) \propto k^{n_s}$, tapered by a discrete Gaussian scale filter

$$W_\text{Gauss}(k) = \exp\!\left(\sigma^2 / \Delta x^2 \cdot (\cos(k \Delta x) - 1)\right) \tag{6}$$

and a hard Nyquist cutoff at $|\mathbf{k}| > k_\text{Nyq} = N\pi/L$.

### 3.2 Gaussian Random Field Generation

`garfield(B, P, T_filt, seed)` generates the initial potential $\phi$ by: (i) drawing white noise $\xi(\mathbf{x})$; (ii) applying spectral amplitude $\hat{\phi}_0 = \hat{\xi} \cdot \sqrt{P(k)}$; (iii) applying the transfer filter; (iv) returning $\phi = \text{Re}[\text{IFFT}(\hat{\phi})]$. All FFT operations use `jnp.fft.fftn` / `ifftn`, which handle arbitrary dimension.

### 3.3 Zeldovich Approximation

Particles are placed on a regular Lagrangian grid $\mathbf{q}_i$ and displaced:

$$\mathbf{x}(a) = \mathbf{q} + a \, \mathbf{u}(\mathbf{q}), \qquad \mathbf{p}(a) = a \, \mathbf{u}(\mathbf{q}), \tag{7}$$

where the displacement field is

$$u_i(\mathbf{q}) = -\partial_i \phi / \Delta x, \quad i = 0, \ldots, d-1. \tag{8}$$

`a2r(B, X)` reshapes $(d, N, \ldots)$ arrays to $(N^d, d)$ particle coordinates; `r2a` is the inverse. Particle mass is $m_p = (N_f / N_m)^d$ to account for the dual-resolution grid.

---

## 4. Force Computation

### 4.1 Dual-Resolution Grid

Two `Box` instances are maintained: $B_m$ (resolution $N^d$, cell size $\Delta x = L/N$) and $B_f$ (resolution $(2N)^d$, cell size $\Delta x/2$). The force grid at $2\times$ the linear resolution reduces large-scale aliasing in the gravitational potential. `PoissonVlasov` uses $B_f$ as its primary box.

### 4.2 Cloud-in-Cell Mass Deposition

CIC distributes each particle's mass over $2^d$ neighbouring cells. For fractional grid position $\mathbf{f}$, the weight assigned to corner $\boldsymbol{\delta} \in \{0,1\}^d$ is

$$W(\boldsymbol{\delta}) = \prod_{i=0}^{d-1} \begin{cases} 1 - f_i & \delta_i = 0 \\ f_i & \delta_i = 1 \end{cases}. \tag{9}$$

`md_cic_nd(shape, pos)` unrolls the $2^d$ corner loop at compile time via `itertools.product([0,1], repeat=d)`, producing a fixed XLA graph. `shape` is a static `jit` argument.

### 4.3 Fourier-Space Poisson Solver

$$\hat{\phi}(\mathbf{k}) = -\frac{G_\text{eff}}{a} \cdot \frac{\hat{\delta}(\mathbf{k})}{|\mathbf{k}|^2}. \tag{10}$$

The kernel $-1/|\mathbf{k}|^2$ is precomputed once as `Potential()(B_f.K)` and stored in `self.kernel`. The zero mode is set to zero, enforcing $\langle\phi\rangle = 0$.

### 4.4 Gradient and Force Interpolation

The acceleration $\mathbf{g} = -\nabla\phi$ uses a fourth-order central finite-difference stencil with periodic boundary conditions:

$$(\partial_i \phi)_\mathbf{n} = \frac{1}{12\Delta x}\left[\phi_{\mathbf{n}+2\hat{e}_i} - 8\phi_{\mathbf{n}+\hat{e}_i} + 8\phi_{\mathbf{n}-\hat{e}_i} - \phi_{\mathbf{n}-2\hat{e}_i}\right]. \tag{11}$$

The gradient field is interpolated to particle positions using `InterpND`, which applies the same $2^d$ corner weights as deposition — ensuring the force interpolation is the exact transpose of the mass deposition operator.

---

## 5. Fourier Filter Algebra

All Fourier-space operations are expressed through a composable `Filter` class supporting operator overloading ($\cdot$, $+$, $\hat{}$, $\sim$). Concrete subclasses:

- `Power_law(n)`: $P(k) = k^n$
- `Scale(B, σ)`: discrete Gaussian smoothing
- `Cutoff(B)`: hard Nyquist truncation
- `Potential()`: $-1/|\mathbf{k}|^2$, Green's function of the Laplacian

All operate on the full $d$-dimensional $\mathbf{K}$ array via `(K**2).sum(axis=0)` and are automatically dimension-agnostic.

---

## 6. P3M Short-Range Force Correction

### 6.1 Force Decomposition

The standard P3M split decomposes the total gravitational force into long-range and short-range parts using a complementary error function:

$$F_\text{PM}(r) = \frac{G_\text{eff} \, m_j}{r^2} \cdot \text{erf}\!\left(\frac{r}{\alpha}\right), \tag{12a}$$

$$F_\text{PP}(r) = \frac{G_\text{eff} \, m_j}{r^2} \cdot \text{erfc}\!\left(\frac{r}{\alpha}\right), \tag{12b}$$

where $\alpha$ is the splitting scale. The PM mesh handles the erf part automatically through its smooth kernel. The PP correction adds only the erfc part, which is significant at $r \ll \alpha$ and negligible at $r \gg \alpha$.

In the implementation, the splitting scale is set as $\alpha = r_\text{cut} / 2.6$, where $r_\text{cut} = \ell_\text{cut} \cdot \Delta x$ and $\ell_\text{cut}$ is the `pp_cutoff` parameter in cell units. This choice makes $\text{erfc}(r_\text{cut}/\alpha) = \text{erfc}(2.6) \approx 2.4 \times 10^{-4}$, so the PP correction is less than 0.03% of the direct force at the cutoff boundary — a clean transition with negligible force error. The total force on particle $i$ from particle $j$ in the PP window is

$$\mathbf{F}_{ij}^\text{PP} = \frac{G_\text{eff}}{a} \cdot \text{erfc}\!\left(\frac{r_{ij}}{\alpha}\right) \cdot \frac{\mathbf{r}_{ij}}{|\mathbf{r}_{ij,\text{soft}}|^3}, \tag{13}$$

where $|\mathbf{r}_{ij,\text{soft}}|^2 = |\mathbf{r}_{ij}|^2 + \varepsilon^2$ and $\varepsilon$ is the gravitational softening length. The factor $G_\text{eff}/a$ matches the dimensional convention used in the PM potential (equation 10), ensuring consistent force units.

### 6.2 Morton Z-Curve Spatial Sorting

Direct pair summation over all $N_p^2$ particle pairs is computationally prohibitive. The PP correction is instead restricted to pairs within a fixed physical cutoff $r_\text{cut}$. To find such neighbours efficiently without a tree structure, particles are sorted along a Morton (Z-curve) space-filling curve, which clusters spatially proximate particles into nearby positions in the sorted array.

The Morton code for a particle at grid position $(c_0, c_1, \ldots, c_{d-1})$ is formed by interleaving the bits of the integer coordinates:

$$\text{code}_i = \bigoplus_{b=0}^{15} \bigoplus_{d'=0}^{d-1} \left[\left(\frac{c_{d'} \gg b}{1}\right) \& 1\right] \ll (b \cdot d + d'), \tag{14}$$

where $\gg$ and $\ll$ denote bit shifts and $\&$ is bitwise AND. The double loop over `bit` (0–15) and dimension index `d'` is a Python-level loop that executes during JAX tracing, producing a fixed sequence of 16$d$ bitwise XLA operations — zero runtime overhead per call after the first JIT compilation. The encoding supports grids up to $65536^d$ ($2^{16}$ points per dimension).

```python
def _morton_encode(self, x_grid):
    coords = jnp.floor(x_grid).astype(jnp.int32).clip(0, self.box.N - 1)
    code = jnp.zeros(coords.shape[0], dtype=jnp.int32)
    for bit in range(16):
        for d in range(self.box.dim):
            code = code | (((coords[:, d] >> bit) & 1) << (bit * self.box.dim + d))
    return code
```

After sorting by Morton code via `jnp.argsort`, spatially adjacent particles are (approximately) adjacent in the sorted array. A fixed window of $\pm W$ neighbours in sorted order therefore captures the nearest spatial neighbours for typical clustered particle distributions.

### 6.3 Sliding-Window vmap Force Computation

The PP force is computed using `jax.vmap` over all particle indices, each evaluating a sliding window of $2W + 1$ neighbours in the Morton-sorted array:

```python
def force_on_i(i):
    js_raw = i + jnp.arange(-W, W + 1)
    js     = jnp.clip(js_raw, 0, N_p - 1)
    neigh_pos = sorted_pos[js]
    r_vec  = sorted_pos[i] - neigh_pos
    r_vec  = r_vec - L * jnp.round(r_vec / L)   # minimum image
    r_bare = jnp.sqrt(jnp.sum(r_vec**2, axis=-1))
    r_soft3 = (r_bare**2 + eps**2)**1.5
    erfc_w  = jax.scipy.special.erfc(r_bare / alpha)
    valid   = (js_raw >= 0) & (js_raw < N_p) & (js_raw != i) & (r_bare < r_cut)
    return G_eff * jnp.sum(valid[:, None] * erfc_w[:, None] * r_vec / r_soft3[:, None], axis=0)

sorted_acc = jax.vmap(force_on_i)(jnp.arange(N_p))
```

The window size $W$ is a Python integer (compile-time constant), so the window arrays have static shape `(2W+1,)` and the entire vmap body is a fixed-shape XLA computation. The minimum-image convention (equation 15 in Hockney & Eastwood 1988) enforces periodic boundary conditions within the window.

After computing accelerations in sorted order, results are mapped back to the original particle order via `inv_order = jnp.argsort(order)`.

### 6.4 Total Force

The momentum equation with P3M is:

$$\frac{d\mathbf{p}}{da} = -\frac{\nabla\phi_\text{PM} + \mathbf{f}_\text{PP}}{H(a)}, \tag{15}$$

where $\mathbf{f}_\text{PP}$ is the summed PP correction from equation (13). The `if self.solver == "p3m"` branch in `momentumEquation` is resolved at Python level during JAX tracing — it produces no runtime conditional and carries zero overhead in the PM path.

---

## 7. Time Integration

### 7.1 Kick-Drift-Kick Leapfrog

A single leapfrog step from $a_n$ to $a_n + \Delta a$:

$$\mathbf{p}_{n+1/2} = \mathbf{p}_n + \frac{\Delta a}{2} \left.\frac{d\mathbf{p}}{da}\right|_n, \tag{16a}$$

$$\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta a \left.\frac{d\mathbf{x}}{da}\right|_{n+1/2}, \tag{16b}$$

$$\mathbf{p}_{n+1} = \mathbf{p}_{n+1/2} + \frac{\Delta a}{2} \left.\frac{d\mathbf{p}}{da}\right|_{n+1}. \tag{16c}$$

The drift step uses only the half-kicked momentum, requiring two force evaluations per full step. `State(time, position, momentum)` is a `NamedTuple`; all operations return new instances with no in-place mutation.

### 7.2 Fixed Stepping with lax.scan

For fixed $\Delta a$, the time loop uses `jax.lax.scan`:

```python
final_state, all_states = jax.lax.scan(step_fn, init_state, xs=None, length=n_steps)
```

`lax.scan` represents the entire loop as a single XLA program, eliminating Python overhead between steps. The `save_every` parameter groups steps into chunks so that one snapshot is emitted per chunk, reducing stored trajectory size.

### 7.3 Adaptive Stepping with lax.while_loop

For the adaptive integrator, the number of steps is not known in advance. The simulation is divided into user-specified checkpoints; between consecutive checkpoints at $a_n$ and $a_{n+1}$, `lax.while_loop` advances the state with variable $\Delta a$:

```python
def cond_fn(carry): return carry[0].time < a_target
def body_fn(carry):
    s, _ = carry
    dt = min(compute_dt(s), a_target - s.time)   # don't overshoot
    return leap_frog(dt, system, s), dt
final_state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, dt_min))
```

`lax.while_loop` is JIT-compatible: JAX traces `cond_fn` and `body_fn` with abstract values, compiles each once, and runs them until the condition is false. Unlike `lax.scan`, `while_loop` does not collect intermediate states; only the final state is returned. Snapshots are therefore obtained at the Python level by running one `while_loop` call per checkpoint interval, which is the pattern used in `main.py`.

### 7.4 CFL Time-Step Estimate

The adaptive step size is computed from the maximum particle drift rate:

$$\Delta a_\text{CFL} = C_\text{CFL} \cdot \frac{\varepsilon}{v_\text{max}}, \quad v_i = \frac{|\mathbf{p}_i|}{a^2 H(a)}, \tag{17}$$

where $\varepsilon$ is the gravitational softening length (which sets the resolution scale below which smaller steps are unnecessary) and $C_\text{CFL} \approx 0.3$ is a safety factor. The result is clamped to $[\Delta a_\text{min}, \Delta a_\text{max}]$ to prevent infinite loops at early times (low velocities) or excessive runtime at late times (high velocities). The step shrinks as clustering develops and particle velocities increase — automatically concentrating computational effort where the dynamics are most rapid.

### 7.5 Incremental I/O

Both fixed and adaptive paths in `main.py` operate in a Python chunk loop, writing VTK snapshots and appending power spectrum rows to CSV after each chunk. Peak memory consumption is bounded by a single snapshot, not the full trajectory.

---

## 8. JAX Performance Architecture

### 8.1 XLA Compilation via jax.jit

All computationally intensive functions are JIT-compiled. On first call, JAX traces the function with abstract values, lowers to HLO, and compiles a hardware-optimised binary. Subsequent calls reuse the binary. For the force pipeline, this means CIC deposition, FFT, convolution, IFFT, gradient, and interpolation execute as a single fused kernel with no Python overhead.

### 8.2 Static Shape Requirement and CIC Unrolling

XLA requires all shapes to be known at compile time. `md_cic_nd(shape, pos)` declares `shape` as a static argument. The `itertools.product([0,1], repeat=d)` corner loop runs at trace time, so the compiler sees exactly $2^d$ fixed scatter-add operations regardless of particle count.

### 8.3 Morton Encoding at Trace Time

The `_morton_encode` bit-interleaving loops (`for bit in range(16): for d in range(dim)`) are Python loops that execute during tracing, not at runtime. XLA receives a fixed sequence of $16d$ bitwise operations (32 for 2D, 48 for 3D) and can optimise them as a single computational block.

### 8.4 vmap over PP Window

`jax.vmap(force_on_i)(jnp.arange(N_p))` vectorises the per-particle PP force computation over the batch dimension. Since `W` is a Python integer, all arrays inside `force_on_i` have static shapes `(2W+1,)` and `(2W+1, d)`. XLA can fuse the entire vmap into a batched matrix operation.

### 8.5 Precision and x64 Mode

When `precision = "float64"`, `jax.config.update("jax_enable_x64", True)` is called before any JAX operations. For `float32`, the flag is unset and arrays are 32-bit, halving memory bandwidth and roughly doubling GPU throughput. Float16 is flagged with a runtime warning.

### 8.6 Apple Silicon (MPS) Acceleration

JAX routes computation to the Metal Performance Shaders backend automatically on Apple M-series hardware. The `float32` precision mode is recommended for MPS.

---

## 9. Power Spectrum Analysis

### 9.1 Estimator

$$\hat{P}(k) = \frac{V}{N^{2d}} |\hat{\delta}(\mathbf{k})|^2, \quad \delta = \rho - 1, \tag{18}$$

modes averaged in annular bins of width $\Delta(\log k)$.

### 9.2 CIC Window Correction

$$P_\text{corr}(k) = \hat{P}(k) \left/ \prod_{i=0}^{d-1} \text{sinc}^2\!\left(\frac{k_i \Delta x}{2\pi}\right). \right. \tag{19}$$

### 9.3 Shot Noise Subtraction

$$P_\text{true}(k) = P_\text{corr}(k) - 1/\bar{n}, \quad \bar{n} = N_p / V. \tag{20}$$

Results are appended to `results/{config_name}/power_spectrum.csv` with columns `[step, a, k, Pk]`.

---

## 10. Output and Visualisation

### 10.1 VTK Particle Snapshots

Particle positions and momenta are saved as Legacy Binary VTK PolyData files. Two-dimensional positions are padded with $z = 0$. All data are written in big-endian 32-bit format. The `VERTICES` connectivity array is built as a native `int32` array and converted to big-endian in one operation to avoid mixed-endianness corruption. Momenta are stored as a point-data `VECTORS` field named `momentum`.

### 10.2 VTK Density Fields

The CIC density field is written as an ASCII `DATASET STRUCTURED_POINTS` file. For two-dimensional grids, $n_z = 1$ with spacing equal to the 2D cell size in all three dimensions, preventing degenerate zero-thickness cells that produce grid-box rendering artefacts in ParaView. Data are written in Fortran (x-fastest) order to match the VTK `STRUCTURED_POINTS` convention.

### 10.3 Snapshot Frequency

The `vtk_freq` config key controls how often VTK files are written. Setting `vtk_freq: 5` writes one snapshot every five chunks, reducing I/O overhead for long runs while preserving the power spectrum CSV at every chunk.

---

## 11. Code Architecture

### 11.1 Module Structure

1. `src/core/` — dimension-agnostic primitives: `Box`, `md_cic_nd`, `InterpND`, `gradient_2nd_order`, `garfield`, Fourier filters.
2. `src/physics/` — cosmological physics: `Cosmology`, `PoissonVlasov` (PM + P3M), `Zeldovich`.
3. `src/solver/` — integrator infrastructure: `State`, `HamiltonianSystem` ABC, `leap_frog`, `step_chunk` (fixed), `step_chunk_adaptive` (CFL adaptive), `compute_dt`.
4. `src/utils/` — I/O, analysis, plotting.

### 11.2 Solver Architecture

`PoissonVlasov.__init__` accepts a `solver` parameter (`"pm"` or `"p3m"`). Force computation is split into private helpers:

- `_pm_force(x_grid, phi, da)` — gradient + interpolation, always called.
- `_morton_encode(x_grid)` — bit-interleaved spatial index.
- `_pp_force(pos, a, da)` — Morton sort, sliding-window erfc PP correction, only called when `solver="p3m"`.

The `if self.solver == "p3m"` branch in `momentumEquation` is a Python-level branch resolved at trace time, producing a distinct compiled graph for each solver — zero runtime overhead for the PM path.

### 11.3 Dimension Agnosticism

All core routines are parameterised by `Box.dim`:

- `md_cic_nd`: corners unrolled as `product([0,1], repeat=dim)`.
- `InterpND`: same corner structure for field reading.
- `Zeldovich.u`: displacement computed for `range(dim)` axes.
- `_morton_encode`: bit-interleaving loop over `range(self.box.dim)`.
- `PoissonVlasov.momentumEquation`: force stacked with `jnp.stack(..., axis=-1)` over `range(dim)`.
- `particle_mass`: scaled as $(N_f / N_m)^d$.

---

## 12. Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 2 | Spatial dimension (2 or 3) |
| `N` | int | — | Particles per side; $N^d$ total |
| `L` | float | — | Box side length [Mpc/h] |
| `A` | float | — | Initial displacement amplitude |
| `seed` | int | — | Random seed |
| `a_start` | float | — | Initial scale factor |
| `a_end` | float | — | Final scale factor |
| `power_index` | float | — | Primordial spectral index $n_s$ |
| `H0` | float | 70.0 | Hubble constant [km/s/Mpc] |
| `OmegaM` | float | 1.0 | Matter density $\Omega_M$ |
| `OmegaL` | float | 0.0 | Cosmological constant $\Omega_\Lambda$ |
| `precision` | string | `"float64"` | `"float16"` \| `"float32"` \| `"float64"` |
| `solver` | string | `"pm"` | `"pm"` \| `"p3m"` |
| `pp_window` | int | 2 | P3M: half-window width $W$ (window = $2W+1$ particles) |
| `pp_softening` | float | 0.1 | P3M: gravitational softening $\varepsilon$ [Mpc/h] |
| `pp_cutoff` | float | 2.5 | P3M: cutoff in force-cell units ($r_\text{cut} = \ell_\text{cut} \cdot \Delta x$) |
| `timestepping` | string | `"fixed"` | `"fixed"` \| `"adaptive"` |
| `dt` | float | — | Fixed $\Delta a$ step (used when `timestepping="fixed"`) |
| `save_every` | int | 1 | Fixed stepping: leapfrog steps per snapshot |
| `C_cfl` | float | 0.3 | Adaptive: CFL safety factor |
| `dt_min` | float | 0.001 | Adaptive: minimum $\Delta a$ |
| `dt_max` | float | 0.05 | Adaptive: maximum $\Delta a$ |
| `n_chunks` | int | 50 | Adaptive: number of output checkpoints |
| `save_vtk` | bool | true | Write VTK snapshots |
| `vtk_freq` | int | 1 | Write VTK every $N$ chunks |
| `save_power_spectrum` | bool | true | Compute and write $P(k)$ |

---

## 13. Available Configurations

**`default.json`**: 2D EdS, $N = 128$, $L = 50$ Mpc/h, float64, PM, $\Delta a = 0.02$. $128^2 = 16{,}384$ particles from $a = 0.02$ to $a = 1.0$. Recommended starting point.

**`high_res.json`**: 2D LCDM, $N = 256$, float32, $\Delta a = 0.015$. $65{,}536$ particles. Power spectrum convergence tests.

**`3d_default.json`**: 3D EdS, $N = 64$, float32. $64^3 = 262{,}144$ particles. Primary 3D validation config.

**`3d_heigh_res.json`**: 3D LCDM, $N = 128$, float32, $a_\text{end} = 1.32$, `save_every = 10`. $\approx 2.1 \times 10^6$ particles. Production run on GPU or Apple Silicon.

**`p3m_adaptive.json`**: 2D EdS, $N = 64$, P3M ($W = 4$, $\varepsilon = 0.2$, $\ell_\text{cut} = 2.5$), adaptive stepping ($C_\text{CFL} = 0.3$, $\Delta a \in [0.001, 0.05]$, 50 checkpoints). Demonstration config for the full P3M + adaptive pipeline.

---

## 14. What Can Be Studied with This Code

1. **Linear perturbation theory validation**: At early times, $P(k)$ should grow as $D^2(a) \cdot P_\text{prim}(k)$.
2. **Zeldovich pancake formation**: EdS with $n_s = -0.5$ forms sheet structures visible in VTK snapshots.
3. **Power spectrum evolution**: Non-linear growth of $P(k)$ from $a = 0.02$ to $a = 1$ can be compared to Peacock-Dodds (1996) or Halofit (Smith et al. 2003).
4. **Cosmology dependence**: EdS ($D \propto a$) vs LCDM growth suppression visible in time-tagged power spectrum CSV.
5. **PM vs P3M force accuracy**: Comparing PM and P3M runs at the same resolution quantifies the sub-cell force improvement from the PP correction, particularly near halo centres.
6. **Adaptive vs fixed stepping**: The adaptive integrator uses larger steps at early times and automatically refines near shell-crossing; comparing trajectories tests the CFL condition.
7. **Resolution convergence**: Running at $N = 64$, $128$, $256$ in 2D (or $32$, $64$, $128$ in 3D) tests force and particle resolution convergence.

---

## 15. Validation Tests

The test suite `tests/test_core.py` comprises 17 unit tests:

**Phases 1–3 (infrastructure)**
1. `test_md_cic_2d_mass_conservation` — 2D CIC mass conservation to $10^{-5}$.
2. `test_interp2d_identity` — interpolation at integer positions recovers exact cell values.
3. `test_box_wave_numbers` — fundamental and Nyquist frequencies correct.
4. `test_gradient_2nd_order` — FD gradient of $\sin(x)$ agrees with $\cos(x)$ to $10^{-2}$ on a 64-point grid.
5. `test_box_3d_shapes` — `Box(3, 64, 50.0)` yields `K.shape == (3, 64, 64, 64)`.
6. `test_potential_filter_3d` — `Potential()(box.K)` returns shape $(N, N, N)$.
7. `test_garfield_3d` — 3D Gaussian random field returns correct shape.
8. `test_md_cic_nd_mass_conservation_3d` — 3D CIC mass conservation.
9. `test_md_cic_nd_unit_deposit_3d` — particle at integer position deposits entirely into one cell.
10. `test_interpnd_3d_grid_points` — 3D interpolation at integer positions exact.
11. `test_zeldovich_3d_shape` — 3D Zeldovich state shape and mean density $\approx 1.0$.

**Phase 4 / 6 (PM and P3M forces)**
12. `test_pm_uniform_force_is_zero` — uniform particle grid $\Rightarrow$ PM acceleration $= 0$ everywhere.
13. `test_p3m_close_pair_force_larger_than_pm` — P3M force magnitude exceeds PM for a close pair, confirming PP correction adds at short range.
14. `test_pp_force_zero_beyond_cutoff` — PP correction vanishes for particles separated by $r > r_\text{cut}$, confirming the cutoff mask and erfc suppression.
15. `test_pp_erfc_less_than_direct` — erfc weight $< 1$ at short range, confirming the splitting kernel is applied (not raw Newtonian).

**Phase 7 (adaptive stepping)**
16. `test_compute_dt_decreases_with_velocity` — higher particle velocities yield smaller $\Delta a_\text{CFL}$.
17. `test_compute_dt_respects_bounds` — `compute_dt` output is always in $[\Delta a_\text{min}, \Delta a_\text{max}]$ for all velocity scales.

Run the full suite:

```bash
pytest tests/
```

---

## 16. Limitations and Future Work

**TSC mass assignment (Phase 5)**: The Triangular-Shaped Cloud scheme distributes mass over $3^d$ cells using a parabolic kernel, reducing aliasing relative to CIC. The corresponding deconvolved Green's function that corrects the Fourier-space potential for the TSC window function is not yet implemented.

**Zeldovich momentum for LCDM**: The initial momentum is set as $\mathbf{p} = a_\text{init} \cdot \mathbf{u}$, which is exact for Einstein-de Sitter ($D(a) = a$, growth rate $f = 1$) but omits the $f(a) \cdot H(a) \cdot D(a)$ prefactor for LCDM cosmologies. This introduces a small error in initial particle velocities for LCDM configurations (`high_res.json`, `3d_heigh_res.json`); positions are unaffected.

**PP force-split accuracy**: The current PP implementation uses the erfc splitting kernel with the standard $\alpha = r_\text{cut}/2.6$ prescription, but does not subtract an analytic model of the PM force at short range. In exact P3M, the PM contribution at separation $r$ is subtracted from the direct force to avoid double-counting at intermediate separations. The present implementation is equivalent to a force split at which the PM force is negligible below $r_\text{cut}$ — a valid regime when $r_\text{cut} \ll \Delta x_\text{force}$, which is the typical operating point for P3M.

---

## References

Efstathiou, G., Davis, M., White, S. D. M., & Frenk, C. S. 1985, ApJS, 57, 241

Hockney, R. W. & Eastwood, J. W. 1988, *Computer Simulation Using Particles* (Bristol: IOP)

Peacock, J. A. & Dodds, S. J. 1996, MNRAS, 280, L19

Smith, R. E., et al. 2003, MNRAS, 341, 1311

Springel, V. 2005, MNRAS, 364, 1105

Springel, V. 2010, MNRAS, 401, 791

Zel'dovich, Ya. B. 1970, A&A, 5, 84
