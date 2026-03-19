# P3M-JAX: A Dimension-Agnostic Particle-Particle-Particle-Mesh Cosmological N-Body Simulation Framework

**Ashwin Shirke**

---

## Abstract

We present P3M-JAX, a cosmological N-body simulation framework implementing the Particle-Particle-Particle-Mesh (P3M) algorithm in the JAX numerical computing library. The code solves the collisionless Boltzmann equation coupled to Poisson's equation in an expanding Friedmann-Lemaître-Robertson-Walker (FLRW) universe, using comoving coordinates and the Kick-Drift-Kick (KDK) leapfrog integrator. The implementation is fully dimension-agnostic, supporting two-dimensional and three-dimensional domains via a single `dim` parameter with no code duplication. The Zeldovich approximation generates self-consistent initial conditions from a user-specified primordial power spectrum. Force computation supports two runtime-selectable modes: a pure Particle-Mesh (PM) mode using Cloud-in-Cell (CIC) deposition, FFT-based Poisson solution, second-order finite-difference gradient, and bilinear/trilinear force interpolation; and a full P3M mode that augments the PM long-range force with a short-range direct particle-particle correction using a Morton Z-curve sorted sliding window and an error-function force-splitting kernel. Time integration supports both fixed $\Delta a$ stepping via `jax.lax.scan` and a CFL-adaptive scheme via `jax.lax.while_loop`, which automatically refines the timestep during structure formation. Robustness features include per-step NaN/Inf detection, comprehensive config value validation, and an OOM guard for large trajectory stacking. JAX's XLA compilation delivers hardware-accelerated performance on CPU, GPU, and Apple Silicon without platform-specific code. Outputs include particle and density snapshots in Legacy VTK format and the matter power spectrum $P(k)$, corrected for CIC window function aliasing and Poisson shot noise. The framework is validated by 54 automated tests spanning unit, integration, and physics correctness checks.

---

## 1. Introduction

Cosmological N-body simulations are the primary theoretical tool for studying the formation of large-scale structure in the universe. Beginning with the work of Hockney & Eastwood (1988) and Efstathiou et al. (1985), successive generations of codes — from GADGET (Springel 2005) to AREPO (Springel 2010) and beyond — have grown in complexity to incorporate hydrodynamics, star formation, feedback, and multi-scale force decomposition. For pedagogical, methodological, and rapid-prototyping purposes, however, a clean, minimal, and hardware-portable implementation remains valuable.

P3M-JAX addresses this need. It is written entirely in Python and JAX, requires no compiled extensions, and runs without modification on CPUs, NVIDIA GPUs, and Apple Silicon accelerators. The design emphasises three properties: correctness of the physics, transparency of the numerics, and performance through compilation rather than low-level optimisation.

The code solves the self-gravitating equations of motion in comoving coordinates for $N_p = N^d$ equal-mass particles in a periodic domain of side $L$, where $d \in \{2, 3\}$ is the spatial dimension. This document describes the physical model, numerical algorithms, design architecture, JAX-specific implementation choices, robustness infrastructure, validation tests, and the full configuration space supported by the code.

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

$$\frac{da}{dt} \equiv a H(a) = H_0 \, a \sqrt{\Omega_\Lambda + \Omega_M a^{-3} + \Omega_K a^{-2}}. \tag{4}$$

Implemented as `cosmology.da(a)`, this quantity appears in both the position equation (1) and the CFL time-step estimate (Section 7.4). Two presets are provided: `EDS_PRESET` (Einstein-de Sitter: $\Omega_M = 1$, $\Omega_\Lambda = 0$, $H_0 = 70$) and `LCDM_PRESET` ($H_0 = 68$, $\Omega_M = 0.31$, $\Omega_\Lambda = 0.69$). Any combination can be specified via the configuration file.

### 2.3 Linear Growth Factor

The linear growth factor $D(a)$ is computed numerically via the Heath (1977) integral formula:

$$D(a) = \frac{5}{2} \Omega_M \frac{H(a)}{a} \int_0^a \frac{da'}{[a' H(a')]^3}. \tag{5}$$

The integral is evaluated using `scipy.integrate.quad` starting from a lower limit of $\varepsilon_0 = 10^{-5}$ to avoid the $1/a^3$ singularity at $a \to 0$; a seed value of $10^{-5}$ is added to approximate $D(\varepsilon_0) \approx \varepsilon_0$ in the matter-dominated regime. This function is available as `cosmology.growing_mode(a)` and is used diagnostically; the simulation itself does not call it at runtime.

**Note:** The additive $10^{-5}$ offset introduces a small ($\lesssim 0.1\%$ for $a \gtrsim 2$, $\lesssim 13\%$ for $a \sim 0.2$ with $H_0 = 70$) fractional bias at small scale factors because $H_0^{-2} \cdot a$ can be comparable to $10^{-5}$ for typical $H_0$ values. This does not affect the simulation, which does not use the growth factor.

---

## 3. Initial Conditions

### 3.1 Primordial Power Spectrum

The initial density field is a Gaussian random field with power spectrum $P_\text{prim}(k) \propto k^{n_s}$, tapered by a discrete Gaussian scale filter

$$W_\text{Gauss}(k) = \exp\!\left(\sigma^2 / \Delta x^2 \cdot (\cos(k \Delta x) - 1)\right) \tag{6}$$

(implemented as `Scale(B, σ)`) and a hard Nyquist cutoff at $|\mathbf{k}| > k_\text{Nyq} = N\pi/L$ (implemented as `Cutoff(B)`). The spectral index $n_s$ is set by `power_index` in the configuration.

### 3.2 Gaussian Random Field Generation

`garfield(B, P, T_filt, seed)` generates the initial potential $\phi$ in a single FFT round-trip:

$$\phi = \text{Re}\!\left[\text{IFFT}\!\left(\hat{\xi}(\mathbf{k}) \cdot \sqrt{P(|\mathbf{k}|)} \cdot T(|\mathbf{k}|)\right)\right], \tag{7}$$

where $\hat{\xi}$ is the FFT of a white-noise field drawn from `jax.random.normal`, $P$ is the power spectrum filter, and $T$ is the transfer filter (typically `Potential()` = $-1/|\mathbf{k}|^2$). All FFT operations use `jnp.fft.fftn` / `ifftn`, which handle arbitrary dimension transparently. The amplitude is then scaled by the config parameter `A` before being passed to `Zeldovich`.

### 3.3 Zeldovich Approximation

Particles are placed on a regular Lagrangian grid $\mathbf{q}_i$ and displaced:

$$\mathbf{x}(a) = \mathbf{q} + a \, \mathbf{u}(\mathbf{q}), \qquad \mathbf{p}(a) = a \, \mathbf{u}(\mathbf{q}), \tag{8}$$

where the displacement field is

$$u_i(\mathbf{q}) = -\partial_i \phi / \Delta x, \quad i = 0, \ldots, d-1. \tag{9}$$

The gradient is computed using the same fourth-order finite-difference stencil as the force pipeline (Section 4.4). The module-private function `_grid_to_particles(B, X)` reshapes a $(d, N, \ldots)$ grid array to the $(N^d, d)$ particle coordinate array by `.reshape(dim, -1).T`. Particle mass is $m_p = (N_f / N_m)^d$, where $N_f$ and $N_m$ are the linear resolutions of the force and mass grids respectively.

**Array ordering convention.** `jnp.indices(shape)` returns a `(dim, N, ...)` array where axis 0 of the index array corresponds to axis 0 of the grid (row index). After `_grid_to_particles`, particle $k = i \cdot N + j$ has `position[k, 0] = i \cdot \Delta x` (first grid axis) and `position[k, 1] = j \cdot \Delta x` (second grid axis). Tests that reconstruct the Lagrangian lattice must use `jnp.meshgrid(..., indexing='ij')` to match this convention.

---

## 4. Force Computation

### 4.1 Dual-Resolution Grid

Two `Box` instances are maintained throughout the simulation:

- $B_m$ (mass box): resolution $N^d$, cell size $\Delta x = L/N$. Used for density output, power spectrum computation, and end-of-run diagnostics.
- $B_f$ (force box): resolution $(2N)^d$, cell size $\Delta x/2$. Used as the primary box in `PoissonVlasov`. The $2\times$ finer force grid reduces aliasing in the gravitational potential.

`PoissonVlasov` is constructed with `B_f`; the mass box `B_m` is only used in the `main.py` I/O loop.

### 4.2 Cloud-in-Cell Mass Deposition

CIC distributes each particle's mass over $2^d$ neighbouring cells. For a particle at fractional grid position $\mathbf{f} = \mathbf{x} / \Delta x - \lfloor \mathbf{x} / \Delta x \rfloor$, the weight assigned to the corner offset $\boldsymbol{\delta} \in \{0,1\}^d$ is

$$W(\boldsymbol{\delta}) = \prod_{i=0}^{d-1} \begin{cases} 1 - f_i & \delta_i = 0 \\ f_i & \delta_i = 1 \end{cases}. \tag{10}$$

`md_cic_nd(shape, pos)` unrolls the $2^d$ corner loop at trace time using `itertools.product([0,1], repeat=d)`, producing a fixed-size XLA scatter-add graph. Cell indices are wrapped modulo `shape` for periodic boundary conditions. `shape` is declared a static `jit` argument so the graph is reused across calls with the same grid resolution.

### 4.3 Fourier-Space Poisson Solver

Given the density contrast $\delta = \rho / \bar{\rho} - 1$ (where $\bar{\rho}$ is maintained exactly at 1 by the CIC normalization), the gravitational potential is

$$\hat{\phi}(\mathbf{k}) = -\frac{G_\text{eff}}{a} \cdot \frac{\hat{\delta}(\mathbf{k})}{|\mathbf{k}|^2}. \tag{11}$$

The kernel $-1/|\mathbf{k}|^2$ (with the zero mode set to zero, enforcing $\langle\phi\rangle = 0$) is precomputed once as `Potential()(B_f.K)` and stored in `self.kernel`. At each force evaluation, the pipeline executes: FFT($\delta$) → multiply by kernel → IFFT → multiply by $G_\text{eff}/a$.

### 4.4 Gradient and Force Interpolation

The gravitational acceleration $\mathbf{g} = -\nabla\phi$ is computed from the potential using a fourth-order central finite-difference stencil with periodic boundary conditions:

$$\left(\partial_i \phi\right)_\mathbf{n} = \frac{1}{12\Delta x}\left[\phi_{\mathbf{n}+2\hat{e}_i} - 8\phi_{\mathbf{n}+\hat{e}_i} + 8\phi_{\mathbf{n}-\hat{e}_i} - \phi_{\mathbf{n}-2\hat{e}_i}\right]. \tag{12}$$

Note: `gradient_2nd_order` returns the stencil numerator (not divided by $\Delta x$); the calling code in `_pm_force` divides by `box.res` to recover the correct physical gradient.

The gradient field is interpolated to particle positions using `InterpND`, which applies the same $2^d$ corner weights as `md_cic_nd` — this makes the force interpolation the exact transpose of the mass deposition operator, a property that helps conserve momentum.

---

## 5. Fourier Filter Algebra

All Fourier-space operations are expressed through a composable `Filter` class supporting operator overloading. The base class stores a callable that maps a $\mathbf{K}$ array to a filter response array. Concrete subclasses:

| Class | Formula | Purpose |
|-------|---------|---------|
| `Power_law(n)` | $P(k) = k^n$ | Primordial power spectrum |
| `Scale(B, σ)` | $\exp(\sigma^2/\Delta x^2 \cdot (\cos(k\Delta x)-1))$ | Discrete Gaussian smoothing |
| `Cutoff(B)` | $\mathbf{1}[k \leq k_\text{Nyq}]$ | Hard Nyquist truncation |
| `Potential()` | $-1/|\mathbf{k}|^2$ (zero mode = 0) | Poisson Green's function |

Operators: `__mul__` (pointwise product), `__add__` (sum), `__pow__` (power), `__invert__` (reciprocal), `__rmul__` (scalar left-multiply). All return `NotImplemented` for unsupported types, preserving Python's normal operator dispatch. Operations on scalars fall through to `__rmul__`, so `2.0 * filter` works correctly.

All filters operate on the full $d$-dimensional `K` array via `(K**2).sum(axis=0)` and are automatically dimension-agnostic.

---

## 6. P3M Short-Range Force Correction

### 6.1 Force Decomposition

The P3M algorithm splits the total gravitational force into a smooth long-range part handled by the PM mesh and a short-range correction handled by direct pair summation. The split uses complementary error functions:

$$F_\text{PM}(r) \propto \frac{1}{r^2} \cdot \text{erf}\!\left(\frac{r}{\alpha}\right), \tag{13a}$$

$$F_\text{PP}(r) \propto \frac{1}{r^2} \cdot \text{erfc}\!\left(\frac{r}{\alpha}\right). \tag{13b}$$

These satisfy $F_\text{PM} + F_\text{PP} = F_\text{direct}$ exactly, so the total force is exact when the PM solver is ideal. The splitting scale $\alpha = r_\text{cut}/2.6$, where $r_\text{cut} = \ell_\text{cut} \cdot \Delta x_f$, ensures $\text{erfc}(r_\text{cut}/\alpha) = \text{erfc}(2.6) \approx 2.4 \times 10^{-4}$ — the PP correction contributes less than 0.03% of the direct force at the cutoff boundary.

The complete PP force on particle $i$ from a neighbour $j$ within $r_\text{cut}$ is

$$\mathbf{F}_{ij}^\text{PP} = \frac{G_\text{eff}}{a} \cdot \text{erfc}\!\left(\frac{r_{ij}}{\alpha}\right) \cdot \frac{\mathbf{r}_{ij}}{|\mathbf{r}_{ij,\text{soft}}|^3}, \tag{14}$$

where $|\mathbf{r}_{ij,\text{soft}}|^2 = |\mathbf{r}_{ij}|^2 + \varepsilon^2$ introduces gravitational softening $\varepsilon$, and $\mathbf{r}_{ij} = \mathbf{x}_i - \mathbf{x}_j$ is the minimum-image separation vector (see Section 6.3).

**Sign convention.** `_pp_force` returns $\sum_j \mathbf{F}_{ij}^\text{PP}$ as defined in equation (14), with $\mathbf{r}_{ij}$ pointing from $j$ to $i$. `momentumEquation` negates this contribution: the final acceleration is $-(F_\text{PM} + F_\text{PP}) / H(a)$, making gravity attractive.

**Cutoff validation.** The constructor enforces $r_\text{cut} < L/2$; a violation raises `ValueError` immediately, before any JAX tracing occurs.

### 6.2 Morton Z-Curve Spatial Sorting

Direct pair summation over all $N_p^2$ particle pairs is $O(N_p^2)$ and prohibitive. The PP correction is restricted to pairs within $r_\text{cut}$ using a Morton-sorted sliding window, avoiding a tree structure.

The Morton code for a particle at grid position $(c_0, c_1, \ldots, c_{d-1})$ interleaves bits of the integer coordinates:

$$\text{code} = \bigoplus_{b=0}^{15} \bigoplus_{d'=0}^{d-1} \left[\left(\frac{c_{d'} \gg b}{1}\right) \& 1\right] \ll (b \cdot d + d'), \tag{15}$$

implemented as:

```python
def _morton_encode(self, x_grid):
    coords = jnp.floor(x_grid).astype(jnp.int32).clip(0, self.box.N - 1)
    code = jnp.zeros(coords.shape[0], dtype=jnp.int32)
    for bit in range(16):          # Python loop — executes at trace time
        for d in range(self.box.dim):
            code = code | (((coords[:, d] >> bit) & 1) << (bit * self.box.dim + d))
    return code
```

The double Python loop runs at JAX trace time, producing a fixed sequence of $16d$ bitwise XLA operations (32 for 2D, 48 for 3D) with zero runtime overhead per call after JIT compilation. The encoding supports grids up to $2^{16} = 65536$ points per dimension. After sorting by Morton code via `jnp.argsort`, spatially proximate particles are adjacent in the sorted array; a window of $\pm W$ neighbours then captures the nearest spatial neighbours for clustered distributions.

### 6.3 Sliding-Window vmap Force Computation

The PP force is computed using `jax.vmap` over all $N_p$ particle indices simultaneously:

```python
def force_on_i(i):
    js_raw   = i + jnp.arange(-W, W + 1)          # (2W+1,) window indices
    js       = jnp.clip(js_raw, 0, N_p - 1)        # safe gather (no out-of-bounds)
    neigh_pos = sorted_pos[js]                      # (2W+1, dim)
    r_vec    = sorted_pos[i] - neigh_pos            # points from j to i
    r_vec    = r_vec - L * jnp.round(r_vec / L)    # minimum-image convention
    r_bare   = jnp.sqrt(jnp.sum(r_vec**2, axis=-1))
    r_soft3  = (r_bare**2 + eps**2)**1.5
    erfc_w   = jax.scipy.special.erfc(r_bare / alpha)
    valid    = (js_raw >= 0) & (js_raw < N_p) & (js_raw != i) & (r_bare < r_cut)
    return G_eff * jnp.sum(valid[:, None] * erfc_w[:, None] * r_vec / r_soft3[:, None], axis=0)

sorted_acc = jax.vmap(force_on_i)(jnp.arange(N_p))
```

`W` is a Python integer (compile-time constant), so all arrays inside `force_on_i` have static shapes and XLA can fuse the vmap into a single batched kernel. The `valid` mask simultaneously handles: out-of-range indices at array boundaries, self-interaction ($j = i$), and physical cutoff ($r > r_\text{cut}$). Results are unsorted back to original particle order via `sorted_acc[inv_order]`.

### 6.4 Total Momentum Equation with P3M

$$\frac{d\mathbf{p}}{da} = -\frac{\nabla\phi_\text{PM} + \mathbf{f}_\text{PP}}{H(a)}, \tag{16}$$

The `if self.solver == "p3m"` branch in `momentumEquation` is resolved at Python level during JAX tracing — it compiles to a distinct XLA graph per solver, with zero runtime conditional overhead in either path.

---

## 7. Time Integration

### 7.1 Kick-Drift-Kick Leapfrog

A single step from $a_n$ to $a_n + \Delta a$ uses the KDK (Kick-Drift-Kick) scheme, a second-order symplectic integrator:

$$\mathbf{p}_{n+1/2} = \mathbf{p}_n + \frac{\Delta a}{2} \left.\frac{d\mathbf{p}}{da}\right|_n, \tag{17a}$$

$$\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta a \left.\frac{d\mathbf{x}}{da}\right|_{n+1/2}, \tag{17b}$$

$$\mathbf{p}_{n+1} = \mathbf{p}_{n+1/2} + \frac{\Delta a}{2} \left.\frac{d\mathbf{p}}{da}\right|_{n+1}. \tag{17c}$$

Each full step requires two force evaluations. `State(time, position, momentum)` is an immutable `NamedTuple`; all operations return new instances. This functional style is a prerequisite for `jax.lax.scan` and `jax.lax.while_loop`.

### 7.2 Fixed Stepping with lax.scan

For fixed $\Delta a$, the time loop compiles to a single XLA program via `jax.lax.scan`:

```python
final_state, snapshots = jax.lax.scan(step_fn, init_state, xs=None, length=n_steps)
```

`lax.scan` eliminates all Python overhead between steps. The `save_every` parameter wraps inner steps into a chunked scan, emitting one snapshot per chunk. `n_steps` must be divisible by `save_every`; a `ValueError` is raised at construction if not. The chunk function is JIT-compiled with `partial(step_chunk, system, dt=dt, save_every=k)` so only one compiled binary exists per `(dt, k)` pair.

### 7.3 Adaptive Stepping with lax.while_loop

When `timestepping = "adaptive"`, the simulation is divided into `n_chunks` checkpoint intervals. Within each interval $[a_n, a_{n+1}]$, `jax.lax.while_loop` advances the state with variable $\Delta a$:

```python
def cond_fn(carry): return carry[0].time < a_target
def body_fn(carry):
    s, _ = carry
    dt = jnp.minimum(compute_dt(s, ...), a_target - s.time)  # clamp last step
    return leap_frog(dt, system, s), dt
final_state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, dt_min))
```

`lax.while_loop` is JIT-compatible: both `cond_fn` and `body_fn` are traced once and compiled, then executed until the condition becomes false. Unlike `lax.scan`, `while_loop` does not collect intermediate states; only the final state is returned. Snapshots are obtained at the Python level by calling one `while_loop` per checkpoint, which is the pattern used in `main.py`.

The chunk function is JIT-compiled as `partial(step_chunk_adaptive, system, C_cfl=..., eps=..., dt_min=..., dt_max=...)` and called as `chunk_fn(state, a_target)`.

### 7.4 CFL Time-Step Estimate

The adaptive step size is computed from the maximum particle drift rate:

$$\Delta a_\text{CFL} = C_\text{CFL} \cdot \frac{\varepsilon}{v_\text{max}}, \quad v_i = \frac{|\mathbf{p}_i|}{a^2 H(a)}, \tag{18}$$

clamped to $[\Delta a_\text{min}, \Delta a_\text{max}]$. The softening length $\varepsilon$ (= `pp_softening` for P3M, = `pp_softening` default for PM) sets the resolution scale below which finer steps are unnecessary. The safety factor $C_\text{CFL} \approx 0.3$ is standard for N-body simulations. As clustering develops and particle velocities peak during structure formation, $\Delta a_\text{CFL}$ decreases automatically, concentrating computational effort where the dynamics are most rapid.

**Numerical guard.** The denominator uses `a² H(a) + 1e-20` to prevent division by zero. At the typical starting scale factor $a_\text{start} = 0.02$, the denominator is $\sim H_0 a^{5/2} \approx 0.004$, so the guard never activates in practice.

### 7.5 Numerical Stability and the CFL Condition

A common instability occurs with fixed timestepping when the displacement amplitude `A` is large. With $N = 144$, $L = 100$ Mpc/h, $\Delta x = 0.694$ Mpc/h, the CFL stability limit for the leapfrog is $v_\text{max} < \Delta x / \Delta a \approx 34.7$ Mpc/h. During non-linear collapse (typically $a \approx 0.3$–$0.6$ for $A = 10$), particle velocities can exceed this bound: the leapfrog overshoots, particles scatter, the power spectrum develops a transient dip, then the simulation recovers as velocities drop. This is a well-known N-body artefact.

**Resolution:** Use adaptive timestepping, which automatically shrinks $\Delta a$ when $v_\text{max}$ is large. The `3d_visual.json` config (which uses $A = 10$ and $N = 144$) is configured with adaptive stepping for this reason, using $\Delta a_\text{min} = 0.0005$, $\Delta a_\text{max} = 0.02$, $C_\text{CFL} = 0.3$.

### 7.6 Incremental I/O

Both fixed and adaptive paths in `main.py` operate in a Python chunk loop, writing VTK snapshots and appending power spectrum rows to CSV after each chunk. Peak memory consumption is bounded by a single snapshot, not the full trajectory.

---

## 8. Robustness and Validation Infrastructure

### 8.1 Config Validation

`load_config(path)` performs two levels of validation before any simulation code runs:

1. **Key presence** — `_REQUIRED_KEYS = {"N", "L", "A", "a_start", "a_end", "power_index", "seed"}` must all be present; `dt` required for fixed timestepping. Raises `KeyError` with a message listing missing keys.

2. **Value ranges** — `_validate_ranges(config, path)` checks:
   - `N ≥ 4` (integer), `L > 0`, `a_start > 0`, `a_end > a_start`, `H0 > 0`, `OmegaM ≥ 0`, `OmegaL ≥ 0`
   - Fixed: `dt > 0` and `dt < (a_end - a_start)`
   - Adaptive: `dt_min > 0`, `dt_max > dt_min`, `n_chunks ≥ 1`
   - P3M: `pp_window ≥ 1`, `pp_softening > 0`, `pp_cutoff > 0`
   - `dim ∈ {2, 3}`
   - Raises `ValueError` with a clear message on the first violation.

3. **Solver string** — must be `"pm"` or `"p3m"`; raises `ValueError` otherwise.

### 8.2 NaN/Inf Detection

After each chunk, the main loop checks:

```python
if not np.all(np.isfinite(pos_np)):
    raise RuntimeError(
        f"NaN/Inf detected in particle positions at chunk {chunk_idx + 1}, "
        f"a = {a_val:.4f}. Consider reducing dt or the amplitude A."
    )
```

This catches numerical blow-ups at the earliest possible point, before corrupt data is written to VTK files or the power spectrum CSV.

### 8.3 OOM Guard

Before stacking all snapshots into a trajectory array for end-of-run plots, the estimated memory is checked:

```python
_traj_bytes = n_chunks * N_p * dim * 2 * bytes_per_element  # positions + momenta
```

If this exceeds `oom_threshold_gb` (default 4 GB, configurable via the config file key `"oom_threshold_gb"`), trajectory stacking is skipped and per-chunk VTK files serve as the visualisation output instead.

---

## 9. JAX Performance Architecture

### 9.1 XLA Compilation via jax.jit

All computationally intensive functions are JIT-compiled. On first call, JAX traces the function with abstract values, lowers to HLO (High-Level Operations), and compiles a hardware-optimised binary. Subsequent calls reuse the binary with no Python overhead. For the force pipeline, CIC deposition, FFT, convolution, IFFT, gradient, and interpolation execute as a single fused kernel.

### 9.2 Static Shape Requirement and CIC Unrolling

XLA requires all array shapes to be known at compile time. `md_cic_nd(shape, pos)` declares `shape` as a static `jit` argument via `@partial(jax.jit, static_argnums=(0,))`. The `itertools.product([0,1], repeat=d)` corner loop runs at trace time, so the compiler sees exactly $2^d$ scatter-add operations — 4 for 2D, 8 for 3D.

### 9.3 Morton Encoding at Trace Time

The `_morton_encode` bit-interleaving loops are Python loops that run during JAX tracing, not at runtime. XLA receives a fixed sequence of $16d$ bitwise operations and optimises them as a single computational block.

### 9.4 vmap over Particles

`jax.vmap(force_on_i)(jnp.arange(N_p))` vectorises the per-particle PP computation over the batch dimension. Since `W` is a Python integer, all arrays inside `force_on_i` have static shapes `(2W+1,)` and `(2W+1, d)`, enabling XLA to fuse the vmap into a batched matrix operation.

### 9.5 lax.scan and lax.while_loop

`lax.scan` represents the entire fixed-dt time loop as a single XLA program, equivalent to an unrolled loop with shared compilation. `lax.while_loop` handles variable iteration counts; both `cond_fn` and `body_fn` are compiled once and reused across iterations without Python re-entry.

### 9.6 Precision and x64 Mode

| `precision` | x64 enabled | Recommended for |
|------------|-------------|----------------|
| `"float64"` | yes | Default; required for accurate gravitational potentials |
| `"float32"` | no | GPU/MPS; ~2× memory reduction; acceptable for large $N$ |
| `"float16"` | no | Experimental; numerically unstable — emits a startup warning |

When `precision = "float64"`, `jax.config.update("jax_enable_x64", True)` is called before any JAX operations. The `md_cic_nd` function infers its output dtype from the input position array, so precision follows the chosen setting automatically.

### 9.7 Apple Silicon (MPS) Acceleration

JAX routes computation to the Metal Performance Shaders backend automatically on Apple M-series hardware. Float32 is recommended for MPS. No platform-specific code is required.

---

## 10. Power Spectrum Analysis

### 10.1 Estimator

The matter power spectrum is estimated from the CIC density field on the mass grid $B_m$:

$$\hat{P}(k) = \frac{V}{N^{2d}} |\hat{\delta}(\mathbf{k})|^2, \quad \delta = \rho / \bar{\rho} - 1, \tag{19}$$

where modes are averaged in logarithmic bins of width $\Delta(\log k)$.

### 10.2 CIC Window Correction

The CIC assignment introduces a mass-weighting window that suppresses power at high $k$. This is corrected by dividing by the squared CIC window function:

$$P_\text{corr}(k) = \hat{P}(k) \left/ \prod_{i=0}^{d-1} \text{sinc}^2\!\left(\frac{k_i \Delta x}{2\pi}\right). \right. \tag{20}$$

The correction is applied in each dimension independently, with a floor of $10^{-6}$ to prevent division by zero near the Nyquist frequency.

### 10.3 Shot Noise Subtraction

Discrete sampling of a continuous field introduces Poisson shot noise $1/\bar{n}$:

$$P_\text{true}(k) = P_\text{corr}(k) - 1/\bar{n}, \quad \bar{n} = N_p / V. \tag{21}$$

### 10.4 Output Format

Results are appended to `results/{config_name}/power_spectrum.csv` with columns `[step, a, k_1, Pk_1, k_2, Pk_2, ...]`. The `append_to_csv` function writes one row per chunk, building the evolution history incrementally. A companion function `plot_power_spectrum_evolution` reads this CSV and produces a multi-epoch $P(k)$ plot at the end of the run.

---

## 11. Output and Visualisation

### 11.1 VTK Particle Snapshots

Particle positions and momenta are saved as Legacy Binary VTK PolyData files in `results/{name}/vtk/particles/`. Key format details:

- All floating-point data written as big-endian 32-bit float (`>f4`)
- Two-dimensional positions zero-padded to $(N_p, 3)$ for VTK compatibility
- The `VERTICES` connectivity array is built as native `int32` then converted to big-endian `>i4` in one operation to avoid mixed-endianness corruption
- Momenta stored as a `VECTORS` point-data field named `momentum`
- Filename: `{config_name}_particles_a{aval}.vtk`

### 11.2 VTK Density Fields

The CIC density field is written as an ASCII `DATASET STRUCTURED_POINTS` file in `results/{name}/vtk/density/`. Key format details:

- For 2D grids: `nz = 1` with spacing equal to `res` in all three dimensions, preventing degenerate zero-thickness cells that produce grid-box rendering artefacts in ParaView
- Data written in Fortran (x-fastest) order via `rho.flatten(order='F')` to match the VTK `STRUCTURED_POINTS` convention
- Filename: `{config_name}_density_a{aval}.vtk`

### 11.3 End-of-Run Plots

Two matplotlib figures are generated from the trajectory at the end of a run (subject to the OOM guard):

- `density_evolution.png` — CIC density field at three key scale factors
- `particle_evolution.png` — subsampled particle positions at three key scale factors
- `power_spectrum.png` — $P(k)$ curves at all saved epochs, read from CSV

Plotting uses `with plt.style.context('dark_background')` (context manager, not global state mutation) and reproducible subsampling via `np.random.default_rng(seed=0)`.

---

## 12. Code Architecture

### 12.1 Module Structure

```
src/
  core/
    box.py           — Box: periodic domain, FFT wavenumber grids K and k
    ops.py           — md_cic_nd, InterpND, gradient_2nd_order, garfield
    filters.py       — Composable Fourier filters: Power_law, Scale, Cutoff, Potential
  physics/
    cosmology.py     — Cosmology dataclass: H(a), G_eff, growing_mode(a); presets
    system.py        — PoissonVlasov: PM and P3M force computation
    initial_conds.py — Zeldovich approximation; _grid_to_particles helper
  solver/
    state.py         — State NamedTuple; HamiltonianSystem ABC
    integrator.py    — leap_frog, step_chunk, step_chunk_adaptive, compute_dt
  utils/
    config_parser.py — load_config with key and range validation
    io.py            — write_vtk_particles, write_vtk_density
    analysis.py      — compute_power_spectrum, append_to_csv, plot_power_spectrum_evolution
    plotting.py      — plot_density_evolution, plot_particles
```

### 12.2 Data Flow

```
Config JSON
   → load_config (validation)
   → Cosmology + Box setup (B_mass, force_box)
   → garfield → phi (Gaussian random potential)
   → Zeldovich → initial State
   → PoissonVlasov (system)
   → step_chunk / step_chunk_adaptive (JIT-compiled)
   → Python chunk loop
      → NaN/Inf check
      → VTK write (optional)
      → power spectrum CSV append (optional)
   → end-of-run plots (OOM-guarded)
```

### 12.3 Key Design Patterns

**Immutable State.** `State(NamedTuple)` holds `(time, position, momentum)`. All integrator steps return new State objects — never mutated in-place. This is a hard requirement for `lax.scan` and `lax.while_loop`.

**HamiltonianSystem ABC.** `PoissonVlasov` implements `positionEquation(s)` and `momentumEquation(s)`. The ABC enforces this interface, making it straightforward to add alternative force models.

**Trace-time branching.** The `if self.solver == "p3m"` branch in `momentumEquation` is resolved at Python level during JAX tracing. JAX compiles a different XLA binary for each solver choice; there is no runtime conditional.

**Dimension agnosticism.** Every core routine is parameterised by `Box.dim`:
- `md_cic_nd`: corners unrolled as `product([0,1], repeat=dim)`
- `InterpND`: same corner structure for field reading
- `Zeldovich.u`: displacement computed for `range(dim)` axes
- `_morton_encode`: bit-interleaving over `range(self.box.dim)`
- `PoissonVlasov.momentumEquation`: force stacked over `range(dim)`
- `particle_mass`: scaled as $(N_f / N_m)^d$

### 12.4 Naming Conventions

| Identifier | Meaning |
|-----------|---------|
| `B_mass`, `bm` | Mass-resolution box ($N^d$) |
| `force_box`, `bf` | Force-resolution box ($(2N)^d$) |
| `x_grid` | Particle positions in grid units (divided by `res`) |
| `pos` / `position` | Particle positions in physical units (Mpc/h) |
| `pp_*` | Parameters for the PP short-range force |
| `_pm_force` | Private: PM long-range force |
| `_pp_force` | Private: PP short-range force (before negation in `momentumEquation`) |
| `_morton_encode` | Private: Morton Z-curve code |
| `_grid_to_particles` | Private: reshape `(dim, N, ...)` → `(N^dim, dim)` |

---

## 13. Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 2 | Spatial dimension (2 or 3) |
| `N` | int | — | Particles per side; $N^d$ total |
| `L` | float | — | Box side length [Mpc/h] |
| `A` | float | — | Amplitude scaling of initial potential |
| `seed` | int | — | Random seed for Gaussian random field |
| `a_start` | float | — | Initial scale factor |
| `a_end` | float | — | Final scale factor |
| `power_index` | float | — | Primordial spectral index $n_s$ |
| `H0` | float | 70.0 | Hubble constant [km/s/Mpc] |
| `OmegaM` | float | 1.0 | Matter density $\Omega_M$ |
| `OmegaL` | float | 0.0 | Cosmological constant $\Omega_\Lambda$ |
| `precision` | string | `"float64"` | `"float16"` \| `"float32"` \| `"float64"` |
| `solver` | string | `"pm"` | `"pm"` \| `"p3m"` |
| `pp_window` | int | 2 | P3M: half-window $W$; window covers $2W+1$ neighbours |
| `pp_softening` | float | 0.1 | P3M: gravitational softening $\varepsilon$ [Mpc/h]; also used as CFL resolution scale |
| `pp_cutoff` | float | 2.5 | P3M: cutoff in force-cell units ($r_\text{cut} = \ell_\text{cut} \cdot \Delta x_f$) |
| `timestepping` | string | `"fixed"` | `"fixed"` \| `"adaptive"` |
| `dt` | float | — | Fixed $\Delta a$ step (required when `timestepping="fixed"`) |
| `save_every` | int | 1 | Fixed stepping: leapfrog steps per snapshot chunk |
| `C_cfl` | float | 0.3 | Adaptive: CFL safety factor |
| `dt_min` | float | 0.001 | Adaptive: minimum $\Delta a$ |
| `dt_max` | float | 0.05 | Adaptive: maximum $\Delta a$ |
| `n_chunks` | int | 50 | Adaptive: number of output checkpoints |
| `save_vtk` | bool | false | Write VTK snapshots each chunk |
| `vtk_freq` | int | 1 | Write VTK every $n$ chunks |
| `save_power_spectrum` | bool | false | Compute and append $P(k)$ each chunk |
| `oom_threshold_gb` | float | 4.0 | Skip end-of-run trajectory stacking if estimated size exceeds this [GB] |

---

## 14. Available Configurations

**`default.json`** — 2D EdS, $N = 128$, $L = 50$ Mpc/h, float64, PM, fixed $\Delta a = 0.02$. $128^2 = 16{,}384$ particles from $a = 0.02$ to $a = 1.0$. Recommended starting point; runs in minutes on CPU.

**`high_res.json`** — 2D LCDM ($H_0=68$, $\Omega_M=0.31$, $\Omega_\Lambda=0.69$), $N = 256$, float32, $\Delta a = 0.015$. $65{,}536$ particles. Used for power spectrum convergence and window-function tests.

**`3d_default.json`** — 3D EdS, $N = 64$, float32, fixed $\Delta a = 0.02$. $64^3 = 262{,}144$ particles. Primary 3D validation config; accessible on laptop-class hardware.

**`3d_heigh_res.json`** — 3D LCDM, $N = 128$, float32, fixed $\Delta a = 0.02$, `save_every = 10`. $\approx 2.1 \times 10^6$ particles. Production PM run. Recommended for GPU or Apple Silicon.

**`3d_heigh_res_p3m.json`** — identical to `3d_heigh_res.json` but with `"solver": "p3m"`. Enables direct PM vs P3M comparison at the same resolution.

**`3d_256.json`** — 3D LCDM, $N = 256$, float32, $L = 50$ Mpc/h, fixed $\Delta a = 0.02$. $\approx 1.7 \times 10^7$ particles. GPU-scale run; requires the OOM guard or VTK-only output.

**`3d_visual.json`** — 3D LCDM, $N = 144$, float32, $L = 100$ Mpc/h, $A = 10$, **adaptive stepping** ($C_\text{CFL}=0.3$, $\Delta a \in [0.0005, 0.02]$, 65 checkpoints). Optimised for ParaView visualisation. The large amplitude $A = 10$ combined with fixed stepping would violate the CFL condition during structure formation; adaptive stepping is mandatory for this config.

**`p3m_adaptive.json`** — 2D EdS, $N = 64$, float64, P3M ($W = 4$, $\varepsilon = 0.2$ Mpc/h, $\ell_\text{cut} = 2.5$), adaptive stepping ($C_\text{CFL}=0.3$, $\Delta a \in [0.001, 0.05]$, 50 checkpoints). Reference config for the full P3M + adaptive pipeline.

---

## 15. Validation Tests

The test suite comprises **54 tests** across two files, all passing under `pytest tests/`. Float64 is enabled for the entire session via a session-scoped `conftest.py` autouse fixture.

### 15.1 `tests/test_core.py` — 22 unit and integration tests

**Phases 1–3 (infrastructure)**

1. `test_md_cic_2d_mass_conservation` — total deposited mass equals particle count to $10^{-5}$.
2. `test_interp2d_identity` — `Interp2D` returns exact cell values at integer positions.
3. `test_box_wave_numbers` — `Box` generates correct $k_\text{min} = 2\pi/L$ and $k_\text{max} = N\pi/L$.
4. `test_gradient_2nd_order` — FD gradient of $\sin(x)$ agrees with $\cos(x)$ to $10^{-2}$ on a 64-point grid.
5. `test_box_3d_shapes` — `Box(3, 64, 50.0)` yields `K.shape = (3, 64, 64, 64)`.
6. `test_potential_filter_3d` — `Potential()(box.K)` returns shape $(N, N, N)$.
7. `test_garfield_3d` — 3D Gaussian random field returns correct shape.
8. `test_md_cic_nd_mass_conservation_3d` — 3D CIC mass conservation.
9. `test_md_cic_nd_unit_deposit_3d` — particle at integer position deposits entirely into one cell.
10. `test_interpnd_3d_grid_points` — 3D interpolation at integer positions is exact.
11. `test_zeldovich_3d_shape` — 3D Zeldovich state shape and mean CIC density $\approx 1.0$.

**Phases 4 and 6 (PM and P3M forces)**

12. `test_pm_uniform_force_is_zero` — uniform grid $\Rightarrow$ PM acceleration $= 0$ everywhere.
13. `test_p3m_close_pair_force_larger_than_pm` — P3M magnitude $>$ PM for a close pair.
14. `test_pp_force_zero_beyond_cutoff` — PP correction $= 0$ for separation $> r_\text{cut}$.
15. `test_pp_erfc_less_than_direct` — erfc weight $< 1$ at short range.

**Phase 7 (adaptive stepping)**

16. `test_compute_dt_decreases_with_velocity` — higher velocities yield smaller $\Delta a_\text{CFL}$.
17. `test_compute_dt_respects_bounds` — output in $[\Delta a_\text{min}, \Delta a_\text{max}]$ for all velocity scales.

**VTK I/O regression tests**

18. `test_write_vtk_particles_creates_file` — particle file is non-empty binary VTK with correct header.
19. `test_write_vtk_particles_2d_padded` — 2D positions are zero-padded to $(N, 3)$.
20. `test_write_vtk_density_2d_creates_file` — 2D density file contains correct `DIMENSIONS` and keyword.
21. `test_write_vtk_density_3d_creates_file` — 3D density file contains correct `DIMENSIONS`.

**3D end-to-end**

22. `test_3d_pm_simulation_advances_and_stays_finite` — 2 full 3D PM leapfrog steps produce finite positions and momentum, and advance time.

### 15.2 `tests/test_physics.py` — 32 physics validation tests

Organised into seven test classes. Each test checks a specific physical property against an analytic prediction.

**`TestCosmology`**

1. `test_eds_hubble_at_unity` — $H(a=1) = H_0$ exactly for EdS.
2. `test_eds_hubble_scaling` — $da/dt = H_0 a^{-1/2}$ to $10^{-10}$ relative error at 4 scale factors.
3. `test_lcdm_hubble_at_unity` — flat LCDM: $H(1) = H_0$ to $10^{-10}$.
4. `test_eds_growth_factor_linear` — $D(a) \propto a$ for EdS at $a \in [2, 4]$ (where the integration offset is negligible), within 2%.
5. `test_gravitational_coupling` — $G_\text{eff} = \frac{3}{2}\Omega_M H_0^2$ to $10^{-10}$.

**`TestForcePipeline`**

6. `test_uniform_density_zero_force` — $\delta = 0 \Rightarrow$ PM acceleration $= 0$ to $10^{-6}$.
7. `test_pm_force_newton_third_law` — total force on a 2-particle system $= 0$ (momentum conservation).
8. `test_pm_force_direction_attractive` — force on a particle points toward its neighbour (gravity is attractive), verified with nearest-image geometry.
9. `test_single_fourier_mode_potential` — single cosine density mode → Poisson gives $\phi = -G A/k_0^2 \cos(kx)$ to $10^{-4}$.
10. `test_gradient_of_potential_is_force` — `gradient_2nd_order` of $\sin(x)$ recovers $\cos(x)$ to $10^{-2}$.

**`TestP3MForce`**

11. `test_pm_plus_pp_closer_to_direct_newtonian` — P3M magnitude $>$ PM for a sub-cell-separation pair; PP raw force is negative (gets negated in `momentumEquation` to give attractive direction).
12. `test_erfc_split_sums_to_direct` — $\text{erfc}(r/\alpha) + \text{erf}(r/\alpha) = 1$ to $10^{-6}$ for all $r$.
13. `test_pp_force_finite_and_bounded` — PP force is finite and bounded for 32 random particles.
14. `test_p3m_cutoff_validation` — $r_\text{cut} \geq L/2$ raises `ValueError` at construction.

**`TestIntegrator`**

15. `test_leapfrog_zero_force_constant_momentum` — zero force $\Rightarrow$ momentum unchanged after one step.
16. `test_leapfrog_advances_time` — `state.time` increments by $\Delta a$ after one step.
17. `test_leapfrog_position_drift_direction` — positive momentum $\Rightarrow$ position increases.
18. `test_leapfrog_approximate_time_reversibility` — forward then backward step returns to within $O(\Delta a^3)$ of origin.
19. `test_adaptive_step_lands_exactly_at_target` — `step_chunk_adaptive` lands exactly on `a_target`.

**`TestZeldovichICs`**

20. `test_particle_count` — state has shape $(N^d, d)$ for positions and momenta.
21. `test_mean_density_unity` — CIC mean density $= 1.0$ to $10^{-4}$.
22. `test_small_displacement_at_early_time` — max displacement $< L/4$ at $a_\text{init} = 0.02$.
23. `test_positions_within_box` — all positions in $(-0.1, L+0.1)$ Mpc/h (Zeldovich does not apply periodic wrapping; small overruns at boundaries are expected).
24. `test_momentum_proportional_to_displacement_eds` — $\mathbf{p} = \mathbf{x} - \mathbf{q}$ exactly (Zeldovich identity $\mathbf{p} = a\mathbf{u}$).

**`TestPowerSpectrum`**

25. `test_white_noise_flat_spectrum` — white-noise field produces flat $P(k)$ (ratio of high-$k$ to low-$k$ power within 50%).
26. `test_shot_noise_scaling_with_particle_count` — shot noise $\propto 1/N_p$ as expected.
27. `test_power_spectrum_total_variance` — Parseval's theorem: $\sum_k P(k) \Delta k \approx \sigma^2_\delta$.
28. `test_cic_window_correction_increases_high_k_power` — corrected $P(k)$ $>$ uncorrected at high $k$.
29. `test_power_spectrum_output_shape` — `compute_power_spectrum` returns arrays of equal length.
30. `test_3d_power_spectrum_shape` — 3D power spectrum output shape correct.

**`TestEndToEnd`**

31. `test_single_pm_step_runs_and_advances` — one full PM leapfrog step: finite state, time advances.
32. `test_p3m_step_runs_and_advances` — one full P3M leapfrog step: finite state, time advances.

---

## 16. What Can Be Studied with This Code

1. **Linear perturbation theory validation** — at early times, $P(k, a)$ should grow as $D^2(a) \cdot P_\text{prim}(k)$.
2. **Zeldovich pancake formation** — EdS with $n_s = -0.5$ forms characteristic sheet structures visible in VTK snapshots.
3. **Non-linear power spectrum evolution** — $P(k)$ from $a = 0.02$ to $a = 1$ can be compared to Peacock-Dodds (1996) or Halofit (Smith et al. 2003).
4. **Cosmology dependence** — EdS ($D \propto a$) vs LCDM growth suppression visible in the time-tagged power spectrum CSV.
5. **PM vs P3M force accuracy** — comparing PM and P3M runs at the same resolution quantifies the sub-cell force improvement near halo centres (`3d_heigh_res.json` vs `3d_heigh_res_p3m.json`).
6. **Adaptive vs fixed stepping** — the adaptive integrator uses larger steps at early times and automatically refines near shell-crossing; comparing $P(k)$ between runs tests CFL stability (`p3m_adaptive.json` vs `default.json`).
7. **Resolution convergence** — 2D: $N \in \{64, 128, 256\}$; 3D: $N \in \{32, 64, 128\}$. Tests force and particle resolution convergence of $P(k)$.

---

## 17. Limitations and Future Work

**TSC mass assignment (Phase 5):** The Triangular-Shaped Cloud scheme distributes mass over $3^d$ cells using a parabolic kernel, reducing aliasing relative to CIC. The corresponding deconvolved Green's function for the Fourier-space potential is not implemented.

**Zeldovich momentum for LCDM:** The initial momentum is set as $\mathbf{p} = a_\text{init} \cdot \mathbf{u}$, which is exact for EdS ($D(a) = a$, growth rate $f = 1$) but omits the $f(a) \cdot H(a) \cdot D(a)$ prefactor for LCDM cosmologies. This introduces a small error in initial particle velocities for `high_res.json` and `3d_heigh_res.json`; particle positions at $a_\text{start}$ are unaffected.

**PP force-split accuracy:** The current PP correction uses the erfc splitting kernel but does not subtract an analytic model of the PM force at short separations. In the original Hockney-Eastwood P3M, this analytic subtraction prevents double-counting at intermediate $r$. The present implementation is equivalent to assuming the PM force is negligible below $r_\text{cut}$, which holds when $r_\text{cut} \ll \Delta x_f$ — the typical operating regime.

**Large-$N$ PP scaling:** The Morton-sorted sliding window is $O(N_p \cdot W)$. For $W \gtrsim 10$ or very large $N_p$, spatial hashing (which has $O(N_p)$ lookup in the average case) would outperform the fixed window approach.

---

## References

Efstathiou, G., Davis, M., White, S. D. M., & Frenk, C. S. 1985, ApJS, 57, 241

Heath, D. J. 1977, MNRAS, 179, 351

Hockney, R. W. & Eastwood, J. W. 1988, *Computer Simulation Using Particles* (Bristol: IOP)

Peacock, J. A. & Dodds, S. J. 1996, MNRAS, 280, L19

Smith, R. E., et al. 2003, MNRAS, 341, 1311

Springel, V. 2005, MNRAS, 364, 1105

Springel, V. 2010, MNRAS, 401, 791

Zel'dovich, Ya. B. 1970, A&A, 5, 84
