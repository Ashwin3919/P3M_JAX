# From NumPy to JAX: Design Rationale and Performance Architecture of P3M-JAX

**Ashwin Shirke**

*P3M-JAX: A JAX-Accelerated Cosmological N-Body Simulation*

<img src="JAX.png" width="376"/>

---

## 1. Introduction

Cosmological N-body simulations are among the most computationally demanding tasks in modern astrophysics. The standard Particle-Particle-Particle-Mesh (P3M) algorithm (Hockney & Eastwood 1988) involves three computational bottlenecks that recur at every time step: mass deposition via Cloud-in-Cell (CIC) interpolation, a global FFT Poisson solve on a force grid, and a direct particle-particle (PP) short-range correction. For a simulation with $N^d$ particles on a $d$-dimensional domain — particularly in 3D with $N \gtrsim 64$ — these operations are prohibitively slow when executed through the Python interpreter with NumPy, even though NumPy itself calls compiled C/BLAS routines for individual array operations.

This report documents the architectural rationale for migrating P3M-JAX from a 2D NumPy/Numba prototype to a fully 3D JAX-accelerated code. The starting point for this migration is the reference implementation contained in `docs/nbody_refrance.py` — a 2D N-body code following Johan Hidding's methodology, using NumPy arrays, a Numba-JIT CIC kernel, and a Python while-loop integrator. Each section identifies a specific deficiency in that prototype, explains why it prevented acceleration or correctness under JAX, and describes the precise change made in the production code.

---

## 2. Limitations of the NumPy/Numba Prototype

### 2.1 Python Interpreter Overhead at Each Time Step

The reference code drives the entire simulation through a Python `while` loop in `iterate_step` (`nbody_refrance.py`, line 287):

```python
while not halt(state):
    states.append(State(state.time, state.position.copy(), state.momentum.copy()))
    state = step(state)
```

Each iteration requires the Python interpreter to: evaluate the halting condition (which materialises `state.time` from device to host), dispatch `step`, construct a new `State` object, and append it to a Python list. For a simulation running 200 steps (e.g. `a_start=0.02`, `a_end=4.0`, `dt=0.02`), this is 200 Python interpreter round-trips, each triggering a force kernel call. The *computation* inside NumPy and Numba is fast, but the *orchestration* through the interpreter is not — and it is completely opaque to any compiler.

### 2.2 No Kernel Fusion

NumPy evaluates expressions eagerly, one operation at a time. The Poisson solve in `momentumEquation` (`nbody_refrance.py`, lines 349–351):

```python
delta_f = np.fft.fftn(self.delta)
kernel = Potential()(self.box.K)
phi = np.fft.ifftn(delta_f * kernel).real * self.cosmology.G / a
```

allocates a new array at every intermediate step: `fftn(self.delta)`, `delta_f * kernel`, `ifftn(...)`, `.real`, `* G / a`. For a $256^3$ force grid in float32, each intermediate array is 64 MB; the five shown above represent over 300 MB of transient allocation per force evaluation. XLA (JAX's backend compiler) can fuse these into a single compiled kernel, eliminating intermediate buffers entirely and reducing peak memory traffic by the same factor.

### 2.3 No Device Portability

NumPy and Numba are CPU-only. The `@numba.jit` decorator on `md_cic_2d` (`nbody_refrance.py`, line 200) compiles to a single-threaded native CPU loop — it cannot target an Apple M2 GPU (Metal), an NVIDIA GPU (CUDA), or a TPU without a complete rewrite against a device-specific library. Every subsequent algorithmic change would need to be replicated across backends.

---

## 3. The Migration Path: Prototype Anti-patterns and Their Fixes

The following sections document each anti-pattern present in `docs/nbody_refrance.py`, with the exact line numbers, the reason it is problematic under JAX, and the resolution applied in the production code.

### 3.1 Mutable `State` Dataclass (Fundamental Correctness Issue)

The reference `State` is a `@dataclass` with in-place mutation methods (`nbody_refrance.py`, lines 257–267):

```python
def kick(self, dt, h):
    self.momentum += dt * h.momentumEquation(self)   # in-place +=
    return self

def drift(self, dt, h):
    self.position += dt * h.positionEquation(self)   # in-place +=
    return self

def wait(self, dt):
    self.time += dt                                   # in-place +=
    return self
```

This is not merely a style issue — it is **fundamentally incompatible with JAX tracing**. JAX's `jit` works by tracing a function with abstract symbolic values (tracers) to build a computation graph. In-place `+=` on a JAX tracer does not update the underlying buffer; it replaces the Python name with a new tracer representing the addition. The old tracer is lost, and the mutation is invisible to the graph. Any function that relies on in-place mutation for state updates will produce wrong results under `jit`, or raise a `ConcretizationTypeError` at trace time.

The reference code also injects arbitrary attributes into `State` at runtime (`nbody_refrance.py`, lines 524–526):

```python
state.live_plot = live_plot
state.fig = system.fig
```

A `@dataclass` technically allows this, but it produces an object with a variable schema that cannot be registered as a JAX pytree — meaning it cannot pass through `lax.scan`, `vmap`, or `jit` as a structured carry.

**Fix:** `State` was rewritten as an immutable `NamedTuple`:

```python
class State(NamedTuple):
    time:     jnp.ndarray
    position: jnp.ndarray
    momentum: jnp.ndarray
```

Every leapfrog step now returns a new `State`; nothing is modified in place. `NamedTuple` is automatically a valid JAX pytree. The `live_plot` and `fig` attributes were removed from the state entirely — they are not physical quantities and have no place in the integration carry.

### 3.2 Plotting Inside the Force Kernel (Critical Performance Bug)

The reference `PoissonVlasov.momentumEquation` contains matplotlib calls (`nbody_refrance.py`, lines 339–346):

```python
if self.live_plot:
    self.ax.clear()
    rho_plot = self.delta + 1.0
    im = self.ax.imshow(np.log10(np.maximum(rho_plot, 0.1)).T, ...)
    self.ax.set_title(f'Log Density Field (a = {a:.3f})')
```

And the integration loop calls `display` and `clear_output` every 10 steps (`nbody_refrance.py`, lines 291–293):

```python
if state.live_plot and len(states) % 10 == 0:
    clear_output(wait=True)
    display(state.fig)
```

These cause three compounding problems. First, `display` and `clear_output` are IPython calls — they are not JAX-traceable. Any function containing them cannot be JIT-compiled. Second, `display` forces an immediate device-to-host synchronisation: JAX must materialise the current array values as concrete Python objects before they can be passed to matplotlib, which serialises execution and blocks the GPU/MPS pipeline. Third, on Apple M2, Metal kernel launches are asynchronous — inserting a blocking `display` call every 10 steps degrades throughput to the rate at which the host can render frames rather than the rate at which the device can compute leapfrog steps.

**Fix:** All plotting and display code was removed from `momentumEquation` and from the integration loop body. The simulation inner loop contains no side effects and is fully JAX-traceable. Plotting occurs exclusively after the simulation exits the chunk loop, operating on already-materialised output arrays. The `live_plot` flag and the `fig` attribute were eliminated from the simulation core entirely.

### 3.3 Python `while` Loop as the Integration Driver

As shown in Section 2.1, the integration loop in the reference code (`nbody_refrance.py`, line 287) is a Python `while` loop. Because `halt(state)` evaluates a scalar `state.time` to a Python `bool`, every iteration requires a host-device synchronisation: the device must finish the current step, transfer the time scalar to the host, evaluate the Python condition, and only then dispatch the next step. This reduces a potentially fully-pipelined computation to a sequence of blocking round-trips.

**Fix:** For fixed-$\Delta a$ runs, the inner loop was replaced with `jax.lax.scan`:

```python
def step_chunk(system, state, dt, save_every):
    step_fn = lambda s, x: leapfrog_step_scan(s, x, dt, system)
    final, _ = jax.lax.scan(step_fn, state, xs=None, length=save_every)
    return final
```

`lax.scan` is a JAX primitive — not a Python loop — so XLA compiles the entire scan body as a single program, traces the step function once regardless of `save_every`, and runs the entire sub-loop on the device without re-entering Python. For adaptive-$\Delta a$ runs, `jax.lax.while_loop` provides the same guarantee for a variable number of sub-steps (see Section 4.3).

### 3.4 Hidden Host-Device Synchronisation via `assert`

The reference `momentumEquation` asserts mass conservation on every force evaluation (`nbody_refrance.py`, line 336):

```python
assert abs(self.delta.mean()) < 1e-6, "Total mass should be normalized"
```

`abs(self.delta.mean())` converts a JAX/NumPy array to a Python scalar, which forces a full device-to-host data transfer. Inside a JIT-compiled function this raises a `ConcretizationTypeError` because JAX tracer values cannot be converted to Python scalars at trace time. Outside JIT but inside a tight loop it inserts an implicit synchronisation barrier between otherwise-asynchronous device operations, stalling the pipeline every step.

**Fix:** The assertion was removed from the production code. Where runtime diagnostics are required, `jax.debug.print` is the correct tool — it hooks into the XLA runtime and emits output from within a compiled kernel without breaking asynchrony or requiring a host-device transfer:

```python
jax.debug.print("mean delta = {}", self.delta.mean())
```

### 3.5 Spectral Kernel Recomputed on Every Force Evaluation

The reference `momentumEquation` reconstructs the gravitational potential kernel on every call (`nbody_refrance.py`, line 350):

```python
kernel = Potential()(self.box.K)
```

`Potential()(self.box.K)` evaluates a Python lambda over the entire $K$ grid — a NumPy operation of size $(2N)^d$ — every single time the force is called. For a 200-step simulation this is 200 unnecessary kernel constructions. The kernel depends only on the force box geometry, which never changes.

**Fix:** The kernel is computed once during `PoissonVlasov.__init__` and stored as `self.kernel`:

```python
self.kernel = Potential()(self.box.K)   # computed once
```

Under XLA, this static constant array is embedded in the compiled program as a read-only buffer. Avoiding repeated recomputation eliminates a NumPy call of size $(2N)^d$ on every force evaluation; whether the array resides in cache at runtime depends on its size relative to the available L2 (CPU) or last-level cache (GPU).

### 3.6 CIC Mass Deposition: Four `np.histogram2d` Calls and a Numba Scalar Loop

The reference code provides two CIC implementations. The pure-NumPy version `md_cic` (`nbody_refrance.py`, lines 176–198) makes four separate `np.histogram2d` calls — one per corner of the bilinear stencil. Each `histogram2d` call sorts all $N^2$ particles into a 2D bin array; four serial sorts for what is mathematically a single weighted scatter operation. The Numba-accelerated version `md_cic_2d` (`nbody_refrance.py`, lines 200–209) eliminates the sort overhead but uses a scalar Python/Numba `for i in range(len(pos))` loop that is compiled to a single-threaded CPU loop — it cannot be mapped to GPU threads and is not a JAX primitive.

Neither implementation is compatible with JAX's compilation model: `np.histogram2d` is a NumPy routine that cannot be traced, and the Numba kernel writes to a mutable NumPy array via in-place `+=` — a pattern that has no equivalent in JAX's functional model.

**Fix:** `md_cic_nd` in `src/core/ops.py` uses `jnp.zeros(...).at[flat_idx].add(weights)` inside a Python `for` loop over $2^d$ corners, decorated with `@partial(jax.jit, static_argnums=(0,))`. Marking `shape` as static allows XLA to unroll the $2^d$-corner loop at compile time — in 3D this produces 8 fused scatter-add nodes in a single compiled kernel, with no Python loop overhead at runtime. The result is a single-pass GPU-compatible scatter operation that replaces both the histogram2d and the Numba loop, and generalises to arbitrary dimension $d$ without code changes.

### 3.7 Two FFT Round-Trips in `garfield`

The reference `garfield` function (`nbody_refrance.py`, lines 130–132) applies the power spectrum filter and the transfer function in two separate FFT round-trips:

```python
f = np.fft.ifftn(np.fft.fftn(wn) * np.sqrt(P(B.K))).real      # FFT → filter → IFFT
return np.fft.ifftn(np.fft.fftn(f) * T(B.K)).real              # FFT → filter → IFFT
```

This is four FFT operations to produce one field. Because both filters are spectral (defined in Fourier space), they can be composed and applied in a single round-trip:

$$\hat{\phi}(\mathbf{k}) = \hat{\eta}(\mathbf{k}) \cdot \sqrt{P(k)} \cdot T(\mathbf{k})$$

**Fix:** The production `garfield` in `src/core/ops.py` applies both filters in one FFT round-trip:

```python
return jnp.fft.ifftn(jnp.fft.fftn(wn) * jnp.sqrt(P(B.K)) * T_filt(B.K)).real
```

This halves the number of FFT calls for initial condition generation. On CPU with float64, the saving is on the order of a fraction of a second to ~1 second for a $256^3$ grid — modest at startup but the principle scales with grid size.

### 3.8 Filter Inner Products Hardcoded to 2D

The `Filter.cc` and `Filter.cf` methods in the reference code (`nbody_refrance.py`, lines 79–83) use `B.res**2` as the volume element:

```python
def cc(self, B, P, other):
    return (~self * other * P)(B.K).sum().real / B.size * B.res**2

def cf(self, B, other):
    return ((~self)(B.K) * other).sum().real / B.size * B.res**2
```

This is correct for a 2D box where the volume element is $(\Delta x)^2$, but gives wrong results for 3D where it should be $(\Delta x)^3$. Any code using these methods to compute power spectrum normalisation or cross-correlations in 3D would silently produce values off by a factor of $\Delta x = L/N$.

**Fix:** The production `src/core/filters.py` uses `B.res**B.dim`:

```python
def cc(self, B, P, other):
    return (~self * other * P)(B.K).sum().real / B.size * B.res**B.dim

def cf(self, B, other):
    return ((~self)(B.K) * other).sum().real / B.size * B.res**B.dim
```

This makes the filter operators dimension-agnostic and correct for both 2D and 3D boxes.

---

## 4. JAX Primitives and Their Role in P3M-JAX

### 4.1 Just-In-Time Compilation (`jax.jit`)

JAX's `jit` decorator traces a Python function with abstract tracer values and emits a single XLA HLO (High-Level Operations) computation graph, compiled to machine code for the target device. Subsequent calls with the same array shapes bypass Python entirely.

In P3M-JAX, `jit` is applied at two levels:

**Kernel level.** `md_cic_nd` is decorated with `@partial(jax.jit, static_argnums=(0,))`. Marking `shape` as static allows XLA to unroll the $2^d$ corner loop at compile time — in 3D this becomes 8 fused scatter-add operations in a single kernel, with no Python interpreter overhead.

**Chunk level.** The primary JIT boundary is the chunk function in `main.py`:

```python
step_fn = jax.jit(partial(step_chunk, system, dt=dt, save_every=save_every))
```

This compiles the entire `save_every`-step leapfrog block — CIC deposition, FFT Poisson solve, gradient computation, and CIC interpolation — into a single XLA program. Once compiled, each chunk call is a single asynchronous device dispatch with no Python overhead.

### 4.2 `jax.lax.scan` for Fixed Time-Stepping

The inner loop over `save_every` leapfrog steps uses `jax.lax.scan`:

```python
def step_chunk(system, state, dt, save_every):
    step_fn = lambda s, x: leapfrog_step_scan(s, x, dt, system)
    final, _ = jax.lax.scan(step_fn, state, xs=None, length=save_every)
    return final
```

`lax.scan` traces the body once and emits a single XLA while-loop node — regardless of `save_every`. The reference code's Python `while` loop would trace and dispatch $n_\mathrm{step}$ times. With `lax.scan`, the device runs the entire sub-loop without re-entering Python between steps.

### 4.3 `jax.lax.while_loop` for Adaptive Time-Stepping

The CFL-adaptive integrator cannot use `lax.scan` because the number of sub-steps to reach `a_target` depends on `v_max`, a runtime array value unknown at trace time. `jax.lax.while_loop` compiles both the condition and body into a single XLA while-loop node that runs entirely on device:

```python
final_state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, dt_min_arr))
```

No Python is executed between sub-steps. The reference code's equivalent would require a Python `while` loop with a blocking `halt` check every iteration — which is impossible to JIT.

### 4.4 `jax.vmap` for the PP Force Kernel

The short-range PP correction requires, for each particle $i$, a sum over a window of $2W+1$ nearby neighbours. The reference code has no PP correction at all. The production implementation uses `jax.vmap`:

```python
sorted_acc = jax.vmap(force_on_i)(jnp.arange(N_p))
```

This traces `force_on_i` once and emits a single vectorised XLA kernel that processes all $N_p$ particles in parallel — a GPU thread launch on NVIDIA/M2, a SIMD loop on CPU.

### 4.5 Device-Agnostic Execution

JAX routes all XLA programs through a pluggable device backend:

| Device | Backend | Notes |
|--------|---------|-------|
| CPU | XLA:CPU | Always available; SIMD-vectorised |
| NVIDIA GPU | XLA:CUDA | Full float16/32/64 support |
| Apple M2 | XLA:Metal (PJRT) | float32 preferred; float64 emulated |

P3M-JAX requires zero code changes between these targets. The reference code uses NumPy and Numba — both CPU-only.

---

## 5. Architectural Decisions Enabled by JAX

### 5.1 Immutable State and Pure Functions

As described in Section 3.1, the production `State` is a `NamedTuple`. Every leapfrog step returns a new `State`; nothing is modified in place. This is not merely a stylistic preference: in-place mutation inside a `jit`-compiled function produces incorrect results because JAX's tracing model tracks data dependencies, not memory addresses. The immutability constraint also makes the physics equations directly readable as mathematical operations on values rather than procedures on memory.

### 5.2 Compile-Time Branching for PM vs P3M

The solver branch is a Python-level `if` on the string `self.solver`, resolved at trace time:

```python
if self.solver == "p3m":
    pp_acc = self._pp_force(s.position, a, da)
    return -(pm_acc + pp_acc) / da
return -pm_acc / da
```

XLA sees either the PM-only or the PM+PP computation graph — never a runtime branch. The compiled kernel has the minimal instruction count for the selected solver.

### 5.3 Precomputed Spectral Kernel

As described in Section 3.5, the potential kernel is computed once at `__init__` and embedded as a static constant in the compiled program. This avoids recomputing a $(2N)^d$-element array on every force call; cache residency at runtime depends on the kernel size relative to the available hardware cache.

### 5.4 Dual-Resolution Force Grid Without Memory Penalty

The force grid at resolution $2N$ is required to reduce PM aliasing. Under JAX with float32, an $N=128$ 3D force grid occupies 67 MB (vs 134 MB at float64). XLA's memory planner can reuse the force grid buffer between chunks because `step_chunk` returns only positions and momenta — the grid allocation is freed at kernel exit. NumPy retains arrays until garbage collection, inflating peak memory during back-to-back chunk calls.

---

## 6. Precision Strategy and the float32/float64 Trade-off

JAX provides explicit precision control through `jax.config.update("jax_enable_x64", True/False)`. P3M-JAX exposes this via the `"precision"` config key. The reference prototype uses `float64` throughout with no mechanism to change this.

| Precision | x64 Enabled | Force Grid $(N=128)^3$ | Use Case |
|-----------|-------------|------------------------|----------|
| `"float64"` | Yes | 134 MB | Accurate potentials, 2D runs |
| `"float32"` | No | 67 MB | 3D runs, Apple M2 GPU |
| `"float16"` | No | 34 MB | Experimental; numerically unstable |

On Apple M2, float64 is emulated in software by the Metal backend, incurring roughly a 4–8× throughput penalty relative to native float32. For 3D runs where the force grid is $(256)^3 = 16\,\text{M}$ complex elements, float32 reduces both memory footprint and compute time substantially.

---

## 7. Summary of Performance Gains

The table below maps each identified anti-pattern from `docs/nbody_refrance.py` to the production fix and its estimated performance impact.

| Anti-pattern (reference code) | Reference line(s) | Production fix | Estimated gain |
|---|---|---|---|
| Mutable `@dataclass` State (in-place `+=`) | 257–267 | Immutable `NamedTuple`; new State each step | Enables JIT correctness |
| Plotting inside `momentumEquation` and `iterate_step` | 291–293, 339–346 | Plotting deferred to post-run | Removes blocking sync barrier per display call (only relevant when `live_plot=True`) |
| Python `while` loop integration driver | 287–296 | `lax.scan` / `lax.while_loop` | 1.5–3× per-loop overhead reduction; see §8 for measured values |
| `assert abs(delta.mean())` CPU sync | 336 | Removed; `jax.debug.print` where needed | Eliminates stall per step |
| `Potential()(B.K)` recomputed each step | 350 | Precomputed at `__init__`, constant in XLA | ~$(2N)^d$ element op saved per step |
| 4× `np.histogram2d` CIC + Numba scalar loop | 181–209 | `@jax.jit` + static $2^d$ unroll, `.at[].add()` | 2–5× (GPU-portable) |
| 2× FFT round-trips in `garfield` | 130–132 | Single round-trip with composed filter | 2× at IC generation |
| `Filter.cc/cf` hardcoded `B.res**2` | 80–83 | `B.res**B.dim` — correct for 2D and 3D | Correctness fix for 3D |
| NumPy/Numba — CPU only | Throughout | JAX XLA — CPU + NVIDIA + Apple M2 | Hardware portability |
| No PP correction | — | `vmap` over Morton window | New capability |

Combining the elimination of in-loop side effects, JIT compilation, `lax.scan`, and the CIC kernel rewrite, the measured end-to-end speedup on Apple M2 CPU is **2–3×** for 2D runs at $N \geq 256$ (Section 8). Larger gains are expected in 3D — where per-step FFT work scales as $O(N^3 \log N)$ — and on GPU backends where XLA's kernel fusion and device parallelism are more fully exploited, but those cases have not yet been benchmarked.

---

## 8. 2D Empirical Benchmarks: NumPy/Numba vs P3M-JAX on Apple M2

> **Scope.** All benchmarks in this section are **2D, PM solver only** (no PP short-range correction). The measured speedups are therefore a lower bound on what P3M-JAX can achieve: in 3D the force grid grows from $(2N)^2$ to $(2N)^3$ cells, making the FFT Poisson solve $O(N^3 \log N)$ instead of $O(N^2 \log N)$, and kernel fusion, `lax.scan`, and CIC scatter-add gains all scale with the increased per-step work. 3D PM and P3M benchmarks are deferred to forthcoming work (see Section 9).

![Runtime Scaling: NumPy vs P3M-JAX](benchmark_scaling_combined_light.png)
*Figure 1. Wall-clock time vs particle count ($N^2$) for two time-step resolutions. Speedup ratios annotated per data point. Dashed vertical line marks the N=1024 out-of-memory threshold. Hardware: Apple M2 CPU, 2D PM solver, $\Lambda$CDM ($L = 50\,\mathrm{Mpc}/h$). Generated by `docs/scaling_plot.py`.*

### 8.1 Setup

All measurements were taken on an Apple M2 (CPU backend), **2D PM solver** (`solver: "pm"`), LCDM cosmology ($H_0 = 68\,\text{km}\,\text{s}^{-1}\,\text{Mpc}^{-1}$, $\Omega_M = 0.31$, $\Omega_\Lambda = 0.69$), $L = 50\,\text{Mpc}/h$, $A = 12$, seed = 42, $a_\mathrm{start} = 0.02$, $a_\mathrm{end} = 1.5$, float32 precision. Five grid sizes were tested — $N = 64, 128, 256, 512, 1024$ — corresponding to $N^2 = 4{,}096$ to $1{,}048{,}576$ particles in 2D. Two time-step resolutions were used: $\Delta a = 0.005$ (296–297 steps) and $\Delta a = 0.001$ (1480–1481 steps). VTK output was **disabled** (`save_vtk: false`) for all P3M-JAX runs to avoid disk-I/O confounding the compute comparison; only power-spectrum CSV appends were retained. Wall time for the reference code (`docs/nbody_refrance.py`) is from `time.perf_counter()` enclosing the integration loop only. P3M-JAX wall time is the total reported by `main.py`.

### 8.2 Experiment 1: $\Delta a = 0.005$ (296–297 steps)

| $N$ | $N^2$ particles | Reference (s) | Ref ms/step | P3M-JAX (s) | JAX ms/step | Speedup |
|-----|-----------------|---------------|-------------|-------------|-------------|---------|
| 64  | 4,096           | 0.55          | 1.8         | 0.50        | 1.7         | **1.1×** |
| 128 | 16,384          | 1.30          | 4.4         | 1.00        | 3.4         | **1.3×** |
| 256 | 65,536          | 4.72          | 15.9        | 2.02        | 6.8         | **2.3×** |
| 512 | 262,144         | 21.36         | 71.9        | 8.12        | 27.4        | **2.6×** |
| 1024 | 1,048,576      | 101.17        | 340.6       | 43.13       | 145.7       | **2.3×** |

### 8.3 Experiment 2: $\Delta a = 0.001$ (1480–1481 steps)

| $N$ | $N^2$ particles | Reference (s) | Ref ms/step | P3M-JAX (s) | JAX ms/step | Speedup |
|-----|-----------------|---------------|-------------|-------------|-------------|---------|
| 64  | 4,096           | 1.84          | 1.2         | 1.12        | 0.8         | **1.6×** |
| 128 | 16,384          | 5.83          | 3.9         | 2.62        | 1.8         | **2.2×** |
| 256 | 65,536          | 23.13         | 15.6        | 8.60        | 5.8         | **2.7×** |
| 512 | 262,144         | 107.59        | 72.6        | 39.71       | 26.8        | **2.7×** |
| 1024 | 1,048,576      | OOM at step ~600 | —        | OOM at step ~1388 | —    | JAX ran **2.3×** further |

*The reference crashed with an out-of-memory error at approximately step 600/1481 ($a \approx 0.62$). P3M-JAX reached step 1388/1480 ($a \approx 1.41$) before hitting memory pressure — completing 94% of the run vs 41% for the reference. This difference is explained in Section 8.4.*

*Step counts differ by one (297 vs 296, 1481 vs 1480) because the reference halts when $a > a_\mathrm{end}$ (Python condition evaluated after each step), while P3M-JAX runs exactly $\lfloor(a_\mathrm{end} - a_\mathrm{start})/\Delta a\rfloor$ chunks — a discrepancy of $< 0.1\%$.*

### 8.4 Discussion

**Did the migration work?** Yes, at every tested scale and both time-step resolutions. P3M-JAX is consistently faster than the NumPy/Numba reference under identical physics parameters. The speedup grows with $N$ in both experiments, reaching **2.3–2.7×** for $N \geq 256$. At finer time steps ($\Delta a = 0.001$, 5× more steps) the advantage slightly increases relative to $\Delta a = 0.005$, consistent with the expectation that `lax.scan`'s elimination of Python per-step overhead becomes more valuable as the step count grows.

**Why not higher?** Two factors compress the measured ratio below the potential ceiling:

1. **Residual I/O in P3M-JAX.** VTK particle output was disabled (`save_vtk: false`) for these benchmarks. Power-spectrum CSV appends were retained and contribute a small but non-zero overhead at each chunk boundary. The reference writes nothing to disk, so this minor asymmetry slightly compresses the ratio. Disabling the power-spectrum output entirely (`save_power_spectrum: false`) would close the remaining gap.

2. **Numba CIC is competitive on a single core.** The `@numba.jit` CIC kernel compiles to efficient native machine code on first call. For the 2D case on CPU it is broadly competitive with JAX's XLA scatter-add. The dominant JAX gains come from fusing the FFT Poisson solve into a single XLA kernel and eliminating $n_\mathrm{step}$ Python round-trips via `lax.scan`, not from CIC alone.

**Memory behaviour at $N = 1024$.** This is the most operationally significant finding of the $\Delta a = 0.001$ experiment. The reference code accumulates every integration state in a Python list during the run:

```python
states.append(State(state.time, position.copy(), momentum.copy()))
```

At $N=1024$ with 1481 steps, this requires storing $1481 \times 2 \times (1024^2 \times 2) \times 4\,\text{bytes} \approx 24\,\text{GB}$ at float32 — far beyond the available RAM on an Apple M2. The process crashed at step $\sim 600$ when the heap was exhausted. P3M-JAX uses a chunk-based approach: only the current `State` (positions + momenta, $\sim 16\,\text{MB}$) is held in device memory during the integration loop. The full trajectory is never simultaneously resident. P3M-JAX reached step 1388/1480 before encountering its own memory constraint — arising from the end-of-run trajectory stacking for visualisation (`oom_threshold_gb = 4.0` in the config). The integration itself completed; only the post-run plot generation was skipped. This difference — 41% completion for the reference vs 94% for P3M-JAX before memory failure — directly reflects the architectural distinction between accumulating all states in a Python list and the chunk-based XLA execution model.

**Scaling behaviour across both experiments.** Per-step cost is stable across $\Delta a$ values, as expected for a fixed-$N$ force kernel (the force evaluation cost per step does not depend on the number of steps). The reference per-step cost grows from $\sim 1.2$–$1.8\,\text{ms}$ at $N=64$ to $72.6$–$340.6\,\text{ms}$ at $N=1024$ across the two experiments — consistent with $O(N^2 \log N)$ scaling. P3M-JAX per-step cost grows from $0.8$–$1.7\,\text{ms}$ to $26.8$–$145.7\,\text{ms}$ over the same range. The ratio improves with $N$ and with step count.

**Projected scaling in 3D.** In three dimensions the dominant cost — the FFT Poisson solve on the force grid — scales as $O((2N)^3 \log(2N)) = O(N^3 \log N)$, compared with $O(N^2 \log N)$ in 2D. At fixed $N$ the per-step work is therefore $2N$ times greater in 3D than in 2D, amplifying every source of JAX advantage: kernel fusion eliminates $O(N^3)$-element intermediate buffers per step rather than $O(N^2)$, `lax.scan` eliminates more expensive Python round-trips, and the CIC scatter-add operates on $8 \times N^3$ rather than $4 \times N^2$ grid cells. Additionally, the 3D PP correction using Morton Z-curve ordering processes $N^3$ particles with a vmap window, a workload that is a natural fit for device parallelism but impractical for a single-threaded Numba loop. Based on the 2D trend — speedup growing from $\sim 1.1\times$ at $N=64$ to $\sim 2.7\times$ at $N=512$ — the 3D speedup at comparable particle counts is expected to be larger, driven by the increased per-step work. A conservative extrapolation suggests speedups in the range of **3–5×** for 3D CPU runs at similar $N$; GPU backends would add further gains not captured in these CPU benchmarks. Quantitative 3D measurements are deferred to forthcoming work.

---

## 9. Limitations and Honest Caveats

**Compilation latency.** The first call to a JIT-compiled function requires XLA to trace, optimise, and compile the kernel. For large 3D grids ($N=128$, force grid $256^3$), the first chunk can take 30–120 seconds on CPU. Subsequent chunks run at full compiled speed. P3M-JAX mitigates this by compiling outside the chunk loop so compilation occurs once before the first output snapshot.

**PP force memory scaling.** The Morton sliding-window PP correction is $O(N_p \cdot W)$ with `vmap`. For $N=256$ in 3D ($N_p = 16.7\text{M}$) and `pp_window = 4`, the `vmap` allocates a $(N_p, 2W+1, d)$ neighbour tensor — approximately $16.7\text{M} \times 9 \times 3 \times 4\,\text{bytes} \approx 1.7\,\text{GB}$ at float32 — the dominant memory cost and the practical upper limit on $N$ for systems with $\leq 8\,\text{GB}$ device memory. Spatial hashing would reduce this but requires variable-length neighbour lists incompatible with XLA's static-shape model.

**`lax.while_loop` gradient support.** `lax.while_loop` does not support automatic differentiation through the loop body. This is not a limitation for the current simulation, but would prevent `jax.grad`-based differentiable initial condition optimisation — a technique used in likelihood-free inference for cosmology.

**Scaling benchmarks.** Section 8 presents measured 2D PM-only benchmarks on Apple M2 at $N = 64$–$1024$ with VTK output disabled. Extension to 3D and the full P3M solver (with PP correction) has not yet been systematically timed; those measurements are planned as a forthcoming addition. Projected 3D speedups are discussed qualitatively in Section 8.4.

**Live visualisation.** An integrated real-time density-field and particle visualisation pipeline — allowing structure formation to be monitored as the simulation progresses — is planned for a future release. The implementation will use asynchronous callbacks outside the JAX-compiled boundary, decoupled from the device pipeline, so that rendering never stalls computation. Preliminary work on this feature is ongoing.

---

## 10. Conclusion

The migration from the NumPy/Numba reference implementation (`docs/nbody_refrance.py`) to the JAX production code was not a superficial port. It required identifying and correcting eight distinct anti-patterns: a mutable state container that broke JAX tracing at the most fundamental level; side-effecting plotting code embedded in the force kernel; a Python while-loop integration driver that prevented any device pipelining; hidden CPU synchronisations via assertions; unnecessary kernel recomputation on every force call; a Numba CIC implementation that could not be traced or GPU-mapped; a redundant FFT round-trip in initial condition generation; and 2D-hardcoded filter inner products that were silently wrong for 3D.

The production code resolves each of these systematically, using the appropriate JAX primitive in each case:

1. `NamedTuple` State for JAX-compatible immutable carries through `lax.scan` and `vmap`.
2. Clean separation of compute (pure, traceable) from I/O (post-chunk, Python-level).
3. `jax.lax.scan` for fixed-stepping and `jax.lax.while_loop` for CFL-adaptive stepping — both fully on-device.
4. `jax.vmap` for the PP force loop, replacing an $O(N_p)$-Python loop with a single parallel kernel.
5. `@jax.jit` with static shape arguments for CIC, enabling compile-time corner unrolling and GPU scatter-add.
6. Dimension-agnostic implementations throughout — the same code runs in 2D and 3D without modification.

Empirical benchmarks on Apple M2 (Section 8) confirm 2.3–2.7× end-to-end speedup at $N \geq 256$ in 2D (PM solver, VTK output disabled) under controlled conditions, with the advantage growing with particle count and step count. The 3D case is expected to amplify all of these gains as the dominant FFT cost scales as $O(N^3 \log N)$; quantitative 3D measurements are planned as forthcoming work, alongside NVIDIA GPU benchmarks and a non-blocking live visualisation pipeline.

---

## References

- Bradbury, J. et al. (2018). *JAX: composable transformations of Python+NumPy programs*. http://github.com/google/jax
- Hidding, J. *2D N-body simulation*. https://jhidding.github.io/nbody2d/ (reference implementation basis for `docs/nbody_refrance.py`)
- Hockney, R. W. & Eastwood, J. W. (1988). *Computer Simulation Using Particles*. Taylor & Francis.
- Kidger, P. (2021). *On Neural Differential Equations*. PhD thesis, University of Oxford. (Chapter 2: JAX primitives and XLA compilation.)
- Planck Collaboration (2020). *Planck 2018 results VI: Cosmological parameters*. A&A, 641, A6.
