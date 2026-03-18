#!/usr/bin/env python3
"""
2D N-Body Simulation Clone - Based on Johan Hidding's Implementation
https://jhidding.github.io/nbody2d/

JAX-converted version: NumPy and Numba replaced with JAX equivalents.
Structure, logic, plotting routines, and overall setup remain identical.
"""

from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np  # kept only for plotting / histogram2d fallback
from scipy.integrate import quad
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, Tuple
from functools import partial, reduce
import operator
from numbers import Number
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

# Enable 64-bit precision in JAX (matches NumPy default)
jax.config.update("jax_enable_x64", True)

# ============================================================================
# Constrained Field Theory (CFT) Module
# ============================================================================

def _wave_number(s):
    """Generate wave numbers for FFT"""
    N = s[0]
    i = jnp.indices(s)
    return jnp.where(i > N / 2, i - N, i)

class Box:
    """Simulation box with periodic boundary conditions"""
    def __init__(self, dim, N, L):
        self.N = N
        self.L = L
        self.res = L / N
        self.dim = dim

        self.shape = (self.N,) * dim
        self.size = reduce(lambda x, y: x * y, self.shape)

        self.K = _wave_number(self.shape) * 2 * jnp.pi / self.L
        self.k = jnp.sqrt((self.K ** 2).sum(axis=0))

        self.k_max = N * jnp.pi / L
        self.k_min = 2 * jnp.pi / L

class Filter:
    """Base class for Fourier space filters"""
    def __init__(self, f):
        self.f = f

    def __call__(self, K):
        return self.f(K)

    def __mul__(self, other):
        if isinstance(other, Filter):
            return Filter(lambda K: self.f(K) * other.f(K))
        elif isinstance(other, Number):
            return Filter(lambda K: other * self.f(K))

    def __pow__(self, n):
        return Filter(lambda K: self.f(K) ** n)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Filter(lambda K: self.f(K) / other)

    def __add__(self, other):
        return Filter(lambda K: self.f(K) + other.f(K))

    def __invert__(self):
        return Filter(lambda K: self.f(K).conj())

    def abs(self, B, P):
        return jnp.sqrt(self.cc(B, P, self))

    def cc(self, B, P, other):
        return (~self * other * P)(B.K).sum().real / B.size * B.res ** 2

    def cf(self, B, other):
        return ((~self)(B.K) * other).sum().real / B.size * B.res ** 2

class Identity(Filter):
    def __init__(self):
        Filter.__init__(self, lambda K: 1)

class Zero(Filter):
    def __init__(self):
        Filter.__init__(self, lambda K: 0)

def _K_pow(k, n):
    """Raise |k| to the n-th power safely"""
    return jnp.where(k == 0, 0.0, k ** n)

class Power_law(Filter):
    """Power law filter P(k) ∝ k^n"""
    def __init__(self, n):
        Filter.__init__(self, lambda K: _K_pow((K ** 2).sum(axis=0), n / 2))

def _scale_filter(B, t):
    """Discrete scale space filter"""
    def f(K):
        return reduce(
            lambda x, y: x * y,
            (jnp.exp(t / B.res ** 2 * (jnp.cos(k * B.res) - 1)) for k in K))
    return f

class Scale(Filter):
    def __init__(self, B, sigma):
        Filter.__init__(self, _scale_filter(B, sigma ** 2))

class Cutoff(Filter):
    def __init__(self, B):
        Filter.__init__(self, lambda K: jnp.where((K ** 2).sum(axis=0) <= B.k_max ** 2, 1.0, 0.0))

class Potential(Filter):
    """Gravitational potential filter: -1/k²"""
    def __init__(self):
        Filter.__init__(self, lambda K: -_K_pow((K ** 2).sum(axis=0), -1.0))

def garfield(B, P, T=Identity(), seed=None):
    """Generate Gaussian random field"""
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(0)
    wn = jax.random.normal(key, shape=B.shape)
    f = jnp.fft.ifftn(jnp.fft.fftn(wn) * jnp.sqrt(P(B.K))).real
    return jnp.fft.ifftn(jnp.fft.fftn(f) * T(B.K)).real

# ============================================================================
# Cosmology
# ============================================================================

@dataclass
class Cosmology:
    H0: float
    OmegaM: float
    OmegaL: float

    @property
    def OmegaK(self):
        return 1 - self.OmegaM - self.OmegaL

    @property
    def G(self):
        return 3. / 2 * self.OmegaM * self.H0 ** 2

    def da(self, a):
        return self.H0 * a * jnp.sqrt(
            self.OmegaL +
            self.OmegaM * a ** -3 +
            self.OmegaK * a ** -2)

    def growing_mode(self, a):
        if isinstance(a, jnp.ndarray):
            return jnp.array([self.growing_mode(float(b)) for b in a])
        elif a <= 0.001:
            return a
        else:
            factor = 5. / 2 * self.OmegaM
            return factor * self.da(a) / a * \
                quad(lambda b: float(self.da(b)) ** (-3), 0.00001, a)[0] + 0.00001

# Standard cosmologies
LCDM = Cosmology(68.0, 0.31, 0.69)
EdS = Cosmology(70.0, 1.0, 0.0)

# ============================================================================
# Mass Deposition and Interpolation
# ============================================================================

def md_cic(B: Box, X: jnp.ndarray) -> jnp.ndarray:
    """Cloud-in-cell mass deposition - JAX version (replaces NumPy histogram2d)"""
    f = X - jnp.floor(X)
    N = B.shape[0]

    def deposit(x, y, wx, wy):
        xi = (jnp.floor(x) % N).astype(jnp.int32)
        yi = (jnp.floor(y) % N).astype(jnp.int32)
        flat_idx = xi * N + yi
        return jnp.zeros(N * N, dtype=jnp.float64).at[flat_idx].add(wx * wy).reshape(B.shape)

    rho = (
        deposit(X[:, 0],       X[:, 1],       1 - f[:, 0], 1 - f[:, 1]) +
        deposit(X[:, 0] + 1,   X[:, 1],           f[:, 0], 1 - f[:, 1]) +
        deposit(X[:, 0],       X[:, 1] + 1,   1 - f[:, 0],     f[:, 1]) +
        deposit(X[:, 0] + 1,   X[:, 1] + 1,       f[:, 0],     f[:, 1])
    )
    return rho

# JAX-JIT compiled version of CIC (replaces Numba md_cic_2d)
# shape must be a static (compile-time) argument so JAX can use it to
# construct the output array inside the JIT body.
@partial(jax.jit, static_argnums=(0,))
def md_cic_2d(shape: Tuple[int, int], pos: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-based cloud-in-cell mass deposition.
    shape is static so JAX can build the zero array at trace time.
    Returns the deposited density array.
    """
    N0, N1 = shape
    idx0 = jnp.floor(pos[:, 0]).astype(jnp.int32)
    idx1 = jnp.floor(pos[:, 1]).astype(jnp.int32)
    f0 = pos[:, 0] - idx0
    f1 = pos[:, 1] - idx1

    def flat(i, j):
        return (i % N0) * N1 + (j % N1)

    weights_00 = (1 - f0) * (1 - f1)
    weights_10 = f0 * (1 - f1)
    weights_01 = (1 - f0) * f1
    weights_11 = f0 * f1

    flat_tgt = jnp.zeros(N0 * N1, dtype=jnp.float64)
    flat_tgt = flat_tgt.at[flat(idx0,     idx1    )].add(weights_00)
    flat_tgt = flat_tgt.at[flat(idx0 + 1, idx1    )].add(weights_10)
    flat_tgt = flat_tgt.at[flat(idx0,     idx1 + 1)].add(weights_01)
    flat_tgt = flat_tgt.at[flat(idx0 + 1, idx1 + 1)].add(weights_11)

    return flat_tgt.reshape(shape)

class Interp2D:
    """Bilinear interpolation"""
    def __init__(self, data):
        self.data = data
        self.shape = data.shape  # keep as plain Python tuple for indexing

    def __call__(self, x):
        N0, N1 = self.shape
        X1 = jnp.floor(x).astype(jnp.int32) % jnp.array([N0, N1])
        X2 = jnp.ceil(x).astype(jnp.int32) % jnp.array([N0, N1])
        xm = x % 1.0
        xn = 1.0 - xm

        f1 = self.data[X1[:, 0], X1[:, 1]]
        f2 = self.data[X2[:, 0], X1[:, 1]]
        f3 = self.data[X1[:, 0], X2[:, 1]]
        f4 = self.data[X2[:, 0], X2[:, 1]]

        return (f1 * xn[:, 0] * xn[:, 1] +
                f2 * xm[:, 0] * xn[:, 1] +
                f3 * xn[:, 0] * xm[:, 1] +
                f4 * xm[:, 0] * xm[:, 1])

def gradient_2nd_order(F, i):
    """Second-order finite difference gradient"""
    return (1. / 12 * jnp.roll(F, 2, axis=i) - 2. / 3 * jnp.roll(F, 1, axis=i) +
            2. / 3 * jnp.roll(F, -1, axis=i) - 1. / 12 * jnp.roll(F, -2, axis=i))

# ============================================================================
# Integrator Framework
# ============================================================================

class VectorABC(ABC):
    @abstractmethod
    def __add__(self, other): raise NotImplementedError
    @abstractmethod
    def __rmul__(self, other): raise NotImplementedError

VectorABC.register(jnp.ndarray)
Vector = TypeVar("Vector", bound=VectorABC)

@dataclass
class State(Generic[Vector]):
    time: float
    position: Vector
    momentum: Vector

    def kick(self, dt: float, h: 'HamiltonianSystem[Vector]') -> 'State[Vector]':
        self.momentum = self.momentum + dt * h.momentumEquation(self)
        return self

    def drift(self, dt: float, h: 'HamiltonianSystem[Vector]') -> 'State[Vector]':
        self.position = self.position + dt * h.positionEquation(self)
        return self

    def wait(self, dt: float) -> 'State[Vector]':
        self.time += dt
        return self

class HamiltonianSystem(ABC, Generic[Vector]):
    @abstractmethod
    def positionEquation(self, s: State[Vector]) -> Vector: raise NotImplementedError
    @abstractmethod
    def momentumEquation(self, s: State[Vector]) -> Vector: raise NotImplementedError

Solver = Callable[[HamiltonianSystem[Vector], State[Vector]], State[Vector]]
Stepper = Callable[[State[Vector]], State[Vector]]
HaltingCondition = Callable[[State[Vector]], bool]

def leap_frog(dt: float, h: HamiltonianSystem[Vector], s: State[Vector]) -> State[Vector]:
    """Leap-frog integration step"""
    return s.kick(dt, h).wait(dt / 2).drift(dt, h).wait(dt / 2)

def iterate_step(step: Stepper, halt: HaltingCondition, init: State[Vector]) -> State[Vector]:
    """Iterate simulation steps until halting condition"""
    state = init
    states = []
    while not halt(state):
        states.append(State(state.time, jnp.array(state.position), jnp.array(state.momentum)))
        state = step(state)
        # Live plot update every 10 steps
        if state.live_plot and len(states) % 10 == 0:
            clear_output(wait=True)
            display(state.fig)
        if len(states) % 10 == 0:
            print(f"Time step {len(states)}, a = {state.time:.3f}")
    return states

# ============================================================================
# Poisson-Vlasov System
# ============================================================================

class PoissonVlasov(HamiltonianSystem[jnp.ndarray]):
    def __init__(self, box, cosmology, particle_mass, live_plot=False):
        self.box = box
        self.cosmology = cosmology
        self.particle_mass = particle_mass
        self.delta = jnp.zeros(self.box.shape, dtype=jnp.float64)
        self.live_plot = live_plot

        # Initialize live plotting
        if live_plot:
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, box.N)
            self.ax.set_ylim(0, box.N)
            self.ax.set_title('Density Field Evolution')

    def positionEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        return s.momentum / (s.time ** 2 * da)

    def momentumEquation(self, s: State[jnp.ndarray]) -> jnp.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        x_grid = s.position / self.box.res

        # Compute density field using JAX CIC
        self.delta = md_cic_2d(self.box.shape, x_grid)
        self.delta = self.delta * self.particle_mass
        self.delta = self.delta - 1.0

        # Verify mass conservation (CIC conserves mass to floating-point precision)
        assert float(jnp.abs(self.delta.mean())) < 1e-4, "Total mass should be normalized"

        # Live plotting
        if self.live_plot:
            self.ax.clear()
            rho_plot = self.delta + 1.0
            im = self.ax.imshow(jnp.log10(jnp.maximum(rho_plot, 0.1)).T,
                                extent=[0, self.box.N, 0, self.box.N],
                                origin='lower', cmap='hot', vmin=-0.5, vmax=1.0)
            self.ax.set_title(f'Log Density Field (a = {a:.3f})')

        # Solve Poisson equation
        delta_f = jnp.fft.fftn(self.delta)
        kernel = Potential()(self.box.K)
        phi = jnp.fft.ifftn(delta_f * kernel).real * self.cosmology.G / a

        # Compute acceleration
        acc_x = Interp2D(gradient_2nd_order(phi, 0))
        acc_y = Interp2D(gradient_2nd_order(phi, 1))
        acc = jnp.c_[acc_x(x_grid), acc_y(x_grid)] / self.box.res

        return -acc / da

# ============================================================================
# Zeldovich Approximation Initialization
# ============================================================================

def a2r(B, X):
    """Convert array to particles"""
    return X.transpose([1, 2, 0]).reshape([B.N ** 2, 2])

def r2a(B, x):
    """Convert particles to array"""
    return x.reshape([B.N, B.N, 2]).transpose([2, 0, 1])

class Zeldovich:
    def __init__(self, B_mass: Box, B_force: Box, cosmology: Cosmology, phi: jnp.ndarray):
        self.bm = B_mass
        self.bf = B_force
        self.cosmology = cosmology

        # Compute displacement field
        self.u = jnp.array([-gradient_2nd_order(phi, 0),
                             -gradient_2nd_order(phi, 1)]) / self.bm.res

    def state(self, a_init: float) -> State[jnp.ndarray]:
        """Generate initial state using Zeldovich approximation"""
        X = a2r(self.bm, jnp.indices(self.bm.shape) * self.bm.res + a_init * self.u)
        P = a2r(self.bm, a_init * self.u)
        return State(time=a_init, position=X, momentum=P)

    @property
    def particle_mass(self):
        return (self.bf.N / self.bm.N) ** self.bm.dim

# ============================================================================
# Visualization Functions
# (Structure and output identical to original; JAX arrays cast to numpy for plotting)
# ============================================================================

def plot_density_evolution(states, box, times=[0.02, 0.5, 2.0]):
    """Plot density field evolution"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, len(times), figsize=(5 * len(times), 10))
    if len(times) == 1:
        axes = axes.reshape(-1, 1)

    state_indices = []
    for t in times:
        idx = min(range(len(states)), key=lambda i: abs(states[i].time - t))
        state_indices.append(idx)

    for i, idx in enumerate(state_indices):
        state = states[idx]

        x_grid = state.position / box.res
        rho = md_cic_2d(box.shape, x_grid)
        rho = rho + 1.0

        # Convert to numpy for matplotlib
        rho_np = np.array(rho)

        ax_top = axes[0, i]
        log_rho = np.log10(np.maximum(rho_np, 0.1))
        im1 = ax_top.imshow(log_rho.T, extent=[0, box.L, 0, box.L],
                            origin='lower', cmap='hot', vmin=-0.3, vmax=0.8)
        ax_top.set_title(f'a = {state.time:.2f}', color='white', fontsize=14)
        ax_top.set_xlabel('x [Mpc/h]', color='white')
        if i == 0:
            ax_top.set_ylabel('y [Mpc/h]', color='white')
        plt.colorbar(im1, ax=ax_top, shrink=0.8, label='log₁₀(ρ/ρ̄)')

        ax_bottom = axes[1, i]
        im2 = ax_bottom.imshow(rho_np.T, extent=[0, box.L, 0, box.L],
                               origin='lower', cmap='viridis', vmin=0.5, vmax=3.0)
        ax_bottom.set_xlabel('x [Mpc/h]', color='white')
        if i == 0:
            ax_bottom.set_ylabel('y [Mpc/h]', color='white')
        plt.colorbar(im2, ax=ax_bottom, shrink=0.8, label='ρ/ρ̄')

    fig.suptitle('Density Field Evolution', color='white', fontsize=16)
    plt.tight_layout()
    return fig

def plot_particles(states, box, times=[0.02, 0.5, 2.0], n_particles=2000):
    """Plot particle positions"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, len(times), figsize=(6 * len(times), 6))
    if len(times) == 1:
        axes = [axes]

    state_indices = []
    for t in times:
        idx = min(range(len(states)), key=lambda i: abs(states[i].time - t))
        state_indices.append(idx)

    n_total = len(states[0].position)
    # Use numpy for random choice (pure indexing utility, no computation)
    particle_indices = np.random.choice(n_total, size=min(n_particles, n_total), replace=False)

    for i, idx in enumerate(state_indices):
        state = states[idx]
        ax = axes[i]

        pos = np.array(state.position[particle_indices] % box.L)
        ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.8, c='cyan', edgecolors='none')
        ax.set_xlim(0, box.L)
        ax.set_ylim(0, box.L)
        ax.set_title(f'a = {state.time:.2f}', color='white', fontsize=14)
        ax.set_xlabel('x [Mpc/h]', color='white')
        if i == 0:
            ax.set_ylabel('y [Mpc/h]', color='white')
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Particle Evolution', color='white', fontsize=16)
    plt.tight_layout()
    return fig

# ============================================================================
# Main Simulation Function
# ============================================================================

def run_nbody_simulation(N=256, L=50.0, A=10, seed=4, a_start=0.02, a_end=4.0,
                         dt=0.02, power_index=-0.5, live_plot=True):
    """
    Run the N-body simulation following Johan Hidding's methodology

    Parameters:
    - N: Grid resolution
    - L: Box size in Mpc/h
    - A: Amplitude of initial perturbations
    - seed: Random seed
    - a_start: Initial scale factor
    - a_end: Final scale factor
    - dt: Time step
    - power_index: Power spectrum index
    - live_plot: Show live plotting during simulation
    """
    print(f"=== Johan Hidding's N-Body Simulation Clone (JAX) ===")
    print(f"Grid: {N}x{N}, Box size: {L} Mpc/h")
    print(f"Scale factor: {a_start} -> {a_end}")
    print(f"Power spectrum: P(k) ∝ k^{power_index}")
    print(f"Amplitude: {A}, Seed: {seed}")
    print()

    # Create simulation boxes
    B_m = Box(2, N, L)
    force_box = Box(2, N * 2, B_m.L)

    # Generate initial power spectrum and potential
    Power_spectrum = Power_law(power_index) * Scale(B_m, 0.2) * Cutoff(B_m)
    phi = garfield(B_m, Power_spectrum, Potential(), seed) * A

    print("Generated initial conditions using Zeldovich approximation...")

    # Initialize with Zeldovich approximation
    za = Zeldovich(B_m, force_box, EdS, phi)
    state = za.state(a_start)

    # Set up the Hamiltonian system
    system = PoissonVlasov(force_box, EdS, za.particle_mass, live_plot=live_plot)
    state.live_plot = live_plot
    if live_plot:
        state.fig = system.fig

    # Create stepper function
    stepper = partial(leap_frog, dt, system)

    print(f"Starting simulation with {len(state.position)} particles...")
    print()

    # Run simulation
    states = iterate_step(stepper, lambda s: s.time > a_end, state)

    if live_plot:
        plt.close(system.fig)

    print(f"\nSimulation completed! Generated {len(states)} time steps.")
    return states, B_m

# ============================================================================
# Automatic Parameter Scaling System
# ============================================================================

def get_optimal_parameters(N):
    """
    Automatically scale simulation parameters based on resolution N
    Returns optimized parameters for the given grid resolution
    """
    params = {
        'N': N,
        'L': 50.0,
        'seed': 4,
        'a_start': 0.02,
        'power_index': -0.5,
        'live_plot': True
    }

    if N <= 64:
        params.update({'dt': 0.03, 'A': 8, 'a_end': 1.5})
        print("🚀 FAST MODE: Quick test resolution")

    elif N <= 128:
        params.update({'dt': 0.02, 'A': 10, 'a_end': 2.0})
        print("⚖️  MEDIUM MODE: Balanced resolution")

    elif N <= 256:
        params.update({'dt': 0.015, 'A': 12, 'a_end': 1.0})
        print("🔬 HIGH MODE: Detailed resolution")

    elif N <= 512:
        params.update({'dt': 0.01, 'A': 15, 'a_end': 1.0})
        print("💎 VERY HIGH MODE: Production quality")

    else:
        params.update({'dt': 0.005, 'A': 18, 'a_end': 4.0})
        print("🏆 EXTREME MODE: Research grade resolution")

    total_particles = N * N
    estimated_time_per_step = total_particles / 50000
    total_steps = int((params['a_end'] - params['a_start']) / params['dt'])
    estimated_total_time = estimated_time_per_step * total_steps / 60

    print(f"📊 Resolution Stats:")
    print(f"   Grid: {N}×{N} = {total_particles:,} particles")
    print(f"   Time step: {params['dt']}")
    print(f"   Amplitude: {params['A']}")
    print(f"   Evolution: a = {params['a_start']} → {params['a_end']}")
    print(f"   Estimated steps: {total_steps}")
    print(f"   Estimated runtime: ~{estimated_total_time:.1f} minutes")
    print()

    return params

# ============================================================================
# Main execution with automatic scaling
# ============================================================================

if __name__ == "__main__":
    # 🎯 CHANGE THIS VALUE TO SET RESOLUTION:
    RESOLUTION = 256  # ← Change this number: 64, 128, 256, 512, 1024, etc.

    print("=" * 60)
    print("🌌 JOHAN HIDDING'S N-BODY SIMULATION CLONE (JAX) 🌌")
    print("=" * 60)

    # Get automatically scaled parameters
    params = get_optimal_parameters(RESOLUTION)

    # Ask user for confirmation
    print("⚡ Press Enter to start simulation, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Simulation cancelled.")
        exit()

    # Run the simulation with auto-scaled parameters
    print("🚀 Starting simulation...")
    start_time = time.time()

    states, box = run_nbody_simulation(**params)

    end_time = time.time()
    runtime = (end_time - start_time) / 60

    print(f"\n✅ Simulation completed in {runtime:.1f} minutes!")

    # Create summary visualizations
    print("📊 Creating final analysis plots...")

    plt.style.use('default')

    key_times = [params['a_start'], params['a_end'] / 2, params['a_end']]

    print("🎨 Generating density evolution plot...")
    fig1 = plot_density_evolution(states, box, key_times)
    plt.show()

    print("🎨 Generating particle evolution plot...")
    fig2 = plot_particles(states, box, key_times)
    plt.show()

    print("🎉 Clone simulation complete!")
    print("📈 Results should match Johan Hidding's reference implementation.")
    print("=" * 60)