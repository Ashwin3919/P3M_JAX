import json
import os

# Keys that must be present in every config, regardless of solver/timestepping.
_REQUIRED_KEYS = {"N", "L", "A", "a_start", "a_end", "power_index", "seed"}

# Keys required only for fixed time-stepping (the default).
_REQUIRED_FIXED_DT = {"dt"}


def _validate_ranges(config: dict, path: str) -> None:
    """Check that config values are physically sensible.

    Raises ValueError with a clear message on the first violation found.
    """
    def check(cond, msg):
        if not cond:
            raise ValueError(f"Config '{path}': {msg}")

    check(isinstance(config['N'], int) and config['N'] >= 4,
          f"'N' must be an integer ≥ 4, got {config['N']!r}")
    check(config['L'] > 0,
          f"'L' must be positive, got {config['L']}")
    check(config['a_start'] > 0,
          f"'a_start' must be > 0, got {config['a_start']}")
    check(config['a_end'] > config['a_start'],
          f"'a_end' ({config['a_end']}) must be > 'a_start' ({config['a_start']})")
    check(config.get('H0', 70.0) > 0,
          f"'H0' must be positive, got {config.get('H0')}")
    check(config.get('OmegaM', 1.0) >= 0,
          f"'OmegaM' must be ≥ 0, got {config.get('OmegaM')}")
    check(config.get('OmegaL', 0.0) >= 0,
          f"'OmegaL' must be ≥ 0, got {config.get('OmegaL')}")

    if config.get('timestepping', 'fixed') == 'fixed':
        check(config['dt'] > 0,
              f"'dt' must be positive, got {config['dt']}")
        check(config['dt'] < (config['a_end'] - config['a_start']),
              f"'dt' ({config['dt']}) must be smaller than the total time span "
              f"({config['a_end'] - config['a_start']:.4f})")

    if config.get('timestepping') == 'adaptive':
        dt_min = config.get('dt_min', 0.001)
        dt_max = config.get('dt_max', 0.05)
        check(dt_min > 0, f"'dt_min' must be positive, got {dt_min}")
        check(dt_max > dt_min, f"'dt_max' ({dt_max}) must be > 'dt_min' ({dt_min})")
        check(config.get('n_chunks', 50) >= 1,
              f"'n_chunks' must be ≥ 1, got {config.get('n_chunks')}")

    if config.get('solver') == 'p3m':
        check(config.get('pp_window', 2) >= 1,
              f"'pp_window' must be ≥ 1, got {config.get('pp_window')}")
        check(config.get('pp_softening', 0.1) > 0,
              f"'pp_softening' must be positive, got {config.get('pp_softening')}")
        check(config.get('pp_cutoff', 2.5) > 0,
              f"'pp_cutoff' must be positive, got {config.get('pp_cutoff')}")

    dim = config.get('dim', 2)
    check(dim in (2, 3), f"'dim' must be 2 or 3, got {dim}")


def load_config(config_path: str) -> dict:
    """Load a JSON simulation config, validate required keys and value ranges.

    Raises
    ------
    FileNotFoundError  — if the config file does not exist.
    KeyError           — if a required key is absent.
    ValueError         — if any value is out of physical range.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Validate unconditionally-required keys
    missing = _REQUIRED_KEYS - config.keys()
    if missing:
        raise KeyError(
            f"Config '{config_path}' is missing required keys: {sorted(missing)}"
        )

    # Validate time-stepping keys
    if config.get('timestepping', 'fixed') == 'fixed' and 'dt' not in config:
        raise KeyError(
            f"Config '{config_path}': 'dt' is required when timestepping='fixed'."
        )

    # Validate solver string
    solver = config.get('solver', 'pm')
    if solver not in ('pm', 'p3m'):
        raise ValueError(
            f"Config '{config_path}': 'solver' must be 'pm' or 'p3m', got '{solver}'."
        )

    # Validate value ranges
    _validate_ranges(config, config_path)

    config['name'] = os.path.splitext(os.path.basename(config_path))[0]
    return config


def get_results_dir(config_name: str) -> str:
    """Create and return the results directory for the given config name."""
    results_dir = os.path.join('results', config_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
