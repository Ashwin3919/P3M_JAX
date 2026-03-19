"""Shared pytest fixtures for P3M-JAX test suite."""
import sys
from pathlib import Path

import pytest
import jax


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def enable_x64():
    """Enable 64-bit floats for the entire test session.

    JAX disables float64 by default. Setting this once at session scope
    prevents per-test calls to jax.config.update, which would otherwise
    mutate global state mid-run and produce non-deterministic behaviour
    under parallel test runners (pytest-xdist).
    """
    jax.config.update("jax_enable_x64", True)
