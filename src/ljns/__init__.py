"""
ljns
====

Lennard-Jones bulk-fluid T-rho phase diagram via NVT Nested Sampling.

Top-level public API
--------------------
``SimConfig``
    Full simulation + analysis configuration (loaded from JSON).
``MAX_CORES``
    Hard upper bound on the number of worker processes (64 on this host).
"""

from .config import SimConfig, MAX_CORES, tiny_test_cfg, configure_logging

__version__ = "1.0.0"

__all__ = [
    "SimConfig",
    "MAX_CORES",
    "tiny_test_cfg",
    "configure_logging",
    "__version__",
]
