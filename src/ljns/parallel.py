"""
Parallel execution wrapper for the density-sweep.

The phase-diagram campaign runs one independent NS simulation per density.
This module handles:

- Spawning a fresh :class:`numpy.random.Generator` for each density via
  :class:`numpy.random.SeedSequence` (schedule-independent reproducibility).
- Dispatching to :func:`joblib.Parallel` with the ``loky`` backend.
- Enforcing the host's 64-core ceiling (the machine reports 128 logical
  cores but only 64 are available to this project).

All functions are deliberately small and pure; the orchestration lives in
:mod:`ljns.cli`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed

from .config import MAX_CORES, SimConfig
from .nested_sampling import run_ns, save_dead_points
from .thermodynamics import analyse, find_cv_peak, save_results

logger = logging.getLogger(__name__)


def _run_one_density(
    rho: float,
    cfg: SimConfig,
    seed: np.random.SeedSequence,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run one density: NS + thermodynamic analysis.

    This function is designed to be called inside a :class:`joblib.Parallel`
    worker. It receives a pre-spawned :class:`SeedSequence` so that each
    density gets an independent, deterministic RNG stream regardless of
    scheduling order.

    Parameters
    ----------
    rho : float
        Number density at which to run.
    cfg : SimConfig
        Configuration (passed by value, so it's safe across processes).
    seed : numpy.random.SeedSequence
        Child seed sequence for this density.

    Returns
    -------
    results : dict
        Thermodynamic grids from :func:`ljns.thermodynamics.analyse`.
    phase_info : dict
        ``{"T_peak", "Cv_peak", "FWHM"}`` from the Cv peak locator.
    """
    rng = np.random.default_rng(seed)
    dead, meta = run_ns(cfg, rho, rng)

    # Save dead points if requested (they will be written to the worker's
    # local filesystem; the main process doesn't need them back).
    if cfg.save_dead_points:
        # path constructed in cli, but here we just ensure the directory
        # exists and write a predictable name
        os.makedirs(cfg.output_dir, exist_ok=True)
        dead_path = os.path.join(cfg.output_dir, f"dead_rho{rho:.4f}.npz")
        save_dead_points(dead, dead_path)

    results = analyse(dead, cfg.T_array, cfg)
    T_pk, Cv_pk, FWHM = find_cv_peak(results, sw=cfg.cv_smooth_window)
    phase_info = {"T_peak": T_pk, "Cv_peak": Cv_pk, "FWHM": FWHM}

    # Save per-density thermodynamic results
    if cfg.save_thermo:
        os.makedirs(cfg.output_dir, exist_ok=True)
        res_path = os.path.join(cfg.output_dir, f"thermo_rho{rho:.4f}.npz")
        save_results(results, res_path)

    return results, phase_info


def run_density_sweep(cfg: SimConfig) -> Tuple[Dict[float, Dict], Dict[float, Dict]]:
    """Execute the full density sweep in parallel.

    Parameters
    ----------
    cfg : SimConfig
        Validated configuration. Must have ``n_workers`` already clamped to
        ``MAX_CORES`` and ``os.cpu_count()`` by :meth:`SimConfig.validate`.

    Returns
    -------
    all_results : dict[float, dict]
        Mapping ``rho -> thermodynamic results dict``.
    phase_data : dict[float, dict]
        Mapping ``rho -> {"T_peak", "Cv_peak", "FWHM"}``.
    """
    # Spawn one child SeedSequence per density so that results are
    # independent of the worker scheduling order.
    seeds = np.random.SeedSequence(cfg.seed).spawn(len(cfg.densities))

    logger.info(
        "Dispatching %d densities to %d workers (backend=%s, max_cores=%d)...",
        len(cfg.densities), cfg.n_workers, cfg.backend, MAX_CORES,
    )

    # The closure captures cfg and each (rho, seed) pair.
    # Note: joblib's loky backend pickles cfg by value, so it's safe.
    results = Parallel(
        n_jobs=cfg.n_workers,
        backend=cfg.backend,
        verbose=10 if logger.isEnabledFor(logging.DEBUG) else 0,
    )(
        delayed(_run_one_density)(rho, cfg, seed)
        for rho, seed in zip(cfg.densities, seeds)
    )

    all_results = {rho: res for rho, (res, _) in zip(cfg.densities, results)}
    phase_data = {rho: info for rho, (_, info) in zip(cfg.densities, results)}

    logger.info("All densities completed.")
    return all_results, phase_data


__all__ = ["run_density_sweep", "_run_one_density"]