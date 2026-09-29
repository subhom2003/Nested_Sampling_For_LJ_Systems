"""
NVT Nested Sampling driver for one density.

The :func:`run_ns` routine below is the workhorse of this package: at a fixed
bulk density it performs ``n_iter`` nested-sampling iterations, producing a
sequence of dead points (energy, virial, log-weight). Downstream analysis
in :mod:`ljns.thermodynamics` converts these dead points into thermodynamic
grids (``Cv``, ``S``, ``F``, ``P``, ``Z``) at any temperature.

Weight convention
-----------------
At iteration ``k`` the live-point configuration-space volume has been
compressed by a factor ``alpha = K / (K + 1)``, so the dead-point log-weight is

    log w_k = k * log(alpha) + log(1 / (K + 1)).

After ``n_iter`` iterations the surviving ``K`` live points are recorded
with log-weight ``n_iter * log(alpha) - log(K)`` (a uniform portion of the
remaining volume).

Dead-point I/O
--------------
Dead points are written to a NumPy compressed ``.npz`` archive storing six
1-D arrays (iteration, energy, virial, log_weight, rho, n_atoms). Re-analysis
at different temperature ranges only requires these archives: no NS
recomputation is needed.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import SimConfig
from .potential import energy_and_virial, tail_virial_volume_corrected
from .walker import LivePoint, make_live_point, walk

logger = logging.getLogger(__name__)


@dataclass
class DeadPoint:
    """One NS dead-point snapshot.

    Attributes
    ----------
    iteration : int
        NS iteration index (``0..n_iter-1`` for free dead points;
        ``n_iter`` for the final K live-point snapshots).
    energy : float
        Total configurational energy (including the energy tail correction).
    virial : float
        Total virial ``W`` including the volume-weighted virial tail
        correction (so pressure follows as ``P = (N T + W / 3) / V``).
    log_weight : float
        Logarithm of the dead-point configuration-space weight.
    rho : float
        Density at which the point was sampled.
    n_atoms : int
        Number of atoms in the box.
    """
    iteration: int
    energy: float
    virial: float
    log_weight: float
    rho: float
    n_atoms: int


def _dead_log_weight(k: int, la: float, l1a: float) -> float:
    """Compute the dead-point log weight at iteration ``k``."""
    return k * la + l1a


def run_ns(cfg: SimConfig, rho: float,
           rng: np.random.Generator) -> Tuple[List[DeadPoint], Dict]:
    """Run nested sampling at a single density.

    Parameters
    ----------
    cfg : SimConfig
        Configuration (physics, NS parameters, IO, etc.).
    rho : float
        Number density at which to run.
    rng : numpy.random.Generator
        Pre-spawned RNG; per-density spawning is handled upstream so the
        result is reproducible and schedule-independent.

    Returns
    -------
    dead : list of DeadPoint
        ``n_iter + K`` dead points (free + final live).
    meta : dict
        Dictionary with ``rho``, ``n_dead``, ``elapsed_s``, etc.
    """
    K = cfg.n_walkers
    n_it = cfg.n_iter
    la = np.log(K / (K + 1))
    l1a = np.log(1 / (K + 1))
    rc = cfg.r_cut_eff(rho)
    N = cfg.n_atoms

    L = cfg.box_length(rho)
    logger.info("Init %d live pts  rho=%.4f  L=%.3f  rc=%.3f ...",
                K, rho, L, rc)
    t0 = time.time()

    # Build the initial live-point population, all at this density.
    pool: List[LivePoint] = [make_live_point(cfg, rho, rng) for _ in range(K)]
    energies = [lp.energy for lp in pool]
    logger.info("Init done in %.1fs  E in [%.2f, %.2f]",
                time.time() - t0, min(energies), max(energies))

    dead: List[DeadPoint] = []
    # Print progress ~8 times during the run; only at INFO or below.
    stride = max(1, n_it // 8)
    for k in range(n_it):
        # Identify the worst (highest-energy) live point and remove it.
        Ev = np.array([lp.energy for lp in pool], dtype=float)
        di = int(np.argmax(Ev))
        E_dead = float(Ev[di])

        # Compute the virial of the about-to-be-discarded live point and the
        # volume-weighted virial tail correction. The total reciprocal-space
        # interaction has no tail (it's a precomputed number); only the
        # missing-beyond-cutoff portion is added via direct multiplication.
        _, W_trunc = energy_and_virial(
            pool[di].positions, pool[di].box_length, rc, cfg)
        W_tail_V = tail_virial_volume_corrected(
            N, rho, pool[di].box_length, rc, cfg)

        dead.append(DeadPoint(
            iteration=k, energy=E_dead, virial=W_trunc + W_tail_V,
            log_weight=_dead_log_weight(k, la, l1a), rho=rho, n_atoms=N,
        ))

        # Replace the dead live point with a clone of a random survivor.
        survivors = [j for j in range(K) if j != di]
        donor = pool[int(rng.choice(survivors))]
        clone = walk(donor, E_dead, cfg, rng)
        pool[di] = clone

        if k % stride == 0:
            logger.debug("k=%6d  E_dead=%10.4f  E_min=%10.4f  t=%.0fs",
                         k, E_dead, float(np.min(Ev)), time.time() - t0)

    # Record the surviving K live points; they occupy the remaining volume,
    # equally split between them.
    lXf = n_it * la
    final_log_weight = lXf - np.log(K)
    for lp in pool:
        _, W_trunc = energy_and_virial(lp.positions, lp.box_length, rc, cfg)
        W_tail_V = tail_virial_volume_corrected(
            N, rho, lp.box_length, rc, cfg)
        dead.append(DeadPoint(
            iteration=n_it, energy=lp.energy, virial=W_trunc + W_tail_V,
            log_weight=final_log_weight, rho=rho, n_atoms=N,
        ))

    elapsed = time.time() - t0
    logger.info("Done rho=%.4f  n_dead=%d  E in [%.3f, %.3f]  t=%.1fs",
                rho, len(dead),
                min(d.energy for d in dead), dead[0].energy, elapsed)
    return dead, {
        "rho": rho,
        "n_dead": len(dead),
        "dead_log_sigma_alpha": la,
        "dead_log_const": l1a,
        "final_log_weight": final_log_weight,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Dead-point I/O
# ---------------------------------------------------------------------------
def save_dead_points(dead: List[DeadPoint], path: str) -> None:
    """Save a list of dead points as a compressed ``.npz`` archive."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        iteration=np.array([d.iteration for d in dead], dtype=np.int32),
        energy=np.array([d.energy for d in dead], dtype=np.float64),
        virial=np.array([d.virial for d in dead], dtype=np.float64),
        log_weight=np.array([d.log_weight for d in dead], dtype=np.float64),
        rho=np.array([d.rho for d in dead], dtype=np.float64),
        n_atoms=np.array([d.n_atoms for d in dead], dtype=np.int32),
    )


def load_dead_points(path: str) -> List[DeadPoint]:
    """Read back a list of dead points produced by :func:`save_dead_points`."""
    data = np.load(path, allow_pickle=False)
    n = len(data["iteration"])
    return [DeadPoint(
        iteration=int(data["iteration"][i]),
        energy=float(data["energy"][i]),
        virial=float(data["virial"][i]),
        log_weight=float(data["log_weight"][i]),
        rho=float(data["rho"][i]),
        n_atoms=int(data["n_atoms"][i]),
    ) for i in range(n)]


__all__ = ["DeadPoint", "run_ns", "save_dead_points", "load_dead_points"]
