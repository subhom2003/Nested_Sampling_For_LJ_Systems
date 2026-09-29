"""
NVT Metropolis Monte Carlo walker for nested sampling.

The walker drives live points through configuration space constrained by a
**hard wall** at the iteration's energy ceiling ``E_ceil``. At each NS
iteration the surviving live points must all carry ``E < E_ceil``; the walker
returns a sample drawn from this truncated canonical distribution.

Contents
--------
- :class:`LivePoint` -- a runnable bundle of positions, energy, box
  parameters, and density.
- :func:`make_live_point` -- construct and equilibrate a new live point in
  the prior distribution (uniform-in-the-box at a fixed density).
- :func:`walk` -- perform ``n_mc_steps * N`` Metropolis moves constrained by
  ``E < E_ceil``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SimConfig
from .potential import (
    total_energy,
    single_atom_energy,
    tail_corrections,
    init_positions,
)

logger = logging.getLogger(__name__)


@dataclass
class LivePoint:
    """Snapshot of a live point in nested sampling.

    The walker clones this structure before each mutation, so a deep copy
    of the ``positions`` array is essential (otherwise consecutive NS
    iterations would corrupt shared state). See :meth:`copy`.

    Attributes
    ----------
    positions : numpy.ndarray
        ``(N, 3)`` coordinates inside the primary cell.
    energy : float
        Total configurational energy, including tail correction.
    box_length : float
        Cubic-box edge ``L``.
    rho : float
        Number density.
    rc : float
        Effective cutoff (already minimum-image safe).
    """

    positions: np.ndarray
    energy: float
    box_length: float
    rho: float
    rc: float

    def copy(self) -> "LivePoint":
        """Return an independent clone whose ``positions`` array is a deep copy."""
        return LivePoint(
            positions=self.positions.copy(),
            energy=self.energy,
            box_length=self.box_length,
            rho=self.rho,
            rc=self.rc,
        )


def _metropolis_step(i: int, lp: LivePoint, cfg: SimConfig,
                     rng: np.random.Generator,
                     E_ceil: float = np.inf) -> bool:
    """Perform a single one-atom Metropolis displacement in place.

    The energy ``U`` is updated incrementally by the per-atom difference
    ``dU = U_new_i - U_old_i``. Two acceptance rules apply:

    1. With an energy ceiling (``E_ceil`` finite): reject the move outright if
       the proposed energy exceeds ``E_ceil`` (hard-wall constraint).
    2. Otherwise, the standard Metropolis criterion for uphill moves:
       accept with probability ``min(1, exp(-dU))``.

    Returns
    -------
    bool
        ``True`` if the move was accepted.
    """
    L = lp.box_length
    rc = lp.rc
    step = cfg.atom_step_size
    positions = lp.positions

    saved = positions[i].copy()
    old = single_atom_energy(i, positions, L, rc, cfg)
    positions[i] = (positions[i] + step * (rng.random(3) - 0.5)) % L
    new = single_atom_energy(i, positions, L, rc, cfg)
    dU = new - old

    proposed_energy = lp.energy + dU

    if np.isfinite(E_ceil) and proposed_energy >= E_ceil:
        # Hard-wall: move violates the NS ceiling, restore.
        positions[i] = saved
        return False

    if dU < 0 or rng.random() < np.exp(-dU):
        lp.energy = proposed_energy
        return True

    positions[i] = saved
    return False


def make_live_point(cfg: SimConfig, rho: float,
                    rng: np.random.Generator) -> LivePoint:
    """Build a new live-point, equilibrated in the prior distribution.

    The new live point's positions start on a lattice (via
    :func:`ljns.potential.init_positions`); a short unconstrained NVT MC burn
    of ``5 N`` trial moves relaxes it into the prior. The returned energy
    *includes* the uniform-fluid tail correction so that all downstream code
    need not worry about it.
    """
    L = cfg.box_length(rho)
    rc = cfg.r_cut_eff(rho)
    N = cfg.n_atoms
    pos = init_positions(N, L, rng)
    U_tail, _ = tail_corrections(N, rho, rc, cfg)
    U_init = total_energy(pos, L, rc, cfg) + U_tail
    lp = LivePoint(positions=pos, energy=U_init, box_length=L, rho=rho, rc=rc)

    # Unconstrained burn-in: relax from the crystal toward a fluid prior.
    for _ in range(5 * N):
        i = int(rng.integers(N))
        _metropolis_step(i, lp, cfg, rng, E_ceil=np.inf)

    # Final recompute to correct accumulated single-atom drift.
    U_tail, _ = tail_corrections(N, rho, rc, cfg)
    lp.energy = total_energy(lp.positions, L, rc, cfg) + U_tail
    return lp


def walk(lp: LivePoint, E_ceil: float, cfg: SimConfig,
         rng: np.random.Generator) -> LivePoint:
    """Perform ``cfg.n_mc_steps * cfg.n_atoms`` constrained NVT moves.

    The live point is **not** mutated: a working copy is used for the whole
    walk, leaving the survivor available for parallel/cloned walks.

    Parameters
    ----------
    lp : LivePoint
        The survivor to be evolved (cloned internally; not modified).
    E_ceil : float
        NS hard-wall; the walker rejects any move with ``E_new >= E_ceil``.
        A small safety margin is applied if the incoming live point already
        exceeds the ceiling.
    cfg : SimConfig
        Configuration.
    rng : numpy.random.Generator
        RNG, ideally specific to this branch of the NS tree so that
        results are schedule-independent (handled in parallel.py).

    Returns
    -------
    LivePoint
        A new live point whose ``energy < E_ceil`` (under normal operation).
    """
    clone = lp.copy()

    # Defensive: an incoming survivor should always satisfy E < E_ceil,
    # but numerics / off-by-eps in argmax could violate it by ~machine-eps.
    if clone.energy >= E_ceil:
        clone.energy = E_ceil - abs(E_ceil) * 1e-10 - 1e-14

    N = cfg.n_atoms
    n_steps = cfg.n_mc_steps * N
    for _ in range(n_steps):
        i = int(rng.integers(N))
        _metropolis_step(i, clone, cfg, rng, E_ceil=E_ceil)

    return clone


__all__ = ["LivePoint", "make_live_point", "walk"]
