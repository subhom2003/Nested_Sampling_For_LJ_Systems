"""
Central configuration for the LJ nested-sampling pipeline.

Reduces operational parameters, physics constants, and parallel-execution policy
into a single, validated ``SimConfig`` dataclass. The configuration is sourced
entirely from a JSON file (created and curated under ``config/``); no
information lives implicitly in CLI defaults.

Reduced LJ units
----------------
Throughout the package: ``length = sigma``, ``energy = epsilon``,
``T = epsilon / k_B``, ``P = epsilon / sigma^3``. In other words ``k_B = 1``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Hard ceiling on the number of worker processes. On the development host this
# project's cores are bounded at 64 even though the machine reports ``nproc``
# == 128 (the other half is reserved). Any request above this limit is
# silently clamped to this value, with a warning.
MAX_CORES: int = 64


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"configuration value {name!r} must be an integer, got {value!r}") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"configuration value {name!r} must be a float, got {value!r}") from exc


@dataclass
class SimConfig:
    """Full configuration of one nested-sampling campaign.

    The dataclass groups parameters under three logical headings (``physics``,
    ``ns``, ``scan``) plus an IO / parallel block. A single instance drives a
    whole density sweep. JSON loading and validation is performed by
    :meth:`from_json`; serialisation by :meth:`to_dict` / :meth:`to_json_file`.

    Attributes
    ----------
    n_atoms : int
        Number of LJ particles in the periodic box. Should equal ``4 m^3`` to
        build a clean FCC lattice (e.g. 32, 108, 256). Other values fall back
        to a simple-cubic lattice with a warning.
    epsilon, sigma : float
        LJ well depth and core diameter. In reduced units both equal 1.
    r_cut : float
        Potential truncation radius. The literature LJ critical / triple
        points tabulated in the README are tabulated for ``r_cut = 2.5``.
    n_walkers : int
        Number of NS live points (``K``). The NS error in ``ln Z`` scales as
        ``K^{-1/2}``; choose ``K`` accordingly.
    n_iter : int
        Number of NS iterations per density.
    n_mc_steps : int
        Number of MC sweeps per NS walker invocation.
    atom_step_size : float
        Maximum per-atom MC displacement, in units of ``sigma``.
    densities : list of float
        Density grid (in ``sigma^{-3}``) at which independent NS runs are
        performed.
    T_min, T_max, n_T : float, float, int
        Analysis temperature grid: ``np.linspace(T_min, T_max, n_T)``.
        Re-analysis at a different grid does not require re-running NS.
    cv_smooth_window : int
        Width of the Gaussian smoothing kernel used by the Cv-peak locator.
        Set ``0`` or ``1`` to disable smoothing.
    n_workers : int
        Number of worker processes. Silently clamped to ``MAX_CORES`` and to
        ``os.cpu_count()``.
    backend : str
        ``joblib`` backend (``"loky"`` is the safe default).
    output_dir : str
        Directory under which dead-point, thermodynamic, and PNG artefacts
        are written. Created on demand.
    save_dead_points, save_thermo, make_plots, per_density_plots : bool
        IO switches.
    seed : int
        Master RNG seed. Each density worker derives its own ``Generator``
        from a child ``SeedSequence`` of this seed, so that the result is
        reproducible and independent of the schedule of workers.
    verbosity : str
        Logging level name. ``"info"`` by default.
    """

    # Physics ----------------------------------------------------------------
    n_atoms: int = 32
    epsilon: float = 1.0
    sigma: float = 1.0
    r_cut: float = 2.5

    # NS ---------------------------------------------------------------------
    n_walkers: int = 200
    n_iter: int = 8000
    n_mc_steps: int = 20
    atom_step_size: float = 0.1

    # Scan -------------------------------------------------------------------
    densities: List[float] = field(
        default_factory=lambda: [
            0.010, 0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250,
            0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650,
            0.700, 0.750, 0.780, 0.810, 0.840, 0.870, 0.890, 0.900,
        ]
    )
    T_min: float = 0.30
    T_max: float = 2.00
    n_T: int = 500
    cv_smooth_window: int = 9

    # Parallel ---------------------------------------------------------------
    n_workers: int = 64
    backend: str = "loky"

    # IO ---------------------------------------------------------------------
    output_dir: str = "runs/default"
    save_dead_points: bool = True
    save_thermo: bool = True
    make_plots: bool = True
    per_density_plots: bool = True

    # Misc -------------------------------------------------------------------
    seed: int = 42
    verbosity: str = "info"

    # ---------------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------------
    @property
    def T_array(self) -> np.ndarray:
        """Linear analysis temperature grid (reduced units)."""
        return np.linspace(self.T_min, self.T_max, self.n_T)

    def box_length(self, rho: float) -> float:
        """Cubic-box edge ``L`` from ``rho`` via ``L = (N / rho)^{1/3}``."""
        return float((self.n_atoms / rho) ** (1.0 / 3.0))

    def r_cut_eff(self, rho: float) -> float:
        """Effective cutoff, respecting the minimum-image convention.

        For a cubic box the convention requires ``L / 2 > r_cut``; the
        effective cutoff is reduced when the user-supplied ``r_cut`` would
        otherwise violate this.
        """
        return float(min(self.r_cut, 0.5 * self.box_length(rho) - 0.01))

    def ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def validate(self) -> None:
        """Validate physics, scan, and parallel settings; clamp cores.

        Performs in-place side effects (e.g. clamping ``n_workers``) and emits
        warnings on suspicious-but-recoverable inputs; raises ``ValueError``
        on hard errors (e.g. negative density).
        """
        if self.n_atoms <= 0:
            raise ValueError("n_atoms must be > 0")
        if self.n_walkers < 2:
            raise ValueError("n_walkers must be >= 2")
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if self.n_mc_steps < 1:
            raise ValueError("n_mc_steps must be >= 1")
        if self.r_cut <= 0:
            raise ValueError("r_cut must be > 0")
        if not self.densities:
            raise ValueError("densities list is empty")
        if any(r <= 0 for r in self.densities):
            raise ValueError("all densities must be > 0")
        if self.T_min <= 0 or self.T_max <= self.T_min:
            raise ValueError("T_min must be > 0 and T_max > T_min")
        if self.n_T < 2:
            raise ValueError("n_T must be >= 2")
        if self.cv_smooth_window < 0:
            raise ValueError("cv_smooth_window must be >= 0")

        # FCC sizes (4 m^3) are strongly recommended for clean lattice init.
        m = round((self.n_atoms / 4) ** (1.0 / 3.0))
        if 4 * m ** 3 != self.n_atoms:
            logger.warning(
                "n_atoms = %d is not a perfect FCC size (4 m^3). "
                "Falling back to a simple-cubic lattice.", self.n_atoms,
            )

        # Per-density minimum-image sanity: report which densities will have
        # an effectively reduced cutoff. Not an error; just informative.
        reduced = [r for r in self.densities
                   if 0.5 * self.box_length(r) <= self.r_cut]
        if reduced:
            logger.info(
                "%d densities have L/2 <= r_cut and will use an "
                "effective reduced cutoff (e.g. rho=%.4f).", len(reduced), reduced[0],
            )

        # Parallel: silently clamp cores to MAX_CORES (and to os.cpu_count()).
        requested = self.n_workers
        cpu = os.cpu_count() or 1
        capped = min(self.n_workers, MAX_CORES, cpu)
        if capped != requested:
            logger.warning(
                "Requested %d worker cores; this run is constrained to %d "
                "(machine cap = %d, os.cpu_count() = %d). Using %d cores.",
                requested, capped, MAX_CORES, cpu, capped,
            )
        self.n_workers = capped

    # ---------------------------------------------------------------------
    # JSON serialisation
    # ---------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "SimConfig":
        """Build a :class:`SimConfig` from a JSON file.

        The JSON layout mirrors the dataclass fields, grouped under
        ``physics``, ``ns``, ``scan``, ``parallel``, ``io``, and ``misc``.
        Only recognised keys are read; extras are ignored. Unspecified keys
        fall back to dataclass defaults.
        """
        with open(path, "r") as handle:
            raw: Dict[str, Any] = json.load(handle)

        # Sentinel instance: provides dataclass defaults for the field-
        # accessors below. Instantiating SimConfig re-runs no validation for
        # default values, so this is cheap.
        default = cls()

        # Allow either flat or grouped JSON; merge into one dict for harmony.
        flat: Dict[str, Any] = {}
        if "physics" in raw:
            flat.update(raw["physics"])
        if "ns" in raw:
            flat.update(raw["ns"])
        if "scan" in raw:
            flat.update(raw["scan"])
        if "parallel" in raw:
            flat.update(raw["parallel"])
        if "io" in raw:
            flat.update(raw["io"])
        for key, value in raw.items():
            if key not in ("physics", "ns", "scan", "parallel", "io"):
                flat.setdefault(key, value)

        cfg = cls(
            n_atoms=_as_int(flat.get("n_atoms", default.n_atoms), "n_atoms"),
            epsilon=_as_float(flat.get("epsilon", default.epsilon), "epsilon"),
            sigma=_as_float(flat.get("sigma", default.sigma), "sigma"),
            r_cut=_as_float(flat.get("r_cut", default.r_cut), "r_cut"),
            n_walkers=_as_int(flat.get("n_walkers", default.n_walkers), "n_walkers"),
            n_iter=_as_int(flat.get("n_iter", default.n_iter), "n_iter"),
            n_mc_steps=_as_int(flat.get("n_mc_steps", default.n_mc_steps), "n_mc_steps"),
            atom_step_size=_as_float(
                flat.get("atom_step_size", default.atom_step_size), "atom_step_size"),
            densities=[float(r) for r in flat.get("densities", default.densities)],
            T_min=_as_float(flat.get("T_min", default.T_min), "T_min"),
            T_max=_as_float(flat.get("T_max", default.T_max), "T_max"),
            n_T=_as_int(flat.get("n_T", default.n_T), "n_T"),
            cv_smooth_window=_as_int(
                flat.get("cv_smooth_window", default.cv_smooth_window),
                "cv_smooth_window"),
            n_workers=_as_int(flat.get("n_workers", default.n_workers), "n_workers"),
            backend=str(flat.get("backend", default.backend)),
            output_dir=str(flat.get("output_dir", default.output_dir)),
            save_dead_points=bool(flat.get("save_dead_points", default.save_dead_points)),
            save_thermo=bool(flat.get("save_thermo", default.save_thermo)),
            make_plots=bool(flat.get("make_plots", default.make_plots)),
            per_density_plots=bool(
                flat.get("per_density_plots", default.per_density_plots)),
            seed=_as_int(flat.get("seed", default.seed), "seed"),
            verbosity=str(flat.get("verbosity", default.verbosity)),
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the (pre-validation) dataclass to a JSON-friendly dict."""
        return asdict(self)

    def to_json_file(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)


# ---------------------------------------------------------------------------
# Convenience: tiny "test" config for unit tests (N=8, K=4, very short).
# ---------------------------------------------------------------------------
def tiny_test_cfg(output_dir: str = "runs/_test") -> SimConfig:
    """Return a deliberately tiny ``SimConfig`` for fast unit tests."""
    cfg = SimConfig(
        n_atoms=8,
        n_walkers=4,
        n_iter=40,
        n_mc_steps=2,
        atom_step_size=0.1,
        densities=[0.3, 0.6],
        T_min=0.30, T_max=2.00, n_T=80,
        cv_smooth_window=0,
        n_workers=1,
        output_dir=output_dir,
        save_dead_points=True,
        save_thermo=True,
        make_plots=False,
        per_density_plots=False,
        seed=42,
        verbosity="warning",
    )
    cfg.validate()
    return cfg


def configure_logging(verbosity: str) -> None:
    """Set the level of the root logger from a verbosity name."""
    level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)-7s %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )


__all__ = ["SimConfig", "MAX_CORES", "tiny_test_cfg", "configure_logging"]
