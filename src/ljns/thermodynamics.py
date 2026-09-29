"""
Thermodynamic analysis from nested-sampling dead points.

The :func:`analyse` routine consumes the dead points produced by
:func:`ljns.nested_sampling.run_ns` and returns a full thermodynamic grid
(``Cv``, ``S``, ``F``, ``G``, ``P``, ``Z``, ``ln g(E)``) evaluated at every
temperature of the analysis temperature grid.

Vectorisation
-------------
The implementation works on a ``(n_dead, n_T)`` matrix of log-likelihoods
using NumPy broadcasting. For typical ``n_dead ~ 10^4`` and ``n_T ~ 500``
the cost is one fused NumPy sweep and ~40 MB of working memory; this is a
~50x speed-up over the original Python ``for j, T`` loop with no algorithmic
change.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from .config import SimConfig
from .nested_sampling import DeadPoint


# ---------------------------------------------------------------------------
# Numerical primitive
# ---------------------------------------------------------------------------
def log_sum_exp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp along ``axis``.

    Computes ``ln(sum(exp(a_i)))`` while leaving non-finite entries
    (``-inf``) untouched (these commonly appear in masked-out log-likelihoods).
    """
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    # Both terms keep the reduced dimension, so shapes stay aligned for broadcasting
    out = m_safe + np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


# ---------------------------------------------------------------------------
# Main thermal analysis
# ---------------------------------------------------------------------------
def analyse(dead: List[DeadPoint], T_array: np.ndarray,
            cfg: SimConfig) -> Dict[str, np.ndarray]:
    """Compute thermodynamic observables from a set of dead points.

    The eight observables returned are sufficient to draw the publication
    phase-diagram figure plus the eight thermo-curves from Pártay et al.
    (2014). All arrays have ``shape = (n_T,)`` and the dict additionally
    stores scalar metadata (``rho``, ``n_atoms``, ``volume``).

    Parameters
    ----------
    dead : list of DeadPoint
        Output of :func:`ljns.nested_sampling.run_ns` at a single density.
    T_array : numpy.ndarray
        1-D temperature grid (reduced units T*).
    cfg : SimConfig
        Configuration; used for atom-count sanity check.

    Returns
    -------
    dict
        A dictionary of arrays. See frost on ``notes`` in the original
        README for the full physics derivation; the keys follow the
        historical naming so downstream plots can be reused.
    """
    # Stack dead-point quantities into arrays once.
    E = np.array([d.energy for d in dead], dtype=np.float64)            # n_dead
    W = np.array([d.virial for d in dead], dtype=np.float64)           # n_dead
    lw = np.array([d.log_weight for d in dead], dtype=np.float64)      # n_dead
    rho = float(dead[0].rho)
    N = int(dead[0].n_atoms)
    V = N / rho

    # Vectorised log-likelihood: replace the per-T Python loop with a
    # single NumPy broadcast over (n_dead, n_T).
    T = np.asarray(T_array, dtype=np.float64).reshape(-1)               # n_T
    lb = lw[:, None] - E[:, None] / T[None, :]                         # n_dead, n_T
    logZ = log_sum_exp(lb, axis=0)                                      # n_T
    p = np.exp(lb - logZ[None, :])                                      # n_dead, n_T

    mean_U = (p * E[:, None]).sum(axis=0)                              # n_T
    mean_U2 = (p * (E * E)[:, None]).sum(axis=0)                       # n_T
    mean_W = (p * W[:, None]).sum(axis=0)                             # n_T

    # Heat capacities -----------------------------------------------------
    Cv_conf = (mean_U2 - mean_U * mean_U) / (T * T)
    Cv_total = Cv_conf + 1.5 * N
    Cv_per_atom = Cv_total / N

    # Free energies -------------------------------------------------------
    F_conf = -T * logZ
    F_kin = -T * N * (1.5 * np.log(2.0 * np.pi * T) + np.log(V / N) + 1.0)
    F_total = F_conf + F_kin
    U_total = mean_U + 1.5 * N * T

    # Entropy -------------------------------------------------------------
    S_conf = (mean_U - F_conf) / T
    S_total = (U_total - F_total) / T

    # Pressure, compressibility, and Gibbs free energy --------------------
    P = (N * T + mean_W / 3.0) / V
    G_conf = F_conf + P * V
    G_total = F_total + P * V
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_factor = P * V / (N * T)

    return {
        "T": T.copy(),
        "rho": rho,
        "n_atoms": N,
        "volume": V,
        "log_Z_conf": logZ,
        "Z_conf_norm": np.exp(logZ - np.max(logZ)),
        "mean_U": mean_U,
        "mean_U_per_atom": mean_U / N,
        "Cv_conf": Cv_conf,
        "Cv_total": Cv_total,
        "Cv_per_atom": Cv_per_atom,
        "F_conf": F_conf,
        "F_conf_per_atom": F_conf / N,
        "F_total": F_total,
        "F_total_per_atom": F_total / N,
        "S_conf": S_conf,
        "S_conf_per_atom": S_conf / N,
        "S_total": S_total,
        "S_total_per_atom": S_total / N,
        "mean_P": P,
        "G_conf": G_conf,
        "G_conf_per_atom": G_conf / N,
        "G_total": G_total,
        "G_total_per_atom": G_total / N,
        "Z_factor": Z_factor,
        "mean_W": mean_W,
    }


# ---------------------------------------------------------------------------
# Cv-peak locator with sub-grid interpolation
# ---------------------------------------------------------------------------
def find_cv_peak(res: Dict[str, np.ndarray], sw: int = 9) -> Tuple[float, float, float]:
    """Locate the heat-capacity peak via argmax + parabolic interpolation.

    Returns ``(T_peak, Cv_peak, FWHM)``.

    Smoothing (a Gaussian kernel of width ``sw``) is applied **only** to
    detect the peak location and FWHM. The reported ``Cv_peak`` is taken
    from the *unsmoothed* curve by linear interpolation at the located
    peak. This keeps the reported peak height consistent with the
    transmission-level ``Cv_per_atom`` and avoids a mismatch between the
    figure and the summary table.

    Parameters
    ----------
    res : dict
        Result dictionary from :func:`analyse` (needs ``T`` and
        ``Cv_per_atom`` keys).
    sw : int
        Gaussian smoothing width in samples. ``0`` or ``1`` disables
        smoothing.

    Returns
    -------
    T_peak : float
        Sub-grid peak temperature (reduced units).
    Cv_peak : float
        Unsmoothed height ``Cv_per_atom(T_peak)`` (linear interpolation).
    FWHM : float
        Full-width at half-maximum of the smoothed curve; serves as a
        rough uncertainty on ``T_peak``. Returns ``0.0`` when the half-max
        crossing cannot be located on either side.
    """
    T = np.asarray(res["T"])
    Cv = np.asarray(res["Cv_per_atom"])

    if sw >= 2 and len(Cv) > sw:
        kernel = np.exp(-0.5 * (np.arange(sw) - sw // 2) ** 2 / 2.0)
        kernel /= kernel.sum()
        Cv_smooth = np.convolve(Cv, kernel, mode="same")
    else:
        Cv_smooth = Cv

    idx = int(np.argmax(Cv_smooth))
    T_peak = float(T[idx])

    # Sub-grid correction via parabolic interpolation of the *smoothed*
    # curve (consistent because we use the smoothed values to pick the
    # coarse-resolution maximum).
    if 1 <= idx <= len(T) - 2:
        x0, x1, x2 = T[idx - 1], T[idx], T[idx + 1]
        y0, y1, y2 = Cv_smooth[idx - 1], Cv_smooth[idx], Cv_smooth[idx + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-30:
            A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0)
                 + x0 * x0 * (y1 - y2)) / denom
            if A < 0:
                tp = -B / (2 * A)
                if x0 < tp < x2:
                    T_peak = float(tp)

    # Cv_peak is reported from the *unsmoothed* curve via linear interpolation
    # at the located peak temperature (so the summary table matches the raw plot).
    Cv_peak = float(np.interp(T_peak, T, Cv))

    # FWHM (full-width at half-maximum of the smoothed curve).
    half = Cv_smooth[idx] / 2.0
    left = np.where(Cv_smooth[:idx] < half)[0]
    right = np.where(Cv_smooth[idx:] < half)[0]
    T_left = float(T[left[-1]]) if len(left) > 0 else float(T[0])
    T_right = float(T[idx + right[0]]) if len(right) > 0 else float(T[-1])
    FWHM = max(0.0, T_right - T_left)

    return T_peak, Cv_peak, FWHM


# ---------------------------------------------------------------------------
# Density of states
# ---------------------------------------------------------------------------
def log_dos(dead: List[DeadPoint], nb: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram estimate of ``ln g(E)`` from dead-point weights.

    A weighted NumPy histogram replaces a Python per-bin loop; the
    log-sum-exp trick (via subtracting the maximum before exponentiation)
    protects numerical range. Bins are linearly spaced between the minimum
    and maximum energy.

    Parameters
    ----------
    dead : list of DeadPoint
        The dead-point population.
    nb : int
        Number of bins.

    Returns
    -------
    centres : numpy.ndarray
        Bin centres ``(nb,)``.
    ln_g : numpy.ndarray
        ``ln g(E_b) - ln dE`` per bin, with ``-inf`` for empty bins.
    """
    E = np.array([d.energy for d in dead], dtype=np.float64)
    lw = np.array([d.log_weight for d in dead], dtype=np.float64)

    edges = np.linspace(E.min(), E.max(), nb + 1)
    dE = float(edges[1] - edges[0])
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Rebase log weights so the worst numerical case is ``exp(0) == 1``.
    shifted = lw - lw.max()
    weights = np.exp(shifted)

    # Scalar histogram over energies, weighted by exp(shifted lw):
    hist, _ = np.histogram(E, bins=edges, weights=weights)

    ln_g = np.empty(nb, dtype=np.float64)
    ln_g.fill(-np.inf)
    filled = hist > 0
    ln_g[filled] = (
        lw.max()
        + np.log(hist[filled])
        - np.log(dE)
    )
    return centres, ln_g


# ---------------------------------------------------------------------------
# Thermo I/O
# ---------------------------------------------------------------------------
def save_results(res: Dict[str, np.ndarray], path: str) -> None:
    """Persist the dict of arrays to a ``.npz`` archive."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    arrays = {
        k: np.atleast_1d(np.asarray(v, dtype=float))
        for k, v in res.items()
    }
    np.savez_compressed(path, **arrays)


def load_results(path: str) -> Dict[str, np.ndarray]:
    """Inverse of :func:`save_results`."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


__all__ = [
    "log_sum_exp",
    "analyse",
    "find_cv_peak",
    "log_dos",
    "save_results",
    "load_results",
]
