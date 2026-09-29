"""
Lennard-Jones pair potential, virial, tail corrections, and lattice initialiser.

The module is the physics-calculation core for the bulk LJ phase-diagram
pipeline. It exposes:

- :func:`minimum_image` -- fold a displacement into the primary periodic cell.
- :func:`total_energy`, :func:`single_atom_energy`, :func:`energy_and_virial`
  -- potential-energy and virial evaluators with truncated-and-shifted cutoffs.
- :func:`tail_corrections` -- energetic and virial tail corrections assuming
  a uniform fluid beyond ``r_cut``.
- :func:`tail_virial_volume_corrected` -- convenience helper that already
  multiplies the per-volume virial tail by the box volume, suitable for use
  in the NS loop where the dead-point snapshot stores a volume-weighted virial.
- :func:`init_positions` -- FCC lattice (preferred) or simple-cubic lattice
  (fallback) generator with small symmetry-breaking noise.

Conventions
-----------
All routines take and return standard ``float`` values (energies) or
``numpy.ndarray`` (positions). Energies and virials are reported in reduced
LJ units, i.e. ``epsilon = sigma = k_B = 1``.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig


# ---------------------------------------------------------------------------
# Minimum image and pair-potential primitives
# ---------------------------------------------------------------------------
def minimum_image(dr: np.ndarray, L: float) -> np.ndarray:
    """Fold a displacement ``dr`` into the minimum-image convention primary cell.

    Uses ``floor(dr / L + 0.5)`` so that each pair contributes exactly one
    periodic image, including the symmetric ``+/- L/2`` boundary. This is
    the canonical choice and is robust to ``np.round`` banker's-rounding
    surprises at half-cell displacements.

    Parameters
    ----------
    dr : numpy.ndarray
        Displacement (or batch of displacements) in shape ``(N, 3)`` or
        ``(M, N, 3)``.
    L : float
        Cubic-box edge.

    Returns
    -------
    numpy.ndarray
        Same shape as ``dr``; the folded displacement.
    """
    return dr - L * np.floor(dr / L + 0.5)


def _lj_energy_from_ir2(ir2: np.ndarray, epsilon: float) -> np.ndarray:
    """Compute ``U`` given ``ir2 = sigma^2 / r^2``.

    Code path takes ``ir2`` (rather than ``r``) to skip division inside the
    inner loops. Returns

    .. math:: U = 4 \\varepsilon \\left[ \\left(\\frac{\\sigma}{r}\\right)^{12}
       - \\left(\\frac{\\sigma}{r}\\right)^6 \\right]
    """
    ir6 = ir2 ** 3
    return 4.0 * epsilon * ir6 * (ir6 - 1.0)


def _lj_virial_from_ir2(ir2: np.ndarray, epsilon: float) -> np.ndarray:
    """Compute ``W_ij = r_ij * dV/dr`` given ``ir2 = sigma^2 / r^2``.

    For the LJ pair potential, the virial contribution of a single pair
    (``i, j``) is

    .. math:: W_{ij} = 24 \\varepsilon \\left[ 2 \\left(\\frac{\\sigma}{r}\\right)^{12}
       - \\left(\\frac{\\sigma}{r}\\right)^6 \\right]
    """
    ir6 = ir2 ** 3
    return 24.0 * epsilon * ir6 * (2.0 * ir6 - 1.0)


# ---------------------------------------------------------------------------
# Energy evaluators
# ---------------------------------------------------------------------------
def total_energy(pos: np.ndarray, L: float, rc: float, cfg: SimConfig) -> float:
    """Compute the total configurational energy ``U`` of the system.

    Uses an O(N^2) pair loop directly; suitable for the small system sizes
    (N <= a few hundred) targeted here.

    Parameters
    ----------
    pos : numpy.ndarray
        Atomic coordinates of shape ``(N, 3)``, in the primary cell.
    L : float
        Box edge.
    rc : float
        Effective cutoff (already respecting minimum-image convention).
    cfg : SimConfig
        Configuration used for ``sigma`` and ``epsilon``.

    Returns
    -------
    float
        Total truncated LJ potential energy.
    """
    n = len(pos)
    rc2 = rc * rc
    s2 = cfg.sigma ** 2
    eps = cfg.epsilon
    U = 0.0
    for i in range(n - 1):
        dr = minimum_image(pos[i + 1:] - pos[i], L)
        r2 = np.einsum("ij,ij->i", dr, dr)
        mask = r2 < rc2
        if mask.any():
            U += float(np.sum(_lj_energy_from_ir2(s2 / r2[mask], eps)))
    return U


def single_atom_energy(i: int, pos: np.ndarray, L: float,
                       rc: float, cfg: SimConfig) -> float:
    """Compute the partial energy of atom ``i`` with all ``j != i`` pairs.

    Used inside the MC walker where trial moves are evaluated from a
    single-atom displacement; O(N) per call rather than O(N^2).

    Parameters
    ----------
    i : int
        Index of the atom whose partial energy is requested.
    pos : numpy.ndarray
        Positions of all atoms, ``(N, 3)``.
    L : float
        Box edge.
    rc : float
        Effective cutoff.
    cfg : SimConfig
        Configuration.

    Returns
    -------
    float
        Sum of ``V_LJ(r_ij)`` for all ``j != i``.
    """
    rc2 = rc * rc
    s2 = cfg.sigma ** 2
    eps = cfg.epsilon
    mask = np.arange(len(pos)) != i
    dr = minimum_image(pos[mask] - pos[i], L)
    r2 = np.einsum("ij,ij->i", dr, dr)
    in_range = r2 < rc2
    if not in_range.any():
        return 0.0
    return float(np.sum(_lj_energy_from_ir2(s2 / r2[in_range], eps)))


def energy_and_virial(pos: np.ndarray, L: float, rc: float,
                      cfg: SimConfig) -> tuple:
    """Compute ``(U, W)`` in a single O(N^2) pass.

    Distance calculations are shared between the energy and virial
    accumulation, removing redundant work compared with two separate calls.

    Parameters
    ----------
    pos : numpy.ndarray
        Atomic positions ``(N, 3)``.
    L : float
        Box edge.
    rc : float
        Effective cutoff.
    cfg : SimConfig
        Configuration.

    Returns
    -------
    U, W : float
        Total truncated potential energy and total virial sum.
    """
    n = len(pos)
    rc2 = rc * rc
    s2 = cfg.sigma ** 2
    eps = cfg.epsilon
    U = 0.0
    W = 0.0
    for i in range(n - 1):
        dr = minimum_image(pos[i + 1:] - pos[i], L)
        r2 = np.einsum("ij,ij->i", dr, dr)
        mask = r2 < rc2
        if mask.any():
            ir2 = s2 / r2[mask]
            U += float(np.sum(_lj_energy_from_ir2(ir2, eps)))
            W += float(np.sum(_lj_virial_from_ir2(ir2, eps)))
    return U, W


# ---------------------------------------------------------------------------
# Tail corrections
# ---------------------------------------------------------------------------
def tail_corrections(N: int, rho: float, rc: float,
                     cfg: SimConfig) -> tuple:
    """Return the uniform-fluid tail corrections for energy and virial.

    The truncated potential is corrected by assuming the fluid is uniform
    beyond ``r_cut``:

    .. math::
        U_{\\text{tail}} &= \\frac{8\\pi}{3}\\, N \\rho \\varepsilon
                          \\left[ \\frac{1}{3}\\left(\\frac{\\sigma}{r_c}\\right)^9
                          - \\left(\\frac{\\sigma}{r_c}\\right)^3 \\right] \\\\
        W_{\\text{tail}}^V &= \\frac{16\\pi}{3}\\, \\rho^2 \\varepsilon
                            \\left[ \\frac{2}{3}\\left(\\frac{\\sigma}{r_c}\\right)^9
                            - \\left(\\frac{\\sigma}{r_c}\\right)^3 \\right]

    The second return value is the **per-volume** virial tail, i.e.
    ``W_tail / V`` -- equivalent to a pressure contribution -- so the caller
    must multiply by the box volume ``V`` to obtain the absolute virial
    correction (use :func:`tail_virial_volume_corrected` for this convenience).

    Parameters
    ----------
    N : int
        Number of atoms.
    rho : float
        Number density.
    rc : float
        Effective cutoff.
    cfg : SimConfig
        Provides ``epsilon`` and ``sigma``.

    Returns
    -------
    U_tail : float
        Energy tail correction ( Joule-reduced units ).
    W_tail_per_V : float
        Virial tail correction per unit volume (pressure contribution).
    """
    eps = cfg.epsilon
    s = cfg.sigma / rc
    s3 = s ** 3
    s9 = s3 ** 3
    U_tail = (8.0 * np.pi / 3.0) * N * rho * eps * (s9 / 3.0 - s3)
    W_tail_per_V = (16.0 * np.pi / 3.0) * (rho ** 2) * eps * (2.0 * s9 / 3.0 - s3)
    return U_tail, W_tail_per_V


def tail_virial_volume_corrected(N: int, rho: float, L: float,
                                 rc: float, cfg: SimConfig) -> float:
    """Return the volume-weighted virial tail ``W_tail^V * V`` directly.

    The full virial stored inside a dead point is ``W_truncated + W_tail^V * V``
    -- where ``W_tail^V`` is the per-volume virial tail returned by
    :func:`tail_corrections` and ``V = L^3`` is the box volume. This helper
    bundles that multiplication to make a once-confusing code path explicit.

    Parameters
    ----------
    N : int
        Number of atoms.
    rho : float
        Number density.
    L : float
        Box edge.
    rc : float
        Effective cutoff.
    cfg : SimConfig
        Configuration.

    Returns
    -------
    float
        The volume-weighted virial tail correction.
    """
    _, W_tail_per_V = tail_corrections(N, rho, rc, cfg)
    return W_tail_per_V * (L ** 3)


# ---------------------------------------------------------------------------
# Initial positions: FCC lattice (preferred), simple-cubic fallback
# ---------------------------------------------------------------------------
def init_positions(N: int, L: float, rng: np.random.Generator) -> np.ndarray:
    """Generate an initial crystal lattice configuration in the primary cell.

    If the requested system size matches a clean face-centred-cubic lattice
    (``N == 4 m^3`` for integer ``m``), an FCC lattice is built from
    conventional 4-point basis vectors on an ``m x m x m`` supercell.
    Otherwise a simple-cubic grid just large enough to hold ``N`` atoms is
    used. Either way, a tiny symmetry-breaking Gaussian noise of width
    ``0.03 a`` is added so that the initial energies do not hit a singular
    crystal energy minimum.

    Parameters
    ----------
    N : int
        Number of atoms.
    L : float
        Box edge.
    rng : numpy.random.Generator
        Source of randomness for the symmetry-breaking noise.

    Returns
    -------
    numpy.ndarray
        Positions ``(N, 3)`` folded into the primary cell (``% L``).
    """
    m = round((N / 4) ** (1.0 / 3.0))
    if 4 * m ** 3 == N:
        # Face-centred cubic on an m x m x m supercell.
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])
        a = L / m
        cells = (np.mgrid[0:m, 0:m, 0:m].reshape(3, -1).T)  # (m^3, 3) integers
        # (m^3, 4, 3) = cells[:,None,:] + basis[None,:,:], times a:
        pos = (cells[:, None, :] + basis[None, :, :]) * a
        pos = pos.reshape(-1, 3)
        pos = pos + rng.standard_normal(pos.shape) * 0.03 * a
    else:
        # Simple-cubic fallback for arbitrary N.
        mc = int(np.ceil(N ** (1.0 / 3.0)))
        a = L / mc
        grid = np.array([(ix, iy, iz)
                         for ix in range(mc)
                         for iy in range(mc)
                         for iz in range(mc)])
        pos = grid[:N].astype(float) * a
        pos = pos + rng.standard_normal(pos.shape) * 0.03 * a
    return pos % L


__all__ = [
    "minimum_image",
    "total_energy",
    "single_atom_energy",
    "energy_and_virial",
    "tail_corrections",
    "tail_virial_volume_corrected",
    "init_positions",
]
