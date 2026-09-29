"""Tests for the LJ potential module."""
import numpy as np
import pytest

from ljns import potential
from ljns.potential import (
    minimum_image,
    _lj_energy_from_ir2,
    _lj_virial_from_ir2,
    total_energy,
    single_atom_energy,
    energy_and_virial,
    tail_corrections,
    tail_virial_volume_corrected,
    init_positions,
)
from ljns.config import SimConfig


def test_minimum_image():
    """Displacements should fold into [-L/2, L/2]."""
    L = 10.0
    # Exact boundary at L/2
    dr = np.array([5.0, -5.0, 0.0])
    folded = minimum_image(dr, L)
    # Banker's rounding can go either way at exact half; just check magnitude
    assert np.all(np.abs(folded) <= L/2 + 1e-10)

    # Regular displacements
    dr = np.array([12.0, -3.0, 7.0])
    folded = minimum_image(dr, L)
    assert np.allclose(folded, [2.0, -3.0, -3.0])


def test_lj_energy_at_known_r():
    """LJ energy at r = 2^(1/6) sigma should be -epsilon (the well minimum)."""
    sigma = 1.0
    eps = 1.0
    r_min = 2**(1/6) * sigma
    ir2 = sigma**2 / r_min**2
    U = _lj_energy_from_ir2(np.array([ir2]), eps)[0]
    assert np.isclose(U, -eps, rtol=1e-10)


def test_lj_virial_at_known_r():
    """Virial at the potential minimum r = 2^(1/6) should be zero."""
    sigma = 1.0
    eps = 1.0
    r_min = 2**(1/6) * sigma
    ir2 = sigma**2 / r_min**2
    W = _lj_virial_from_ir2(np.array([ir2]), eps)[0]
    assert np.isclose(W, 0.0, atol=1e-10)


def test_tail_corrections_formula():
    """Tail corrections should match the analytic formula from the literature."""
    cfg = SimConfig(n_atoms=100, r_cut=2.5, epsilon=1.0, sigma=1.0)
    N = 100
    rho = 0.8
    rc = 2.5
    U_tail, W_tail_per_V = tail_corrections(N, rho, rc, cfg)

    # Compare to hand-coded formula
    s = 1.0 / rc
    s3 = s**3
    s9 = s3**3
    U_expected = (8*np.pi/3) * N * rho * 1.0 * (s9/3 - s3)
    W_expected = (16*np.pi/3) * rho**2 * 1.0 * (2*s9/3 - s3)

    assert np.isclose(U_tail, U_expected, rtol=1e-10)
    assert np.isclose(W_tail_per_V, W_expected, rtol=1e-10)


def test_tail_virial_volume_corrected():
    """Volume-weighted virial tail should equal W_tail_per_V * V."""
    cfg = SimConfig(n_atoms=100, r_cut=2.5)
    N = 100
    rho = 0.8
    rc = 2.5
    L = (N/rho)**(1/3)
    U_tail, W_per_V = potential.tail_corrections(N, rho, rc, cfg)  # returns (U, W_per_V)
    W_V = potential.tail_virial_volume_corrected(N, rho, L, rc, cfg)
    assert np.isclose(W_V, W_per_V * L**3, rtol=1e-10)


def test_single_atom_energy_matches_total():
    """Sum of single-atom energies / 2 should equal total energy."""
    cfg = SimConfig(n_atoms=8, r_cut=2.5)
    rng = np.random.default_rng(0)
    L = cfg.box_length(0.3)
    rc = cfg.r_cut_eff(0.3)
    pos = init_positions(8, L, rng)

    U_tot = total_energy(pos, L, rc, cfg)
    U_sa_sum = sum(single_atom_energy(i, pos, L, rc, cfg) for i in range(8))
    # Each pair counted twice in single-atom sum
    assert np.isclose(U_tot, U_sa_sum / 2, rtol=1e-10)


def test_energy_and_virial_consistency():
    """energy_and_virial should give same energy as total_energy."""
    cfg = SimConfig(n_atoms=8, r_cut=2.5)
    rng = np.random.default_rng(1)
    L = cfg.box_length(0.3)
    rc = cfg.r_cut_eff(0.3)
    pos = init_positions(8, L, rng)

    U1 = total_energy(pos, L, rc, cfg)
    U2, W = energy_and_virial(pos, L, rc, cfg)
    assert np.isclose(U1, U2, rtol=1e-10)


def test_fcc_lattice_for_N32():
    """N=32 is a perfect FCC size (4 * 2^3); check NN distance."""
    cfg = SimConfig(n_atoms=32, r_cut=2.5)
    L = cfg.box_length(0.3)
    rng = np.random.default_rng(0)
    pos = potential.init_positions(32, L, rng)
    # FCC nearest neighbor distance = a / sqrt(2) where a = L/2 = 1.5
    # But we added noise (0.03*a). Expected ~1.06 without noise,
    # with noise ~1.06 ± 0.03*a = 1.06 ± 0.05
    # Just check it's roughly FCC scale
    diffs = pos[:, None, :] - pos[None, :, :]
    diffs = potential.minimum_image(diffs.reshape(-1, 3), L).reshape(32, 32, 3)
    dists = np.sqrt((diffs**2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    # FCC NN = a/sqrt(2) where a = L/m = L/2
    a = L / 2
    expected_nn = a / np.sqrt(2)
    # With 0.03*a noise, expect ~10% variation
    assert np.isclose(min_dist, expected_nn, rtol=0.15)


def test_sc_fallback_for_non_fcc():
    """N=8 is not FCC (4*m^3); should fall back to simple cubic."""
    cfg = SimConfig(n_atoms=8)
    rng = np.random.default_rng(42)
    L = cfg.box_length(0.3)
    pos = init_positions(8, L, rng)
    assert len(pos) == 8
    # Just check no overlaps
    dists = np.linalg.norm(pos[:, None] - pos, axis=2)
    np.fill_diagonal(dists, np.inf)
    assert np.all(dists > 0.1)  # noise is 0.03*a, so >0.1 is safe