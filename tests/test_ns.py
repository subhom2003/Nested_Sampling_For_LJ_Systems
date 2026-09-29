"""Tests for the nested sampling core."""
import numpy as np
import pytest

from ljns.nested_sampling import run_ns, save_dead_points, load_dead_points, DeadPoint
from ljns.config import SimConfig


def test_run_ns_produces_monotonic_energies():
    """Dead-point energies must be non-increasing (monotonic compression)."""
    cfg = SimConfig(n_atoms=8, n_walkers=4, n_iter=40, n_mc_steps=2, r_cut=2.5)
    rng = np.random.default_rng(0)
    dead, meta = run_ns(cfg, 0.3, rng)

    # Free dead points (first n_iter) should have non-increasing energy
    free_dead = [d for d in dead if d.iteration < cfg.n_iter]
    energies = [d.energy for d in free_dead]
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i-1] + 1e-10, \
            f"Energy increased: {energies[i-1]} -> {energies[i]} at iter {i}"


def test_run_ns_correct_number_of_dead_points():
    """Total dead points = n_iter + n_walkers."""
    cfg = SimConfig(n_atoms=8, n_walkers=4, n_iter=40, n_mc_steps=2, r_cut=2.5)
    rng = np.random.default_rng(0)
    dead, meta = run_ns(cfg, 0.3, rng)
    assert len(dead) == cfg.n_iter + cfg.n_walkers
    assert meta["n_dead"] == cfg.n_iter + cfg.n_walkers


def test_dead_point_log_weights_match_theory():
    """log_weight of k-th free dead point = k*ln(K/(K+1)) + ln(1/(K+1))."""
    cfg = SimConfig(n_atoms=8, n_walkers=4, n_iter=40, n_mc_steps=2, r_cut=2.5)
    rng = np.random.default_rng(0)
    dead, _ = run_ns(cfg, 0.3, rng)

    K = cfg.n_walkers
    la = np.log(K / (K + 1))
    l1a = np.log(1 / (K + 1))

    for k in range(cfg.n_iter):
        d = dead[k]
        expected = k * la + l1a
        assert np.isclose(d.log_weight, expected, rtol=1e-10), \
            f"iter {k}: got {d.log_weight}, expected {expected}"

    # Final live points
    lXf = cfg.n_iter * la
    expected_final = lXf - np.log(K)
    for d in dead[cfg.n_iter:]:
        assert np.isclose(d.log_weight, expected_final, rtol=1e-10)


def test_save_load_dead_points_roundtrip(tmp_path):
    """Dead points saved to .npz must load back identically."""
    cfg = SimConfig(n_atoms=8, n_walkers=4, n_iter=20, n_mc_steps=2, r_cut=2.5)
    rng = np.random.default_rng(0)
    dead, _ = run_ns(cfg, 0.3, rng)

    path = tmp_path / "dead.npz"
    save_dead_points(dead, path)
    reloaded = load_dead_points(path)

    assert len(reloaded) == len(dead)
    for orig, new in zip(dead, reloaded):
        assert orig.iteration == new.iteration
        assert np.isclose(orig.energy, new.energy)
        assert np.isclose(orig.virial, new.virial)
        assert np.isclose(orig.log_weight, new.log_weight)
        assert np.isclose(orig.rho, new.rho)
        assert orig.n_atoms == new.n_atoms


def test_ns_meta_contains_expected_keys():
    cfg = SimConfig(n_atoms=8, n_walkers=4, n_iter=40, n_mc_steps=2, r_cut=2.5)
    rng = np.random.default_rng(0)
    _, meta = run_ns(cfg, 0.3, rng)
    for key in ["rho", "n_dead", "dead_log_sigma_alpha",
                "dead_log_const", "final_log_weight", "elapsed_s"]:
        assert key in meta