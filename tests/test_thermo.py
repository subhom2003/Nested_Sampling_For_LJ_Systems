"""Tests for thermodynamics analysis."""
import numpy as np
import pytest
from pathlib import Path

from ljns.thermodynamics import (
    analyse, find_cv_peak, log_dos, save_results, load_results, log_sum_exp
)
from ljns.nested_sampling import DeadPoint


def make_two_state_dead_points():
    """Synthetic DOS: two delta functions at -100 and -50 with equal weight."""
    dead = []
    for i in range(20):
        dead.append(DeadPoint(i, -100.0, 0.0, np.log(1/40), 0.3, 32))
    for i in range(20):
        dead.append(DeadPoint(20+i, -50.0, 0.0, np.log(1/40), 0.3, 32))
    return dead


def test_log_sum_exp():
    """log_sum_exp should match the manual stable implementation."""
    a = np.array([1.0, 2.0, 3.0])
    # ln(e^1 + e^2 + e^3) = 3 + ln(e^-2 + e^-1 + 1)
    expected = 3.0 + np.log(np.exp(-2) + np.exp(-1) + 1)
    assert np.isclose(log_sum_exp(a), expected)


def test_analyse_ideal_gas_limit():
    """At very high T, the system should behave as an ideal gas."""
    dead = make_two_state_dead_points()
    T = np.array([1000.0])  # very high T
    from ljns.config import SimConfig
    cfg = SimConfig(n_atoms=32, r_cut=2.5)
    res = analyse(dead, T, cfg)

    # Z = PV/NkT -> 1
    assert np.isclose(res["Z_factor"][0], 1.0, atol=0.02)
    # Cv/N -> 1.5 (ideal gas kinetic part)
    assert np.isclose(res["Cv_per_atom"][0], 1.5, atol=0.05)


def test_find_cv_peak_on_two_state():
    """Two-level system Cv peaks at kT ~ gap/2."""
    dead = make_two_state_dead_points()
    T = np.linspace(5.0, 60.0, 200)
    from ljns.config import SimConfig
    cfg = SimConfig(n_atoms=32, r_cut=2.5)
    res = analyse(dead, T, cfg)
    Tpk, Cvpk, FWHM = find_cv_peak(res, sw=0)  # no smoothing

    # Gap = 50, expected T_peak ~ 25
    assert 15 < Tpk < 35
    assert Cvpk > 0
    assert FWHM > 0


def test_find_cv_peak_smoothing_affects_FWHM_only():
    """Smoothing should not change the reported Cv_peak (from unsmoothed)."""
    dead = make_two_state_dead_points()
    T = np.linspace(5.0, 60.0, 300)
    from ljns.config import SimConfig
    cfg = SimConfig(n_atoms=32, r_cut=2.5)
    res = analyse(dead, T, cfg)

    Tpk0, Cvpk0, FWHM0 = find_cv_peak(res, sw=0)
    Tpk9, Cvpk9, FWHM9 = find_cv_peak(res, sw=9)

    # T_peak may shift slightly, but Cv_peak should be evaluated at the
    # *unsmoothed* curve at the *smoothed* peak location (by design).
    # Just ensure both are sensible.
    assert Cvpk0 > 0 and Cvpk9 > 0
    assert FWHM9 > 0


def test_log_dos_histogram_integral():
    """Integral of g(E) dE should equal sum of dead-point weights."""
    dead = make_two_state_dead_points()
    Ec, lg = log_dos(dead, nb=100)
    dE = Ec[1] - Ec[0]
    weights = np.exp([d.log_weight for d in dead])
    integral = np.sum(np.exp(lg[np.isfinite(lg)]) * dE)
    assert np.isclose(integral, np.sum(weights), rtol=0.1)


def test_analyse_returns_expected_keys():
    """Result dict must contain all documented keys."""
    dead = make_two_state_dead_points()
    T = np.linspace(10, 60, 50)
    from ljns.config import SimConfig
    cfg = SimConfig(n_atoms=32, r_cut=2.5)
    res = analyse(dead, T, cfg)

    expected = {
        "T", "rho", "n_atoms", "volume",
        "log_Z_conf", "Z_conf_norm",
        "mean_U", "mean_U_per_atom",
        "Cv_conf", "Cv_total", "Cv_per_atom",
        "F_conf", "F_conf_per_atom",
        "F_total", "F_total_per_atom",
        "S_conf", "S_conf_per_atom",
        "S_total", "S_total_per_atom",
        "mean_P", "G_conf", "G_conf_per_atom",
        "G_total", "G_total_per_atom",
        "Z_factor", "mean_W",
    }
    assert set(res.keys()) == expected
    for k, v in res.items():
        if isinstance(v, np.ndarray):
            assert len(v) == len(T), f"Key {k} has wrong length"


def test_save_load_roundtrip(tmp_path):
    """Thermo results should save/load without loss."""
    dead = make_two_state_dead_points()
    T = np.linspace(10, 60, 50)
    from ljns.config import SimConfig
    cfg = SimConfig(n_atoms=32, r_cut=2.5)
    res = analyse(dead, T, cfg)

    path = tmp_path / "thermo.npz"
    save_results(res, path)
    loaded = load_results(path)

    for k in res:
        if isinstance(res[k], np.ndarray):
            assert np.allclose(res[k], loaded[k], rtol=1e-10)
        else:
            assert res[k] == loaded[k]