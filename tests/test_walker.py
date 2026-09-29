"""Tests for the MC walker module."""
import numpy as np
import pytest

from ljns import walker
from ljns.walker import LivePoint, make_live_point, walk
from ljns.config import SimConfig


def test_live_point_deep_copy():
    """LivePoint.copy() should produce independent position arrays."""
    pos = np.random.rand(8, 3)
    lp = LivePoint(pos, -10.0, 5.0, 0.3, 2.5)
    lp2 = lp.copy()
    assert lp2.positions is not lp.positions
    assert np.array_equal(lp2.positions, lp.positions)
    # Mutate copy, original unchanged
    lp2.positions[0, 0] = 999.0
    assert lp.positions[0, 0] != 999.0


def test_make_live_point_returns_valid_state():
    """make_live_point should return a thermalised live point with E < 0."""
    cfg = SimConfig(n_atoms=8, r_cut=2.5, n_mc_steps=2, atom_step_size=0.1)
    rng = np.random.default_rng(42)
    lp = walker.make_live_point(cfg, 0.3, rng)
    assert isinstance(lp, walker.LivePoint)
    # Energy should be negative (bound fluid)
    assert lp.energy < 0


def test_walk_respects_ceiling():
    """Walker must never produce energy >= E_ceil."""
    cfg = SimConfig(n_atoms=8, n_walkers=4, n_iter=50, n_mc_steps=5, r_cut=2.5)
    rng = np.random.default_rng(0)
    lp = make_live_point(cfg, 0.3, rng)
    E_ceil = lp.energy - 0.5  # strict ceiling below current energy

    for _ in range(10):
        new_lp = walk(lp, E_ceil, cfg, rng)
        assert new_lp.energy < E_ceil + 1e-12, \
            f"Walk produced E={new_lp.energy} >= E_ceil={E_ceil}"


def test_walk_does_not_mutate_survivor():
    """walk() must not mutate the input live point."""
    cfg = SimConfig(n_atoms=8, n_walkers=4, n_iter=50, n_mc_steps=5, r_cut=2.5)
    rng = np.random.default_rng(0)
    lp = make_live_point(cfg, 0.3, rng)
    E_ceil = lp.energy - 0.5
    orig_pos = lp.positions.copy()
    orig_E = lp.energy

    new_lp = walk(lp, E_ceil, cfg, rng)

    assert np.array_equal(lp.positions, orig_pos)
    assert lp.energy == orig_E
    assert new_lp is not lp


def test_metropolis_detailed_balance_high_T():
    """At very high T, walk should approximate uniform sampling in the box."""
    cfg = SimConfig(n_atoms=4, n_walkers=4, n_iter=50, n_mc_steps=5, r_cut=2.5,
                    atom_step_size=0.5)  # large step for mixing
    rng = np.random.default_rng(0)
    lp = make_live_point(cfg, 0.001, rng)  # very low density
    E_ceil = 1e6  # effectively no ceiling
    # Do many walks, collect positions
    positions = []
    for _ in range(200):
        lp = walk(lp, E_ceil, cfg, rng)
        positions.append(lp.positions.copy())
    positions = np.array(positions)  # (200, 4, 3)
    # Mean position should be near box center
    L = cfg.box_length(0.001)
    assert np.allclose(positions.mean(axis=(0, 1)), L/2, atol=L/4)