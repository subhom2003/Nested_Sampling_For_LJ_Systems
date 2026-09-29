"""End-to-end CLI smoke test using a tiny config."""
import subprocess
import sys
import json
from pathlib import Path
import tempfile


def test_cli_tiny_run():
    """Run a minimal NS campaign and verify output files exist."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Write a minimal test config
        cfg = {
            "physics": {"n_atoms": 8, "epsilon": 1.0, "sigma": 1.0, "r_cut": 2.5},
            "ns": {"n_walkers": 4, "n_iter": 30, "n_mc_steps": 2, "atom_step_size": 0.1},
            "scan": {"densities": [0.3, 0.6], "T_min": 0.3, "T_max": 2.0, "n_T": 50,
                     "cv_smooth_window": 0},
            "parallel": {"n_workers": 1, "backend": "loky"},
            "io": {"output_dir": str(d), "save_dead_points": True,
                   "save_thermo": True, "make_plots": False, "per_density_plots": False},
            "seed": 42,
            "verbosity": "warning",
        }
        cfg_path = d / "test_tiny.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        # Run CLI
        result = subprocess.run(
            [sys.executable, "-m", "ljns.cli", "--config", str(cfg_path),
             "--cores", "1", "--no-plots"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        # Verify outputs
        dead_files = list(d.glob("dead_rho*.npz"))
        thermo_files = list(d.glob("thermo_rho*.npz"))
        assert len(dead_files) == 2, f"Expected 2 dead files, got {len(dead_files)}"
        assert len(thermo_files) == 2, f"Expected 2 thermo files, got {len(thermo_files)}"
        assert (d / "summary.txt").exists()