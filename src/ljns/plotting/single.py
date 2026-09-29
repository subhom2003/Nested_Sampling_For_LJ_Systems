"""
Per-density 6-panel thermodynamic figure.

Panels (2 rows × 3 cols, labelled (a)–(f)):
  (a) C_v/N vs T
  (b) S_conf/N vs T
  (c) ⟨U⟩/N vs T
  (d) P vs T
  (e) F_total/N vs T
  (f) ln g(E) vs E
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .style import publication_style, _style_ax, _save, _panel_label
from ..thermodynamics import log_dos
from ..nested_sampling import DeadPoint


def plot_single_density(
    results: dict,
    dead: list,
    output_dir: str,
) -> str:
    """Create the 6-panel per-density thermodynamic figure.

    Parameters
    ----------
    results : dict
        Output of analyse() for one density.
    dead : list of DeadPoint
        Dead points from run_ns() (for ln g(E)).
    output_dir : str
        Output directory.

    Returns
    -------
    str
        Path to the saved PNG.
    """
    rho = float(results["rho"])
    N = int(results["n_atoms"])
    T = results["T"]

    with publication_style():
        fig = plt.figure(figsize=(7.0, 4.8))  # double-column width
        fig.suptitle(
            f"LJ Bulk Fluid — Thermodynamics at $\\rho^*={rho:.3f}$  "
            f"($N={N}$)",
            fontsize=9, fontweight="bold", y=0.995,
        )
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.32)

        # Panel (a) — Cv/N
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(T, results["Cv_per_atom"], color="#56B4E9", lw=1.3)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$C_v/N k_B$",
                  title="Heat Capacity")
        _panel_label(ax, "a")

        # Panel (b) — S_conf/N
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(T, results["S_conf_per_atom"], color="#009E73", lw=1.3)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$S_{\mathrm{conf}}/N k_B$",
                  title="Config. Entropy")
        _panel_label(ax, "b")

        # Panel (c) — <U>/N
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(T, results["mean_U_per_atom"], color="#E69F00", lw=1.3)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$\langle U\rangle/N\varepsilon$",
                  title="Potential Energy/Atom")
        _panel_label(ax, "c")

        # Panel (d) — P
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(T, results["mean_P"], color="#0072B2", lw=1.3)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$P^*$ [$\varepsilon\sigma^{-3}$]",
                  title="Pressure")
        _panel_label(ax, "d")

        # Panel (e) — F_total/N
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(T, results["F_total_per_atom"], color="#D55E00", lw=1.3)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$F/N\varepsilon$",
                  title="Helmholtz Free Energy")
        _panel_label(ax, "e")

        # Panel (f) — ln g(E)
        ax = fig.add_subplot(gs[1, 2])
        E_ctrs, ln_g = log_dos(dead)
        finite = np.isfinite(ln_g)
        ax.plot(E_ctrs[finite], ln_g[finite], color="#CC79A7", lw=1.2)
        _style_ax(ax, xlabel=r"$E^*$ [$\varepsilon$]", ylabel=r"$\ln g(E^*)$",
                  title="Log Density of States")
        _panel_label(ax, "f")

        path = os.path.join(output_dir, f"thermo_rho{rho:.3f}")
        return _save(fig, path)


__all__ = ["plot_single_density"]