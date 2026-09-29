"""
Overview figures combining multiple densities.

Publication-style figures:
1. Cv/N vs T (all densities, density-colored, T_peak markers)
2. T_peak vs ρ (clean coexistence dome, NS only)
3. Caloric curves <U>/N vs T
4. Entropy S_conf/N vs T
5. Helmholtz free energy F_total/N vs T
6. Pressure isochores P vs T
7. Compressibility factor Z = PV/NkT vs T
8. Energy convergence (E_dead vs iteration)
9. 3D surface: Cv vs T vs ρ
10. Phase diagram in T-ρ plane (projection of 3D)
11. Phase diagram in P-ρ plane
12. Phase diagram in P-T plane
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .style import (
    publication_style, _style_ax, _colors_for_densities,
    _add_colorbar, _save, OKABE_ITO,
)
from ..config import SimConfig


# ---------------------------------------------------------------------------
# 1. Cv vs T (all densities)
# ---------------------------------------------------------------------------
def plot_cv_all(
    all_results: dict,
    phase_data: dict,
    output_dir: str,
) -> str:
    """Heat capacity per particle vs T, all densities overlaid."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)

        for rho in dens:
            res = all_results[rho]
            ax.plot(res["T"], res["Cv_per_atom"],
                    color=colors[rho], lw=1.2)
            if rho in phase_data:
                ax.axvline(phase_data[rho]["T_peak"],
                           color=colors[rho], lw=0.6, ls="--", alpha=0.6)

        _add_colorbar(fig, ax, dens)
        _style_ax(ax,
                  xlabel=r"Temperature $T^*$ [$\varepsilon/k_B$]",
                  ylabel=r"$C_v / N k_B$",
                  title="Heat Capacity vs Temperature")

        path = os.path.join(output_dir, "cv_vs_T_all_densities")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 2. Phase diagram T_peak vs rho (CLEAN - NS only)
# ---------------------------------------------------------------------------
def plot_phase_diagram(phase_data: dict, output_dir: str) -> str:
    """Clean coexistence boundary T_peak vs rho (NS only)."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(phase_data.keys())
        T_peaks = [phase_data[r]["T_peak"] for r in dens]

        ax.plot(dens, T_peaks, "o-", color=OKABE_ITO[1], lw=1.8,
                ms=5, mfc="white", mew=1.2, label="NS (Cv peak)")

        _style_ax(ax,
                  xlabel=r"Density $\rho^*$ [$\sigma^{-3}$]",
                  ylabel=r"Temperature $T^*$ [$\varepsilon/k_B$]",
                  title="Coexistence Boundary")
        ax.legend(fontsize=6, loc="upper right", frameon=False)
        ax.set_xlim(0, max(dens)*1.05)
        ax.set_ylim(0, max(T_peaks)*1.2)

        path = os.path.join(output_dir, "phase_diagram_T_rho")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 3. Caloric curves
# ---------------------------------------------------------------------------
def plot_caloric(all_results: dict, phase_data: dict, output_dir: str) -> str:
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)
        for rho in dens:
            res = all_results[rho]
            ax.plot(res["T"], res["mean_U_per_atom"], color=colors[rho], lw=1.2)
            if rho in phase_data:
                ax.axvline(phase_data[rho]["T_peak"],
                           color=colors[rho], lw=0.5, ls="--", alpha=0.5)
        _add_colorbar(fig, ax, dens)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$\langle U\rangle/N\varepsilon$",
                  title="Caloric Curves")
        path = os.path.join(output_dir, "caloric_curves")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 4. Entropy
# ---------------------------------------------------------------------------
def plot_entropy(all_results: dict, phase_data: dict, output_dir: str) -> str:
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)
        for rho in dens:
            res = all_results[rho]
            ax.plot(res["T"], res["S_conf_per_atom"], color=colors[rho], lw=1.2)
            if rho in phase_data:
                ax.axvline(phase_data[rho]["T_peak"],
                           color=colors[rho], lw=0.5, ls="--", alpha=0.5)
        _add_colorbar(fig, ax, dens)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$S_{\mathrm{conf}}/N k_B$",
                  title="Configurational Entropy")
        path = os.path.join(output_dir, "entropy_vs_T")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 5. Free energy
# ---------------------------------------------------------------------------
def plot_free_energy(all_results: dict, output_dir: str) -> str:
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)
        for rho in dens:
            ax.plot(all_results[rho]["T"],
                    all_results[rho]["F_total_per_atom"],
                    color=colors[rho], lw=1.2)
        _add_colorbar(fig, ax, dens)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$F/N\varepsilon$",
                  title="Helmholtz Free Energy (total)")
        path = os.path.join(output_dir, "free_energy_vs_T")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 6. Pressure isochores
# ---------------------------------------------------------------------------
def plot_pressure(all_results: dict, phase_data: dict, output_dir: str) -> str:
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)
        for rho in dens:
            res = all_results[rho]
            ax.plot(res["T"], res["mean_P"], color=colors[rho], lw=1.2)
            if rho in phase_data:
                ax.axvline(phase_data[rho]["T_peak"],
                           color=colors[rho], lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0, color="0.3", lw=0.5, ls=":")
        _add_colorbar(fig, ax, dens)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$P^*$ [$\varepsilon\sigma^{-3}$]",
                  title="Pressure Isochores (virial + tail)")
        path = os.path.join(output_dir, "pressure_isochores")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 7. Compressibility factor
# ---------------------------------------------------------------------------
def plot_compressibility(all_results: dict, output_dir: str) -> str:
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)
        for rho in dens:
            ax.plot(all_results[rho]["T"],
                    all_results[rho]["Z_factor"],
                    color=colors[rho], lw=1.2)
        ax.axhline(1.0, color="0.3", lw=0.8, ls="--", alpha=0.6,
                   label="Ideal gas $Z=1$")
        _add_colorbar(fig, ax, dens)
        _style_ax(ax, xlabel=r"$T^*$", ylabel=r"$Z = PV/Nk_BT$",
                  title="Compressibility Factor")
        ax.legend(fontsize=6, loc="upper right")
        path = os.path.join(output_dir, "compressibility_factor")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 8. Energy convergence (E_dead vs iteration)
# ---------------------------------------------------------------------------
def plot_energy_convergence(
    all_dead_points: dict,
    output_dir: str,
) -> str:
    """Plot E_dead vs NS iteration for all densities."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_dead_points.keys())
        colors = _colors_for_densities(dens)

        for rho in dens:
            dead = all_dead_points[rho]
            # Filter only free dead points (iteration < n_iter)
            # We don't have n_iter here, so plot all and let the tail be visible
            iters = [d.iteration for d in dead]
            energies = [d.energy for d in dead]
            ax.plot(iters, energies, color=colors[rho], lw=0.8, alpha=0.8)

        _add_colorbar(fig, ax, dens)
        _style_ax(ax,
                  xlabel="NS iteration $k$",
                  ylabel=r"Dead-point energy $E_k$ [$\varepsilon$]",
                  title="Energy Convergence")
        path = os.path.join(output_dir, "energy_convergence")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 9. 3D plot: Cv vs T vs rho (lines, not surface) + phase points in T-rho plane
# ---------------------------------------------------------------------------
def plot_cv_3d(all_results: dict, phase_data: dict, output_dir: str) -> str:
    """3D line plot of Cv/N vs T for each density, with phase points projected on T-rho plane."""
    with publication_style({"figure.figsize": (5.0, 4.0)}):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)

        # Plot Cv vs T lines for each density at its rho value (3D lines)
        for rho in dens:
            res = all_results[rho]
            T = res["T"]
            Cv = res["Cv_per_atom"]
            # Plot as 3D line at constant rho
            ax.plot(T, np.full_like(T, rho), Cv,
                    color=colors[rho], lw=0.8, alpha=0.7)

        # Phase points: (T_peak, rho, Cv_peak) - mark in 3D
        T_peaks = [phase_data[rho]["T_peak"] for rho in dens]
        Cv_peaks = [phase_data[rho]["Cv_peak"] for rho in dens]
        ax.scatter(T_peaks, dens, Cv_peaks, c='red', s=20, depthshade=False,
                   label='Phase boundary ($T_{\\mathrm{peak}}$)', zorder=5)

        # Project phase points onto T-rho plane at bottom (z=0)
        z_min = ax.get_zlim()[0]
        ax.scatter(T_peaks, dens, np.full_like(T_peaks, z_min), c='red', s=20,
                   marker='o', alpha=0.5, depthshade=False, zorder=4)

        # Also draw a line connecting the projected phase points on the T-rho plane
        ax.plot(T_peaks, dens, np.full_like(T_peaks, z_min), 'r--', lw=1.5,
                alpha=0.7, label='Coexistence curve (projected)')

        ax.set_xlabel(r'Temperature $T^*$ [$\varepsilon/k_B$]')
        ax.set_ylabel(r'Density $\rho^*$ [$\sigma^{-3}$]')
        ax.set_zlabel(r'$C_v / N k_B$')
        ax.set_title('Heat Capacity vs T and $\\rho$')
        ax.legend(fontsize=6, loc='upper left', frameon=False)

        path = os.path.join(output_dir, "cv_3d")
        return _save(fig, path)
        ax.set_title('Heat Capacity Surface')

        path = os.path.join(output_dir, "cv_3d_surface")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 10. Phase diagram in T-rho plane (from 3D Cv peak ridge)
# ---------------------------------------------------------------------------
def plot_phase_diagram_Trho(all_results: dict, phase_data: dict, output_dir: str) -> str:
    """T-rho phase diagram showing Cv peak ridge."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)

        # Plot Cv contours
        T = all_results[dens[0]]["T"]
        R, T_grid = np.meshgrid(dens, T)
        Cv_grid = np.array([all_results[rho]["Cv_per_atom"] for rho in dens]).T

        # Filled contours
        levels = np.linspace(Cv_grid.min(), Cv_grid.max(), 20)
        cf = ax.contourf(R, T_grid, Cv_grid, levels=levels, cmap='viridis', alpha=0.7)
        fig.colorbar(cf, ax=ax, label=r'$C_v / N k_B$')

        # Peak line
        T_peaks = [phase_data[rho]["T_peak"] for rho in dens]
        ax.plot(dens, T_peaks, "o-", color='red', lw=2, ms=4,
                mfc='white', mew=1.5, label='Cv peak')

        _style_ax(ax,
                  xlabel=r"Density $\rho^*$ [$\sigma^{-3}$]",
                  ylabel=r"Temperature $T^*$ [$\varepsilon/k_B$]",
                  title="Phase Diagram (T-ρ)")
        ax.legend(fontsize=6, loc="upper right", frameon=False)
        ax.set_xlim(0, max(dens)*1.05)
        ax.set_ylim(T.min(), T.max())

        path = os.path.join(output_dir, "phase_diagram_Trho")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 11. Phase diagram in P-rho plane
# ---------------------------------------------------------------------------
def plot_phase_diagram_Prho(all_results: dict, phase_data: dict, output_dir: str) -> str:
    """P-rho phase diagram at the Cv peak temperature for each density."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())

        # Get pressure at T_peak for each density
        P_at_peak = []
        for rho in dens:
            res = all_results[rho]
            T_peak = phase_data[rho]["T_peak"]
            # Interpolate P at T_peak
            P_interp = np.interp(T_peak, res["T"], res["mean_P"])
            P_at_peak.append(P_interp)

        ax.plot(dens, P_at_peak, "s-", color=OKABE_ITO[0], lw=1.8,
                ms=5, mfc="white", mew=1.2, label=r'$P(T_{\mathrm{peak}})$')

        _style_ax(ax,
                  xlabel=r"Density $\rho^*$ [$\sigma^{-3}$]",
                  ylabel=r"Pressure $P^*$ [$\varepsilon\sigma^{-3}$]",
                  title="Phase Diagram (P-ρ)")
        ax.legend(fontsize=6, loc="upper right", frameon=False)
        ax.set_xlim(0, max(dens)*1.05)

        path = os.path.join(output_dir, "phase_diagram_Prho")
        return _save(fig, path)


# ---------------------------------------------------------------------------
# 12. Phase diagram in P-T plane
# ---------------------------------------------------------------------------
def plot_phase_diagram_PT(all_results: dict, phase_data: dict, output_dir: str) -> str:
    """P-T phase diagram showing coexistence curve."""
    with publication_style():
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        dens = sorted(all_results.keys())
        colors = _colors_for_densities(dens)

        # Plot all isochores
        for rho in dens:
            res = all_results[rho]
            ax.plot(res["T"], res["mean_P"], color=colors[rho], lw=0.8, alpha=0.6)

        # Highlight coexistence curve (P at T_peak)
        T_peaks = [phase_data[rho]["T_peak"] for rho in dens]
        P_at_peak = []
        for rho in dens:
            res = all_results[rho]
            P_interp = np.interp(phase_data[rho]["T_peak"], res["T"], res["mean_P"])
            P_at_peak.append(P_interp)

        ax.plot(T_peaks, P_at_peak, "o-", color='red', lw=2, ms=4,
                mfc='white', mew=1.5, label='Coexistence curve')

        _add_colorbar(fig, ax, dens)
        _style_ax(ax,
                  xlabel=r"Temperature $T^*$ [$\varepsilon/k_B$]",
                  ylabel=r"Pressure $P^*$ [$\varepsilon\sigma^{-3}$]",
                  title="Phase Diagram (P-T)")
        ax.legend(fontsize=6, loc="upper right", frameon=False)

        path = os.path.join(output_dir, "phase_diagram_PT")
        return _save(fig, path)


__all__ = [
    "plot_cv_all", "plot_phase_diagram", "plot_caloric",
    "plot_entropy", "plot_free_energy", "plot_pressure",
    "plot_compressibility", "plot_energy_convergence",
    "plot_cv_3d", "plot_phase_diagram_Trho",
    "plot_phase_diagram_Prho", "plot_phase_diagram_PT",
]