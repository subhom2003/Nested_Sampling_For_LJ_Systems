"""
Publication-quality matplotlib style and helpers.

This module centralises all figure aesthetics so that every plot produced
by the pipeline shares a consistent look suitable for journal submission.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import cm

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
# Okabe–Ito colourblind-safe palette (8 distinct colours)
OKABE_ITO = [
    "#000000",  # black (used sparingly)
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# For density-graded figures: perceptually uniform, colourblind-friendly
DENSITY_CMAP = "viridis"

# ---------------------------------------------------------------------------
# rcParams context manager
# ---------------------------------------------------------------------------
# Base style dictionary. Adjust as needed for specific journal templates.
_BASE_RCPARAMS = {
    # Fonts
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    # Math text
    "mathtext.fontset": "stix",
    "mathtext.default": "it",
    # Axes
    "axes.linewidth": 0.6,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.edgecolor": "0.15",
    "axes.labelcolor": "0.15",
    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "xtick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.major.size": 3.5,
    "ytick.minor.size": 2.0,
    "ytick.major.width": 0.6,
    "ytick.minor.width": 0.4,
    "xtick.color": "0.15",
    "ytick.color": "0.15",
    # Grid
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.35,
    "grid.alpha": 0.3,
    "grid.color": "0.5",
    # Legend
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "legend.columnspacing": 1.0,
    # Figure
    "figure.dpi": 300,
    "figure.figsize": (3.4, 2.6),  # single-column width in inches
    "figure.constrained_layout.use": True,
    # Lines
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.6,
    # Save
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
}


@contextlib.contextmanager
def publication_style(extra: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """Context manager that applies publication rcParams for the duration.

    Example
    -------
    >>> with publication_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot(x, y)
    ...     fig.savefig("fig.pdf")
    """
    orig = {k: matplotlib.rcParams[k] for k in _BASE_RCPARAMS}
    matplotlib.rcParams.update(_BASE_RCPARAMS)
    if extra:
        matplotlib.rcParams.update(extra)
    try:
        yield
    finally:
        matplotlib.rcParams.update(orig)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _style_ax(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> None:
    """Apply consistent axis styling."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _colors_for_densities(densities: List[float]) -> Dict[float, Tuple[float, ...]]:
    """Map densities to a sequential viridis colour for density-graded plots."""
    cmap = cm.get_cmap(DENSITY_CMAP)
    # Use 0.08–0.92 to avoid the extreme ends of viridis
    normed = np.linspace(0.08, 0.92, len(densities))
    return {r: cmap(v) for r, v in zip(sorted(densities), normed)}


def _add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    densities: List[float],
    label: str = r"Density $\rho^*$ [$\sigma^{-3}$]",
) -> None:
    """Add a shared colourbar for density-graded lines."""
    cmap = cm.get_cmap(DENSITY_CMAP)
    norm = plt.Normalize(vmin=min(densities), vmax=max(densities))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=30)
    cbar.set_label(label, fontsize=8)


def _save(fig: plt.Figure, path_no_ext: str) -> Tuple[str, str]:
    """Save figure as PNG (300 dpi). Returns (png_path)."""
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    png_path = path_no_ext + ".png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _panel_label(ax: plt.Axes, label: str, loc: str = "upper left") -> None:
    """Add a bold (a), (b), ... panel label inside the axes."""
    ax.text(
        0.02, 0.98, f"({label})", transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )


__all__ = [
    "publication_style",
    "_style_ax",
    "_colors_for_densities",
    "_add_colorbar",
    "_save",
    "_panel_label",
    "OKABE_ITO",
    "DENSITY_CMAP",
]