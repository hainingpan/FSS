"""Finite-size scaling data collapse toolkit."""

from importlib import metadata

from .data_collapse import (
    DataCollapse,
    bootstrapping,
    extrapolate_fitting,
    grid_search,
    optimal_df,
    plot_chi2_ratio,
    plot_extrapolate_fitting,
)

__all__ = [
    "DataCollapse",
    "bootstrapping",
    "extrapolate_fitting",
    "grid_search",
    "optimal_df",
    "plot_chi2_ratio",
    "plot_extrapolate_fitting",
]

try:
    __version__ = metadata.version("FSS")
except metadata.PackageNotFoundError:  # pragma: no cover - local development fallback
    __version__ = "0.0.0"
