# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Structural graph metrics over plain edge lists (wrapping ``plekton-metrics``).

Metrics that are mathematically undefined or skipped above a size threshold
return ``None`` rather than fabricating ``0.0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from plekton._plekton import metrics as _ext

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

_EDGE_PAIR_NDIM = 2

__all__ = [
    "assortativity",
    "attribute_mixing",
    "clustering",
    "components",
    "degree_ks",
    "degree_summary",
    "density",
    "homophily",
    "transitivity",
]


def _edges(edges: npt.ArrayLike) -> np.ndarray:
    """Coerce ``edges`` to a contiguous ``int64`` array of shape ``[E, 2]``."""
    arr = np.ascontiguousarray(edges, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, _EDGE_PAIR_NDIM), dtype=np.int64)
    if arr.ndim != _EDGE_PAIR_NDIM or arr.shape[1] != _EDGE_PAIR_NDIM:
        msg = f"edges must have shape [E, 2], got {arr.shape}"
        raise ValueError(msg)
    return arr


def _labels(labels: npt.ArrayLike) -> np.ndarray:
    """Coerce ``labels`` to a contiguous 1-D ``int64`` array."""
    arr = np.ascontiguousarray(labels, dtype=np.int64)
    if arr.ndim != 1:
        msg = f"labels must be 1-D, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def homophily(edges: npt.ArrayLike, labels: npt.ArrayLike) -> float | None:
    """Edge homophily (fraction of intra-community edges), or ``None`` if undefined."""
    val: float | None = _ext.homophily(_edges(edges), _labels(labels))
    return val


def clustering(n_nodes: int, edges: npt.ArrayLike) -> float | None:
    """Mean local clustering coefficient, or ``None`` above the size threshold."""
    val: float | None = _ext.clustering(n_nodes, _edges(edges))
    return val


def transitivity(n_nodes: int, edges: npt.ArrayLike) -> float | None:
    """Global transitivity (3 * triangles / open_triplets), or ``None`` if undefined."""
    val: float | None = _ext.transitivity(n_nodes, _edges(edges))
    return val


def assortativity(n_nodes: int, edges: npt.ArrayLike) -> float | None:
    """Newman's degree assortativity r, or ``None`` if edgeless or regular."""
    val: float | None = _ext.assortativity(n_nodes, _edges(edges))
    return val


def components(n_nodes: int, edges: npt.ArrayLike) -> int:
    """Count weakly connected components."""
    return int(_ext.components(n_nodes, _edges(edges)))


def density(n_nodes: int, n_edges: int) -> float:
    """Undirected graph density: ``2E / (N * (N - 1))``."""
    return float(_ext.density(n_nodes, n_edges))


def attribute_mixing(edges: npt.ArrayLike, labels: npt.ArrayLike) -> dict[str, Any] | None:
    """Attribute mixing: ``{"within", "random", "r"}``, or ``None`` if edgeless."""
    val: dict[str, Any] | None = _ext.attribute_mixing(_edges(edges), _labels(labels))
    return val


def degree_summary(n_nodes: int, edges: npt.ArrayLike) -> dict[str, Any] | None:
    """Degree summary: ``{"mean", "max", "p50", "p90", "p99", "gini"}``."""
    val: dict[str, Any] | None = _ext.degree_summary(n_nodes, _edges(edges))
    return val


def degree_ks(
    n_nodes_a: int,
    edges_a: npt.ArrayLike,
    n_nodes_b: int,
    edges_b: npt.ArrayLike,
) -> float | None:
    """Two-sample KS distance in ``[0, 1]`` between two graphs' degree distributions."""
    val: float | None = _ext.degree_ks(n_nodes_a, _edges(edges_a), n_nodes_b, _edges(edges_b))
    return val
