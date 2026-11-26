# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Structural graph metrics (wrapping ``plekton-metrics``).

Every function takes a :class:`plekton.Graph` first. An ``(E, 2)`` edge array is
also accepted when ``n_nodes`` is passed by keyword; it is validated like a
``Graph`` (no ``(2, E)`` layout, no edge in both directions, ids in range,
integer dtype). Metrics that are mathematically undefined or skipped above a
size threshold return ``None`` rather than fabricating ``0.0``::

    import plekton

    graph = plekton.generate_sbm(500, 4, 0.1, 0.01, seed=0)
    plekton.metrics.homophily(graph)       # uses graph.labels
    plekton.metrics.degree_summary(graph)  # {"mean", "max", "p50", "p90", "p99", "gini"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from plekton._graph import Graph, as_graph, node_labels
from plekton._plekton import metrics as _ext

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

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


def homophily(
    graph: Graph | npt.ArrayLike,
    *,
    labels: npt.ArrayLike | None = None,
    n_nodes: int | None = None,
) -> float | None:
    """Edge homophily (fraction of intra-community edges), or ``None`` if undefined.

    ``labels`` defaults to ``graph.labels``. Directed graphs count arcs.
    """
    g = as_graph(graph, n_nodes, caller="metrics.homophily", allow_directed=True)
    val: float | None = _ext.homophily(g.edges, node_labels(g, labels, caller="metrics.homophily"))
    return val


def attribute_mixing(
    graph: Graph | npt.ArrayLike,
    *,
    labels: npt.ArrayLike | None = None,
    n_nodes: int | None = None,
) -> dict[str, Any] | None:
    """Attribute mixing ``{"within", "random", "r"}``, or ``None`` if edgeless.

    ``labels`` defaults to ``graph.labels``. Directed graphs count arcs.
    """
    g = as_graph(graph, n_nodes, caller="metrics.attribute_mixing", allow_directed=True)
    lbls = node_labels(g, labels, caller="metrics.attribute_mixing")
    val: dict[str, Any] | None = _ext.attribute_mixing(g.edges, lbls)
    return val


def clustering(graph: Graph | npt.ArrayLike, *, n_nodes: int | None = None) -> float | None:
    """Mean local clustering coefficient, or ``None`` above the size threshold."""
    g = as_graph(graph, n_nodes, caller="metrics.clustering")
    val: float | None = _ext.clustering(g.n_nodes, g.edges)
    return val


def transitivity(graph: Graph | npt.ArrayLike, *, n_nodes: int | None = None) -> float | None:
    """Global transitivity (3 * triangles / open_triplets), or ``None`` if undefined."""
    g = as_graph(graph, n_nodes, caller="metrics.transitivity")
    val: float | None = _ext.transitivity(g.n_nodes, g.edges)
    return val


def assortativity(graph: Graph | npt.ArrayLike, *, n_nodes: int | None = None) -> float | None:
    """Newman's degree assortativity r, or ``None`` if edgeless or regular."""
    g = as_graph(graph, n_nodes, caller="metrics.assortativity")
    val: float | None = _ext.assortativity(g.n_nodes, g.edges)
    return val


def components(graph: Graph | npt.ArrayLike, *, n_nodes: int | None = None) -> int:
    """Count connected components, isolated nodes included."""
    g = as_graph(graph, n_nodes, caller="metrics.components")
    return int(_ext.components(g.n_nodes, g.edges))


def density(graph: Graph | npt.ArrayLike, *, n_nodes: int | None = None) -> float:
    """Undirected graph density: ``2E / (N * (N - 1))``."""
    g = as_graph(graph, n_nodes, caller="metrics.density")
    return float(_ext.density(g.n_nodes, g.n_edges))


def degree_summary(
    graph: Graph | npt.ArrayLike,
    *,
    n_nodes: int | None = None,
) -> dict[str, Any] | None:
    """Degree summary: ``{"mean", "max", "p50", "p90", "p99", "gini"}``."""
    g = as_graph(graph, n_nodes, caller="metrics.degree_summary")
    val: dict[str, Any] | None = _ext.degree_summary(g.n_nodes, g.edges)
    return val


def degree_ks(
    graph_a: Graph | npt.ArrayLike,
    graph_b: Graph | npt.ArrayLike,
    *,
    n_nodes_a: int | None = None,
    n_nodes_b: int | None = None,
) -> float | None:
    """Two-sample KS distance in ``[0, 1]`` between two graphs' degree distributions."""
    a = as_graph(graph_a, n_nodes_a, caller="metrics.degree_ks")
    b = as_graph(graph_b, n_nodes_b, caller="metrics.degree_ks")
    val: float | None = _ext.degree_ks(a.n_nodes, a.edges, b.n_nodes, b.edges)
    return val
