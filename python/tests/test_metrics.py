# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.metrics`` — structural metrics over a ``plekton.Graph``."""

from __future__ import annotations

import numpy as np
import pytest

plekton = pytest.importorskip("plekton")


def test_triangle_metrics() -> None:
    """A single triangle has density 1, transitivity 1, clustering 1, 1 component."""
    graph = plekton.Graph([[0, 1], [1, 2], [0, 2]], n_nodes=3)
    assert plekton.metrics.density(graph) == pytest.approx(1.0)
    assert plekton.metrics.components(graph) == 1
    assert plekton.metrics.transitivity(graph) == pytest.approx(1.0)
    assert plekton.metrics.clustering(graph) == pytest.approx(1.0)
    # Regular graph: assortativity denominator is zero -> None (never fabricated 0.0)
    assert plekton.metrics.assortativity(graph) is None


def test_isolated_nodes_count() -> None:
    """Nodes without edges still count: as components and in the density."""
    graph = plekton.Graph([[0, 1]], n_nodes=4)
    assert plekton.metrics.components(graph) == 3
    assert plekton.metrics.density(graph) == pytest.approx(2.0 / (4 * 3))


def test_an_edge_array_is_accepted_with_n_nodes() -> None:
    """A raw (E, 2) array gives the same answer as the Graph it describes."""
    edges = np.array([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int64)
    graph = plekton.Graph(edges, n_nodes=5)
    assert plekton.metrics.degree_summary(edges, n_nodes=5) == plekton.metrics.degree_summary(
        graph,
    )


def test_n_nodes_is_required_with_an_array_and_refused_with_a_graph() -> None:
    """n_nodes cannot be guessed from an array, and cannot contradict a Graph."""
    edges = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(TypeError, match="n_nodes"):
        plekton.metrics.clustering(edges)
    with pytest.raises(TypeError, match="n_nodes comes from the Graph"):
        plekton.metrics.clustering(plekton.Graph(edges, n_nodes=2), n_nodes=2)


def test_homophily_and_attribute_mixing() -> None:
    """Homophily and attribute_mixing agree on within-category share."""
    graph = plekton.Graph([[0, 1], [2, 3], [0, 2]], n_nodes=4, labels=[0, 0, 1, 1])
    assert plekton.metrics.homophily(graph) == pytest.approx(2.0 / 3.0)
    mix = plekton.metrics.attribute_mixing(graph)
    assert mix is not None
    assert mix["within"] == pytest.approx(2.0 / 3.0)
    assert mix["random"] == pytest.approx(0.5)
    assert mix["r"] == pytest.approx((2.0 / 3.0 - 0.5) / 0.5)
    # Explicit labels override the graph's own.
    assert plekton.metrics.homophily(graph, labels=[0, 1, 0, 1]) == pytest.approx(1.0 / 3.0)


def test_homophily_needs_labels() -> None:
    """Without labels on the graph or in the call, there is nothing to measure."""
    graph = plekton.Graph([[0, 1]], n_nodes=2)
    with pytest.raises(ValueError, match="no labels"):
        plekton.metrics.homophily(graph)


def test_a_single_community_is_fully_homophilous() -> None:
    """Every node in community 0 is one community, not "unlabelled": every edge stays inside."""
    graph = plekton.Graph([[0, 1], [1, 2]], n_nodes=3, labels=[0, 0, 0])
    assert plekton.metrics.homophily(graph) == 1.0
    edgeless = plekton.Graph(np.zeros((0, 2), dtype=np.int64), n_nodes=3, labels=[0, 1, 1])
    assert plekton.metrics.homophily(edgeless) is None


@pytest.mark.parametrize("n_nodes", [0, 1])
def test_density_is_undefined_below_two_nodes(n_nodes: int) -> None:
    """No pair of nodes, nothing for the edge count to be a fraction of: ``None``, not ``0.0``."""
    graph = plekton.Graph(np.zeros((0, 2), dtype=np.int64), n_nodes=n_nodes)
    assert plekton.metrics.density(graph) is None


def test_an_edgeless_graph_has_density_zero() -> None:
    """With two nodes or more, no edge is a real density of 0."""
    graph = plekton.Graph(np.zeros((0, 2), dtype=np.int64), n_nodes=2)
    assert plekton.metrics.density(graph) == 0.0


def test_clustering_is_undefined_without_nodes() -> None:
    """A mean over no node is ``None``; nodes without edges each count 0, as documented."""
    empty = plekton.Graph(np.zeros((0, 2), dtype=np.int64), n_nodes=0)
    assert plekton.metrics.clustering(empty) is None
    edgeless = plekton.Graph(np.zeros((0, 2), dtype=np.int64), n_nodes=3)
    assert plekton.metrics.clustering(edgeless) == 0.0


def test_directed_graphs_only_reach_arc_metrics() -> None:
    """Homophily counts arcs; metrics with undirected semantics refuse a directed graph."""
    arcs = plekton.Graph([[0, 1], [1, 0], [1, 2]], n_nodes=3, labels=[0, 0, 1], directed=True)
    assert plekton.metrics.homophily(arcs) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="undirected"):
        plekton.metrics.clustering(arcs)


def test_degree_summary_and_ks_distance() -> None:
    """Degree summary captures hubs and KS distance is 0 for identical distributions."""
    star = plekton.Graph([[0, 1], [0, 2], [0, 3], [0, 4]], n_nodes=5)
    summary = plekton.metrics.degree_summary(star)
    assert summary is not None
    assert summary["max"] == 4
    assert summary["p50"] == 1
    assert summary["gini"] > 0.0

    assert plekton.metrics.degree_ks(star, star) == pytest.approx(0.0)
    ring = plekton.Graph([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]], n_nodes=5)
    ks = plekton.metrics.degree_ks(star, ring)
    assert ks is not None
    assert ks > 0.5
    raw = plekton.metrics.degree_ks(star.edges, ring.edges, n_nodes_a=5, n_nodes_b=5)
    assert raw == pytest.approx(ks)
