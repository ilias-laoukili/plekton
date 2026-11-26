# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Graphs with a mixing target per node attribute, through the native Python bindings."""

from __future__ import annotations

import numpy as np
import pytest

plekton = pytest.importorskip("plekton")


def test_marriage_layer_is_heterophilous_on_sex() -> None:
    """A matching inside households, every tie between a woman and a man."""
    n = 60
    graph = plekton.mixing_graph(
        n,
        attributes={"sex": ["female" if i % 2 == 0 else "male" for i in range(n)]},
        groups=[f"h{i // 6}" for i in range(n)],
        degrees=[1] * n,
        max_degree=1,
        mixing={"sex": {"within": 0.0}},
        seed=1,
    )
    assert isinstance(graph, plekton.Graph)
    assert graph.edges.shape == (n // 2, 2)
    assert graph.edges.dtype == np.int64
    assert all(int(a) % 2 != int(b) % 2 and int(a) // 6 == int(b) // 6 for a, b in graph.edges)
    assert graph.metadata["mixing"]["sex"]["r"] == pytest.approx(-1.0)


def test_friend_layer_is_homophilous_on_milieu() -> None:
    """Two targets met at once: homophilous on milieu, random on age."""
    n = 600
    graph = plekton.mixing_graph(
        n,
        attributes={
            "milieu": [("a", "b", "c")[i % 3] for i in range(n)],
            "age": [("young", "old")[(i // 3) % 2] for i in range(n)],
        },
        mean_degree=5,
        degree_spread=0.5,
        mixing={"milieu": {"within": 0.7}, "age": {"within": 0.5}},
        seed=2,
    )
    for report in graph.metadata["mixing"].values():
        assert report["within"] == pytest.approx(report["target"], abs=0.03)
    milieu = graph.metadata["mixing"]["milieu"]
    assert milieu["within"] > milieu["random"]


def test_bad_request_raises() -> None:
    """A target on an attribute that is not given is refused."""
    with pytest.raises(ValueError, match="not among the attributes"):
        plekton.mixing_graph(3, mean_degree=1, mixing={"sex": {"within": 0.5}})


def test_overlapping_communities_share_a_bridge() -> None:
    """A person with friends in two circles belongs to both."""
    circle_a = [(a, b) for a in range(8) for b in range(a + 1, 8)]
    circle_b = [(a, b) for a in range(8, 16) for b in range(a + 1, 16)]
    bridge = [(v, 16) for v in (0, 1, 2, 3, 8, 9, 10, 11)]
    graph = plekton.Graph(circle_a + circle_b + bridge, n_nodes=17)
    found = plekton.fit.overlapping_communities(graph, seed=1)
    assert sum(16 in c for c in found) == 2
    assert len(found) == 2


def test_clustering_target_closes_triangles() -> None:
    """Rewiring closes triangles toward the requested transitivity while keeping mixing."""
    n = 300
    milieu = [["a", "b", "c"][i % 3] for i in range(n)]
    plain = plekton.mixing_graph(
        n,
        attributes={"milieu": milieu},
        mean_degree=6.0,
        mixing={"milieu": {"within": 0.6}},
        seed=2,
    )
    clustered = plekton.mixing_graph(
        n,
        attributes={"milieu": milieu},
        mean_degree=6.0,
        mixing={"milieu": {"within": 0.6}},
        clustering=0.25,
        seed=2,
    )
    assert plain.metadata["clustering"]["transitivity"] < 0.1
    assert clustered.metadata["clustering"]["transitivity"] == pytest.approx(0.25, abs=0.03)
    assert clustered.metadata["mixing"]["milieu"]["within"] == pytest.approx(0.6, abs=0.04)


def test_directed_layer_has_popular_nodes() -> None:
    """Directed layer with popularity_spread produces heavy-tailed in-degrees."""
    n = 800
    graph = plekton.mixing_graph(n, mean_degree=5.0, directed=True, popularity_spread=1.5, seed=3)
    followers = np.bincount(graph.edges[:, 1], minlength=n)
    assert graph.directed
    assert graph.n_edges == 5 * n
    assert int(followers.max()) > 8 * 5


def test_strengths_come_with_the_ties() -> None:
    """Strengths are a read-only float64 edge attribute aligned with the edges."""
    graph = plekton.mixing_graph(
        100,
        mean_degree=4.0,
        strength={"mean": 0.5, "spread": 0.3},
        seed=1,
    )
    strengths = graph.edge_attrs["strength"]
    assert isinstance(strengths, np.ndarray)
    assert strengths.dtype == np.float64
    assert len(strengths) == graph.n_edges
    assert np.all((strengths > 0) & (strengths < 1))
    assert not strengths.flags.writeable


@pytest.mark.parametrize("kind", ["undirected", "directed"])
def test_mixing_output_is_a_simple_graph(kind: str) -> None:
    """mixing_graph output passes the full Graph validation: no loop, no repeat."""
    directed = kind == "directed"
    n = 300
    graph = plekton.mixing_graph(
        n,
        attributes={"milieu": [("a", "b", "c")[i % 3] for i in range(n)]},
        mean_degree=8.0,
        degree_spread=0.8,
        clustering=None if directed else 0.3,
        directed=directed,
        mixing={"milieu": {"within": 0.7}},
        seed=11,
    )
    # The generator wraps its output without the O(E log E) checks; re-check it in full.
    plekton.Graph(np.array(graph.edges), n_nodes=n, directed=directed)


def test_a_drawn_seed_is_recorded_and_reproduces_the_graph() -> None:
    """seed=None draws a seed, records it, and that seed regenerates the same graph."""
    first = plekton.mixing_graph(200, mean_degree=4.0)
    seed = first.metadata["seed"]
    again = plekton.mixing_graph(200, mean_degree=4.0, seed=seed)
    np.testing.assert_array_equal(first.edges, again.edges)
