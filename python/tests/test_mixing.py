# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Graphs with a mixing target per node attribute, through the native Python bindings."""

from __future__ import annotations

import numpy as np
import pytest

gg = pytest.importorskip("plekton")


def test_marriage_layer_is_heterophilous_on_sex() -> None:
    """A matching inside households, every tie between a woman and a man."""
    n = 60
    graph = gg.mixing_graph(
        n,
        attributes={"sex": ["female" if i % 2 == 0 else "male" for i in range(n)]},
        groups=[f"h{i // 6}" for i in range(n)],
        degrees=[1] * n,
        max_degree=1,
        mixing={"sex": {"within": 0.0}},
        seed=1,
    )
    assert isinstance(graph["edges"], np.ndarray)
    assert graph["edges"].shape == (n // 2, 2)
    assert graph["edges"].dtype == np.int64
    assert all(int(a) % 2 != int(b) % 2 and int(a) // 6 == int(b) // 6 for a, b in graph["edges"])
    assert graph["mixing"]["sex"]["r"] == pytest.approx(-1.0)


def test_friend_layer_is_homophilous_on_milieu() -> None:
    """Two targets met at once: homophilous on milieu, random on age."""
    n = 600
    graph = gg.mixing_graph(
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
    for report in graph["mixing"].values():
        assert report["within"] == pytest.approx(report["target"], abs=0.03)
    assert graph["mixing"]["milieu"]["within"] > graph["mixing"]["milieu"]["random"]


def test_bad_request_raises() -> None:
    """A target on an attribute that is not given is refused."""
    with pytest.raises(ValueError, match="not among the attributes"):
        gg.mixing_graph(3, mean_degree=1, mixing={"sex": {"within": 0.5}})


def test_overlapping_communities_share_a_bridge() -> None:
    """A person with friends in two circles belongs to both."""
    circle_a = [(a, b) for a in range(8) for b in range(a + 1, 8)]
    circle_b = [(a, b) for a in range(8, 16) for b in range(a + 1, 16)]
    bridge = [(v, 16) for v in (0, 1, 2, 3, 8, 9, 10, 11)]
    found = gg.overlapping_communities(17, circle_a + circle_b + bridge, seed=1)
    assert sum(16 in c for c in found) == 2
    assert len(found) == 2


def test_clustering_target_closes_triangles() -> None:
    """Rewiring closes triangles toward the requested transitivity while keeping mixing."""
    n = 300
    milieu = [["a", "b", "c"][i % 3] for i in range(n)]
    plain = gg.mixing_graph(
        n,
        attributes={"milieu": milieu},
        mean_degree=6.0,
        mixing={"milieu": {"within": 0.6}},
        seed=2,
    )
    clustered = gg.mixing_graph(
        n,
        attributes={"milieu": milieu},
        mean_degree=6.0,
        mixing={"milieu": {"within": 0.6}},
        clustering=0.25,
        seed=2,
    )
    assert plain["clustering"]["transitivity"] < 0.1
    assert clustered["clustering"]["transitivity"] == pytest.approx(0.25, abs=0.03)
    assert clustered["mixing"]["milieu"]["within"] == pytest.approx(0.6, abs=0.04)


def test_directed_layer_has_popular_nodes() -> None:
    """Directed layer with popularity_spread produces heavy-tailed in-degrees."""
    n = 800
    graph = gg.mixing_graph(n, mean_degree=5.0, directed=True, popularity_spread=1.5, seed=3)
    followers = [0] * n
    for _, to in graph["edges"]:
        followers[int(to)] += 1
    assert graph["directed"]
    assert len(graph["edges"]) == 5 * n
    assert max(followers) > 8 * 5


def test_strengths_come_with_the_ties() -> None:
    """Strengths are returned as a float64 NumPy array aligned with edges."""
    graph = gg.mixing_graph(100, mean_degree=4.0, strength={"mean": 0.5, "spread": 0.3}, seed=1)
    assert isinstance(graph["strengths"], np.ndarray)
    assert graph["strengths"].dtype == np.float64
    assert len(graph["strengths"]) == len(graph["edges"])
    assert np.all((graph["strengths"] > 0) & (graph["strengths"] < 1))
