# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.fit`` — DC-SBM estimation, label propagation, SLPA."""

from __future__ import annotations

import numpy as np
import pytest

plekton = pytest.importorskip("plekton")


def test_fit_blockmodel_roundtrip_and_regeneration() -> None:
    """Generate → fit → regenerate preserves block sizes, matrix shape, and scale."""
    n = 300
    graph = plekton.generate_sbm(n, 3, 0.15, 0.01, seed=7)

    fitted = plekton.fit.fit_blockmodel(graph, attribute="community")  # graph.labels
    assert fitted["provenance"] == "declared"
    assert fitted["seed"] is None
    assert fitted["n_blocks"] == 3
    assert fitted["block_sizes"] == [100, 100, 100]
    assert fitted["b_matrix"].shape == (3, 3)
    assert fitted["theta"].shape == (n,)
    assert np.diag(fitted["b_matrix"]).mean() > 0.10

    regenerated = plekton.generate_sbm_from_fit(
        n,
        fitted["block_sizes"],
        fitted["b_matrix"],
        theta=fitted["theta"],
        seed=8,
    )
    assert regenerated.labels is not None
    assert regenerated.labels.shape == (n,)
    assert abs(regenerated.n_edges - graph.n_edges) / graph.n_edges < 0.25


def test_fit_blockmodel_detected_partition() -> None:
    """labels='detect' runs seeded label propagation and records provenance='detected'."""
    n = 300
    graph = plekton.generate_sbm(n, 2, 0.20, 0.004, seed=31)

    fitted = plekton.fit.fit_blockmodel(graph, labels="detect", seed=5, max_rounds=30)
    assert fitted["provenance"] == "detected"
    assert fitted["seed"] == 5
    assert fitted["n_blocks"] >= 2
    assert fitted["blocks_per_node"] < 0.1


def test_a_drawn_detection_seed_is_recorded_and_reproduces() -> None:
    """With seed=None the seed that ran is returned, and replaying it gives the same fit."""
    graph = plekton.generate_sbm(200, 2, 0.2, 0.01, seed=3)
    first = plekton.fit.fit_blockmodel(graph, labels="detect")
    again = plekton.fit.fit_blockmodel(graph, labels="detect", seed=first["seed"])
    np.testing.assert_array_equal(first["labels"], again["labels"])


def test_fit_blockmodel_never_infers_the_partition_silently() -> None:
    """No labels and no detection request is an error, not a default."""
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match="labels='detect'"):
        plekton.fit.fit_blockmodel(plekton.Graph(edges, n_nodes=4))
    with pytest.raises(ValueError, match="labels='detect'"):
        plekton.fit.fit_blockmodel(edges, n_nodes=4)
    with pytest.raises(ValueError, match="'detect'"):
        plekton.fit.fit_blockmodel(edges, n_nodes=4, labels="louvain")
    with pytest.raises(ValueError, match="seed only applies"):
        plekton.fit.fit_blockmodel(edges, n_nodes=4, labels=[0, 0, 1, 1], seed=1)


def test_label_propagation_is_deterministic() -> None:
    """Same seed produces identical detected labels."""
    graph = plekton.Graph([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], n_nodes=6)
    l1, conv1 = plekton.fit.label_propagation(graph, seed=42)
    l2, conv2 = plekton.fit.label_propagation(graph, seed=42)
    np.testing.assert_array_equal(l1, l2)
    assert conv1 == conv2
    assert len(set(l1.tolist())) == 2


def test_fit_overlapping_communities() -> None:
    """SLPA detects a bridge node belonging to two overlapping cliques."""
    circle_a = [[a, b] for a in range(8) for b in range(a + 1, 8)]
    circle_b = [[a, b] for a in range(8, 16) for b in range(a + 1, 16)]
    bridge = [[v, 16] for v in (0, 1, 2, 3, 8, 9, 10, 11)]
    graph = plekton.Graph(circle_a + circle_b + bridge, n_nodes=17)
    found = plekton.fit.overlapping_communities(graph, seed=1)
    assert sum(16 in c for c in found) == 2
    assert len(found) == 2


def test_slpa_refuses_an_id_outside_the_graph() -> None:
    """An out-of-range id is an error, not an edge SLPA quietly skips."""
    with pytest.raises(IndexError, match="out of range"):
        plekton.fit.overlapping_communities(np.array([[0, 1], [1, 2], [2, 7]]), n_nodes=3)
