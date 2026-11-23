# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.fit`` — DC-SBM estimation, label propagation, SLPA."""

from __future__ import annotations

import numpy as np
import pytest

gg = pytest.importorskip("plekton")


def _canonical_edges(edge_index: np.ndarray) -> np.ndarray:
    mask = edge_index[0] < edge_index[1]
    return np.ascontiguousarray(edge_index[:, mask].T, dtype=np.int64)


def test_fit_blockmodel_roundtrip_and_regeneration() -> None:
    """Generate → fit → regenerate preserves block sizes, matrix shape, and scale."""
    n = 300
    edge_index, labels, _ = gg.generate_sbm(n, 3, 0.15, 0.01, seed=7)
    edges = _canonical_edges(edge_index)

    fitted = gg.fit.fit_blockmodel(edges, n_nodes=n, labels=labels, attribute="community")
    assert fitted["provenance"] == "declared"
    assert fitted["n_blocks"] == 3
    assert fitted["block_sizes"] == [100, 100, 100]
    assert fitted["b_matrix"].shape == (3, 3)
    assert fitted["theta"].shape == (n,)
    assert np.diag(fitted["b_matrix"]).mean() > 0.10

    ei2, labels2, _ = gg.generate_sbm_from_fit(
        n,
        fitted["block_sizes"],
        fitted["b_matrix"],
        fitted["theta"],
        seed=8,
    )
    assert labels2.shape == (n,)
    orig_edges = edges.shape[0]
    regen_edges = ei2.shape[1] // 2
    assert abs(regen_edges - orig_edges) / orig_edges < 0.25


def test_fit_blockmodel_detected_partition() -> None:
    """Omitting labels runs seeded label propagation and records provenance='detected'."""
    n = 300
    edge_index, _, _ = gg.generate_sbm(n, 2, 0.20, 0.004, seed=31)
    edges = _canonical_edges(edge_index)

    fitted = gg.fit.fit_blockmodel(edges, n_nodes=n, seed=5, max_rounds=30)
    assert fitted["provenance"] == "detected"
    assert fitted["n_blocks"] >= 2
    assert fitted["blocks_per_node"] < 0.1


def test_label_propagation_is_deterministic() -> None:
    """Same seed produces identical detected labels."""
    edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int64)
    l1, conv1 = gg.fit.label_propagation(edges, n_nodes=6, seed=42)
    l2, conv2 = gg.fit.label_propagation(edges, n_nodes=6, seed=42)
    np.testing.assert_array_equal(l1, l2)
    assert conv1 == conv2
    assert len(set(l1.tolist())) == 2


def test_fit_overlapping_communities() -> None:
    """SLPA detects a bridge node belonging to two overlapping cliques."""
    circle_a = [[a, b] for a in range(8) for b in range(a + 1, 8)]
    circle_b = [[a, b] for a in range(8, 16) for b in range(a + 1, 16)]
    bridge = [[v, 16] for v in (0, 1, 2, 3, 8, 9, 10, 11)]
    found = gg.fit.overlapping_communities(17, circle_a + circle_b + bridge, seed=1)
    assert sum(16 in c for c in found) == 2
    assert len(found) == 2
