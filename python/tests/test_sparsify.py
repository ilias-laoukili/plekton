# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Smoke tests for :mod:`plekton.sparsify`.

The Rust core carries the algorithm tests; these cover the seam the Rust suite cannot see —
the Python wrapper, its NumPy coercion, and the invariants a caller depends on.

No torch here: sparsify is pure Rust plus NumPy and must stay importable in the torch-free
environment.
"""

from __future__ import annotations

import numpy as np
import pytest
from plekton import sparsify

# An edge has two endpoints.
_EDGE_ENDPOINTS = 2
# The pendant node connectivity preservation must keep attached.
_PENDANT_NODE = 3

# A triangle plus a pendant: enough structure for every metric to say something different,
# small enough to reason about by hand.
EDGES = np.array([[0, 1], [0, 2], [1, 2], [0, 3]], dtype=np.int64)
N_NODES = 4


def test_similarity_scores_returns_one_float_per_edge() -> None:
    """Similarity scores returns one float per edge."""
    scores = sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric="jaccard")

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(EDGES),)
    assert scores.dtype == np.float64


@pytest.mark.parametrize("metric", ["jaccard", "adamic_adar", "degree_product"])
def test_every_structural_metric_scores_the_whole_edge_list(metric: str) -> None:
    """Every structural metric scores the whole edge list."""
    scores = sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric=metric)
    assert scores.shape == (len(EDGES),)
    assert np.all(np.isfinite(scores)), f"{metric} produced a non-finite score"


def test_the_pendant_edge_scores_lowest_under_jaccard() -> None:
    """The pendant edge scores lowest under jaccard."""
    # (0,3) is the only edge whose endpoints share no neighbour, so it must rank last —
    # this is the property a sparsifier is actually relied on for.
    scores = sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric="jaccard")
    assert scores[3] == pytest.approx(0.0)
    assert scores[:3].min() > scores[3]


def test_select_mask_is_boolean_and_edge_aligned() -> None:
    """Select mask is boolean and edge aligned."""
    scores = sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric="jaccard")
    keep = sparsify.select_mask(EDGES, N_NODES, scores, "threshold", retention_ratio=0.5)

    assert keep.dtype == np.bool_
    assert keep.shape == (len(EDGES),)
    # The mask must be usable exactly as documented.
    assert EDGES[keep].shape[1] == _EDGE_ENDPOINTS


def test_connectivity_preservation_leaves_no_node_stranded() -> None:
    """Connectivity preservation leaves no node stranded."""
    scores = sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric="jaccard")
    keep = sparsify.select_mask(
        EDGES,
        N_NODES,
        scores,
        "threshold",
        retention_ratio=0.25,
        preserve_connectivity=True,
    )

    # Node 3 hangs off the lowest-scoring edge, so an unguarded threshold would isolate it.
    survivors = set(EDGES[keep].flatten().tolist())
    assert _PENDANT_NODE in survivors, "preserve_connectivity must keep the pendant attached"


def test_approx_effective_resistance_is_finite_and_edge_aligned() -> None:
    """Approx effective resistance is finite and edge aligned."""
    resistance = sparsify.approx_effective_resistance(EDGES, N_NODES, seed=42)
    assert resistance.shape == (len(EDGES),)
    assert np.all(np.isfinite(resistance))


def test_a_seeded_run_is_reproducible() -> None:
    """A seeded run is reproducible."""
    # The approximation is randomised; a fixed seed must still give a fixed answer, or no
    # result computed with it is comparable to another.
    first = sparsify.approx_effective_resistance(EDGES, N_NODES, seed=7)
    second = sparsify.approx_effective_resistance(EDGES, N_NODES, seed=7)
    np.testing.assert_array_equal(first, second)


def test_metric_backbone_mask_is_boolean_and_edge_aligned() -> None:
    """Metric backbone mask is boolean and edge aligned."""
    scores = sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric="jaccard")
    keep = sparsify.metric_backbone_mask(EDGES, N_NODES, scores, distances_are_similarity=True)
    assert keep.dtype == np.bool_
    assert keep.shape == (len(EDGES),)


def test_list_input_is_coerced_like_an_array() -> None:
    """List input is coerced like an array."""
    # The docstring promises "any array-like"; a caller passing a plain list should not get
    # a different answer from one passing an array.
    from_list = sparsify.similarity_scores(
        [[0, 1], [0, 2], [1, 2], [0, 3]],
        n_nodes=N_NODES,
        metric="jaccard",
    )
    from_array = sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric="jaccard")
    np.testing.assert_array_equal(from_list, from_array)


def test_an_out_of_range_endpoint_is_refused() -> None:
    """An out of range endpoint is refused."""
    # Silently accepting it would index past the node set and score a node that is not there.
    bad = np.array([[0, 99]], dtype=np.int64)
    with pytest.raises((IndexError, ValueError)):
        sparsify.similarity_scores(bad, n_nodes=N_NODES, metric="jaccard")


def test_an_unknown_metric_is_refused_rather_than_defaulted() -> None:
    """An unknown metric is refused rather than defaulted."""
    with pytest.raises((ValueError, RuntimeError)):
        sparsify.similarity_scores(EDGES, n_nodes=N_NODES, metric="not_a_metric")


def test_an_empty_edge_list_is_handled() -> None:
    """An empty edge list is handled."""
    empty = np.empty((0, 2), dtype=np.int64)
    scores = sparsify.similarity_scores(empty, n_nodes=0, metric="jaccard")
    assert scores.shape == (0,)
