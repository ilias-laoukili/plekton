# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Smoke tests for :mod:`plekton.sparsify`.

The Rust core carries the algorithm tests; these cover the seam the Rust suite cannot see —
the Python wrapper, its Graph handling, and the invariants a caller depends on.

No torch here: sparsify is pure Rust plus NumPy and must stay importable in the torch-free
environment.
"""

from __future__ import annotations

import numpy as np
import plekton
import pytest
from plekton import sparsify

# The pendant node the isolates guard must keep attached.
_PENDANT_NODE = 3

# A 40-node path: what the isolates guard leaves of it shows that it is no connectivity
# guarantee.
_PATH_NODES = 40
_PATH_PAIRS = _PATH_NODES // 2

# A triangle plus a pendant: enough structure for every metric to say something different,
# small enough to reason about by hand.
EDGES = np.array([[0, 1], [0, 2], [1, 2], [0, 3]], dtype=np.int64)
N_NODES = 4
GRAPH = plekton.Graph(EDGES, n_nodes=N_NODES)


def test_similarity_scores_returns_one_float_per_edge() -> None:
    """Similarity scores returns one float per edge."""
    scores = sparsify.similarity_scores(GRAPH, "jaccard")

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (GRAPH.n_edges,)
    assert scores.dtype == np.float64


@pytest.mark.parametrize("metric", ["jaccard", "adamic_adar", "degree_product"])
def test_every_structural_metric_scores_the_whole_edge_list(metric: str) -> None:
    """Every structural metric scores the whole edge list."""
    scores = sparsify.similarity_scores(GRAPH, metric)
    assert scores.shape == (GRAPH.n_edges,)
    assert np.all(np.isfinite(scores)), f"{metric} produced a non-finite score"


def test_the_pendant_edge_scores_lowest_under_jaccard() -> None:
    """The pendant edge scores lowest under jaccard."""
    # (0,3) is the only edge whose endpoints share no neighbour, so it must rank last —
    # this is the property a sparsifier is actually relied on for.
    scores = sparsify.similarity_scores(GRAPH, "jaccard")
    assert scores[3] == pytest.approx(0.0)
    assert scores[:3].min() > scores[3]


def test_feature_cosine_uses_the_graph_features() -> None:
    """feature_cosine reads graph.x by default and refuses a graph without features."""
    x = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    with_x = plekton.Graph(EDGES, n_nodes=N_NODES, x=x)
    np.testing.assert_array_equal(
        sparsify.similarity_scores(with_x, "feature_cosine"),
        sparsify.similarity_scores(GRAPH, "feature_cosine", features=x),
    )
    with pytest.raises(ValueError, match="features"):
        sparsify.similarity_scores(GRAPH, "feature_cosine")


def test_select_mask_is_boolean_and_edge_aligned() -> None:
    """Select mask is boolean, edge aligned, and applies through edge_subgraph."""
    scores = sparsify.similarity_scores(GRAPH, "jaccard")
    keep = sparsify.select_mask(GRAPH, scores, "threshold", retention_ratio=0.5)

    assert keep.dtype == np.bool_
    assert keep.shape == (GRAPH.n_edges,)
    sparse = GRAPH.edge_subgraph(keep)
    assert sparse.n_nodes == GRAPH.n_nodes
    assert sparse.n_edges == int(keep.sum())


def test_the_isolates_guard_leaves_no_node_stranded() -> None:
    """The isolates guard, on by default, leaves no node stranded."""
    scores = sparsify.similarity_scores(GRAPH, "jaccard")
    for keep in (
        sparsify.select_mask(GRAPH, scores, "threshold", retention_ratio=0.25, guard="isolates"),
        sparsify.select_mask(GRAPH, scores, "threshold", retention_ratio=0.25),
    ):
        # Node 3 hangs off the lowest-scoring edge, so an unguarded threshold would isolate it.
        survivors = set(GRAPH.edges[keep].flatten().tolist())
        assert _PENDANT_NODE in survivors, "the isolates guard must keep the pendant attached"


def test_without_the_guard_the_pendant_node_is_isolated() -> None:
    """With ``guard="none"`` the selection is kept as the threshold made it."""
    scores = sparsify.similarity_scores(GRAPH, "jaccard")
    keep = sparsify.select_mask(GRAPH, scores, "threshold", retention_ratio=0.25, guard="none")
    assert _PENDANT_NODE not in set(GRAPH.edges[keep].flatten().tolist())


def test_the_isolates_guard_does_not_keep_a_path_connected() -> None:
    """No node is isolated, yet the path falls apart: the guard does not keep connectivity."""
    path = plekton.Graph([(i, i + 1) for i in range(_PATH_NODES - 1)], n_nodes=_PATH_NODES)
    # High scores on (0, 1), (2, 3), …, (38, 39): the threshold keeps 19 of these 20 edges
    # and the guard gives the 20th back, so every other edge of the path survives.
    scores = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(path.n_edges)])
    keep = sparsify.select_mask(path, scores, "threshold", retention_ratio=0.5, guard="isolates")
    sparse = path.edge_subgraph(keep)

    degrees = np.bincount(sparse.edges.ravel(), minlength=_PATH_NODES)
    assert (degrees > 0).all(), "a node was left isolated"
    assert plekton.metrics.components(sparse) == _PATH_PAIRS
    # The restored edge comes on top of the threshold's budget: 20 edges kept, not 19.
    unguarded = sparsify.select_mask(path, scores, "threshold", retention_ratio=0.5, guard="none")
    assert int(unguarded.sum()) == _PATH_PAIRS - 1
    assert int(keep.sum()) == _PATH_PAIRS


def test_an_unknown_guard_is_refused() -> None:
    """A guard that does not exist is refused, not ignored."""
    scores = sparsify.similarity_scores(GRAPH, "jaccard")
    with pytest.raises(ValueError, match=r"unknown guard 'forest' \(expected none, isolates\)"):
        sparsify.select_mask(GRAPH, scores, "threshold", retention_ratio=0.5, guard="forest")


def test_the_old_keyword_is_gone() -> None:
    """``preserve_connectivity`` named a guarantee the guard never gave; it no longer exists."""
    scores = sparsify.similarity_scores(GRAPH, "jaccard")
    with pytest.raises(TypeError, match="preserve_connectivity"):
        sparsify.select_mask(
            GRAPH,
            scores,
            "threshold",
            retention_ratio=0.5,
            preserve_connectivity=True,
        )


def test_approx_effective_resistance_is_finite_and_edge_aligned() -> None:
    """Approx effective resistance is finite and edge aligned."""
    resistance = sparsify.approx_effective_resistance(GRAPH, seed=42)
    assert resistance.shape == (GRAPH.n_edges,)
    assert np.all(np.isfinite(resistance))


def test_a_seeded_run_is_reproducible() -> None:
    """A seeded run is reproducible."""
    # The approximation is randomised; a fixed seed must still give a fixed answer, or no
    # result computed with it is comparable to another.
    first = sparsify.approx_effective_resistance(GRAPH, seed=7)
    second = sparsify.approx_effective_resistance(GRAPH, seed=7)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("epsilon", [0.0, -0.3, float("nan"), float("inf")])
def test_an_invalid_epsilon_is_refused(epsilon: float) -> None:
    """An invalid epsilon is refused."""
    # epsilon = 0 used to overflow the projection and abort the whole interpreter;
    # NaN and inf silently collapsed it to a single projection.
    with pytest.raises(ValueError, match="epsilon"):
        sparsify.approx_effective_resistance(GRAPH, epsilon=epsilon)


def test_a_vanishing_epsilon_is_refused_rather_than_overflowed() -> None:
    """A vanishing epsilon is refused rather than overflowed."""
    # k = 24 ln(n) / epsilon^2 saturates here, so the n x k projection cannot exist. A
    # merely small epsilon would request a real, huge allocation: never test that one.
    with pytest.raises(ValueError, match="projection"):
        sparsify.approx_effective_resistance(GRAPH, epsilon=1e-10)


def test_metric_backbone_mask_is_boolean_and_edge_aligned() -> None:
    """Metric backbone mask is boolean and edge aligned."""
    scores = sparsify.similarity_scores(GRAPH, "jaccard")
    keep = sparsify.metric_backbone_mask(GRAPH, scores, distances_are_similarity=True)
    assert keep.dtype == np.bool_
    assert keep.shape == (GRAPH.n_edges,)


def test_list_input_is_coerced_like_an_array() -> None:
    """List input is coerced like an array."""
    # An edge array-like with n_nodes= must give the same answer as the Graph it describes.
    from_list = sparsify.similarity_scores(
        [[0, 1], [0, 2], [1, 2], [0, 3]],
        "jaccard",
        n_nodes=N_NODES,
    )
    from_graph = sparsify.similarity_scores(GRAPH, "jaccard")
    np.testing.assert_array_equal(from_list, from_graph)


def test_an_out_of_range_endpoint_is_refused() -> None:
    """An out of range endpoint is refused."""
    # Silently accepting it would index past the node set and score a node that is not there.
    bad = np.array([[0, 99]], dtype=np.int64)
    with pytest.raises(IndexError, match="out of range"):
        sparsify.similarity_scores(bad, "jaccard", n_nodes=N_NODES)


def test_an_unknown_metric_is_refused_rather_than_defaulted() -> None:
    """An unknown metric is refused rather than defaulted."""
    with pytest.raises(ValueError, match="unknown metric"):
        sparsify.similarity_scores(GRAPH, "not_a_metric")


def test_an_empty_edge_list_is_handled() -> None:
    """An empty edge list is handled."""
    empty = plekton.Graph(np.empty((0, 2), dtype=np.int64), n_nodes=0)
    scores = sparsify.similarity_scores(empty, "jaccard")
    assert scores.shape == (0,)


def test_a_directed_graph_is_refused() -> None:
    """Sparsification has undirected semantics: a directed graph is refused, not reinterpreted."""
    arcs = plekton.Graph([[0, 1], [1, 0]], n_nodes=2, directed=True)
    with pytest.raises(ValueError, match="undirected"):
        sparsify.similarity_scores(arcs, "jaccard")
