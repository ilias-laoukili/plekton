# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for :class:`plekton.Graph` — construction, validation, layouts, and seeds."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

plekton = pytest.importorskip("plekton")
Graph = plekton.Graph
from plekton.hetero import to_hetero_data  # torch-free: torch is imported only on use

# A triangle plus a pendant edge, canonical (u < v).
EDGES = np.array([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=np.int64)


# ── Construction ──────────────────────────────────────────────────────────────


def test_a_graph_holds_its_edges_and_nodes() -> None:
    """Edges come back as a read-only int64 (E, 2) array; n_nodes counts isolated nodes."""
    graph = Graph(EDGES, n_nodes=6)
    np.testing.assert_array_equal(graph.edges, EDGES)
    assert graph.edges.dtype == np.int64
    assert graph.edges.flags.c_contiguous
    assert graph.n_nodes == 6
    assert graph.n_edges == len(EDGES)
    assert graph.labels is None
    assert graph.x is None
    assert not graph.directed
    assert dict(graph.metadata) == {}
    assert dict(graph.edge_attrs) == {}


def test_a_list_of_pairs_is_accepted() -> None:
    """A plain list of pairs gives the same graph as the array."""
    np.testing.assert_array_equal(Graph(EDGES.tolist(), n_nodes=4).edges, EDGES)


@pytest.mark.parametrize(
    "empty",
    [[], np.empty((0, 2), dtype=np.int64), np.zeros((0, 2)), np.empty((2, 0), dtype=np.int64)],
)
def test_an_empty_edge_list_gives_an_edgeless_graph(empty: object) -> None:
    """Any empty edge input is an edgeless graph, isolated nodes included."""
    graph = Graph(empty, n_nodes=5)
    assert graph.n_edges == 0
    assert graph.edges.shape == (0, 2)
    assert graph.edge_index.shape == (2, 0)
    assert plekton.metrics.components(graph) == 5


def test_labels_features_metadata_and_edge_attrs_are_stored() -> None:
    """Optional parts are validated against the graph and stored read-only."""
    x = np.arange(8, dtype=np.float64).reshape(4, 2)
    graph = Graph(
        EDGES,
        n_nodes=4,
        labels=[0, 0, 1, 1],
        x=x,
        metadata={"source": "test", "report": {"within": 0.5}},
        edge_attrs={"weight": [0.1, 0.2, 0.3, 0.4]},
    )
    np.testing.assert_array_equal(graph.labels, [0, 0, 1, 1])
    assert graph.labels is not None
    assert graph.labels.dtype == np.int64
    assert graph.x is not None
    assert graph.x.dtype == np.float64  # the feature dtype is kept
    assert graph.metadata["report"]["within"] == pytest.approx(0.5)
    np.testing.assert_array_equal(graph.edge_attrs["weight"], [0.1, 0.2, 0.3, 0.4])


def test_repr_summarises_without_arrays() -> None:
    """The repr names the parts, not their contents."""
    graph = Graph(EDGES, n_nodes=4, labels=[0, 0, 1, 1], edge_attrs={"w": [1, 2, 3, 4]})
    assert repr(graph) == "Graph(n_nodes=4, n_edges=4, labels, edge_attrs=['w'])"


# ── Immutability ──────────────────────────────────────────────────────────────


def test_attributes_cannot_be_set_or_deleted() -> None:
    """A Graph never changes."""
    graph = Graph(EDGES, n_nodes=4)
    with pytest.raises(AttributeError, match="immutable"):
        graph.n_nodes = 10
    with pytest.raises(AttributeError, match="immutable"):
        del graph.edges


def test_every_exposed_array_is_read_only() -> None:
    """Edges, labels, features and edge attributes cannot be written through the Graph."""
    graph = Graph(
        EDGES,
        n_nodes=4,
        labels=[0, 0, 1, 1],
        x=np.ones((4, 2)),
        edge_attrs={"w": np.ones(4)},
    )
    arrays = [graph.edges, graph.labels, graph.x, graph.edge_attrs["w"]]
    for arr in arrays:
        assert arr is not None
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0


def test_metadata_is_frozen_all_the_way_down() -> None:
    """Nested metadata mappings are read-only too."""
    graph = Graph(EDGES, n_nodes=4, metadata={"report": {"within": 0.5}, "seq": [1, 2]})
    with pytest.raises(TypeError):
        graph.metadata["report"]["within"] = 1.0
    with pytest.raises(TypeError):
        graph.edge_attrs["new"] = np.zeros(4)
    assert graph.metadata["seq"] == (1, 2)


def test_immutability_does_not_cover_the_callers_array() -> None:
    """Documented: without a copy, the caller's array still writes through."""
    caller = EDGES.copy()
    graph = Graph(caller, n_nodes=4)  # int64, C-contiguous: kept as a view, not copied
    assert np.shares_memory(graph.edges, caller)
    caller[3] = [1, 3]
    np.testing.assert_array_equal(graph.edges[3], [1, 3])


# ── Validation (undirected) ───────────────────────────────────────────────────


def test_a_two_by_e_array_is_refused_with_a_pointer() -> None:
    """A (2, E) array is an edge_index, not an edge list: refused, with the way out."""
    with pytest.raises(ValueError, match="from_edge_index"):
        Graph(EDGES.T, n_nodes=4)


def test_an_edge_in_both_directions_is_refused() -> None:
    """Passing edge_index.T would double every degree; it is refused instead."""
    with pytest.raises(ValueError, match="both directions"):
        Graph(np.vstack([EDGES, EDGES[:1, ::-1]]), n_nodes=4)


def test_a_repeated_edge_is_refused() -> None:
    """A simple graph lists each edge once."""
    with pytest.raises(ValueError, match="appears twice"):
        Graph(np.vstack([EDGES, EDGES[:1]]), n_nodes=4)


def test_a_self_loop_is_refused() -> None:
    """A simple graph has no self-loop."""
    with pytest.raises(ValueError, match="self-loop"):
        Graph([[0, 1], [2, 2]], n_nodes=3)


def test_a_reversed_edge_is_refused_not_reoriented() -> None:
    """(v, u) with v > u is refused with the fix, never silently swapped."""
    with pytest.raises(ValueError, match=r"np\.sort"):
        Graph([[0, 1], [3, 2]], n_nodes=4)


@pytest.mark.parametrize("bad", [[[0, 4]], [[-1, 2]]])
def test_an_id_out_of_range_is_refused(bad: list[list[int]]) -> None:
    """Every id must lie in [0, n_nodes)."""
    with pytest.raises(IndexError, match="out of range"):
        Graph(bad, n_nodes=4)


@pytest.mark.parametrize("bad", [np.array([[0.0, 1.0]]), np.array([[True, False]])])
def test_a_non_integer_dtype_is_refused(bad: np.ndarray) -> None:
    """Floats are refused, never truncated; booleans are not ids either."""
    with pytest.raises(TypeError, match="integer"):
        Graph(bad, n_nodes=4)


def test_a_wrong_shape_is_refused() -> None:
    """Anything that is not (E, 2) is refused."""
    with pytest.raises(ValueError, match=r"\(E, 2\)"):
        Graph([[0, 1, 2]], n_nodes=4)


@pytest.mark.parametrize("n_nodes", [-1, 2.0, True])
def test_n_nodes_must_be_a_non_negative_integer(n_nodes: object) -> None:
    """n_nodes is a count."""
    with pytest.raises((TypeError, ValueError)):
        Graph([], n_nodes=n_nodes)


def test_parts_must_match_the_graph() -> None:
    """Labels, features and edge attributes must line up with nodes and edges."""
    with pytest.raises(ValueError, match="labels"):
        Graph(EDGES, n_nodes=4, labels=[0, 1])
    with pytest.raises(ValueError, match="non-negative"):
        Graph(EDGES, n_nodes=4, labels=[0, -1, 0, 0])
    with pytest.raises(ValueError, match="x must have shape"):
        Graph(EDGES, n_nodes=4, x=np.ones((3, 2)))
    with pytest.raises(ValueError, match="one entry per edge"):
        Graph(EDGES, n_nodes=4, edge_attrs={"w": [1.0, 2.0]})


# ── Validation (directed) ─────────────────────────────────────────────────────


def test_directed_arcs_follow_their_own_rules() -> None:
    """(u, v) and (v, u) are two arcs; u < v does not apply; loops and repeats are refused."""
    arcs = Graph([[0, 1], [1, 0], [2, 1]], n_nodes=3, directed=True)
    assert arcs.n_edges == 3
    np.testing.assert_array_equal(arcs.edge_index, [[0, 1, 2], [1, 0, 1]])
    with pytest.raises(ValueError, match="appears twice"):
        Graph([[0, 1], [0, 1]], n_nodes=2, directed=True)
    with pytest.raises(ValueError, match="self-loop"):
        Graph([[1, 1]], n_nodes=2, directed=True)


# ── edge_index <-> edges ──────────────────────────────────────────────────────


def test_edge_index_lists_every_edge_then_its_reverse() -> None:
    """The PyG layout: E columns, then the same columns reversed (2E in all)."""
    graph = Graph(EDGES, n_nodes=4)
    expected = np.concatenate([EDGES.T, EDGES[:, ::-1].T], axis=1)
    np.testing.assert_array_equal(graph.edge_index, expected)
    assert graph.edge_index.flags.c_contiguous


def test_edge_index_is_rebuilt_on_every_access() -> None:
    """Not cached: each access is a fresh array the caller owns."""
    graph = Graph(EDGES, n_nodes=4)
    first = graph.edge_index
    first[0, 0] = 99
    assert graph.edge_index[0, 0] == EDGES[0, 0]


def test_from_edge_index_round_trips_exactly() -> None:
    """Graph.from_edge_index(g.edge_index) gives back g.edges, in order."""
    for graph in (
        Graph(EDGES, n_nodes=4),
        plekton.generate_sbm(300, 3, 0.1, 0.01, seed=2),
        plekton.generate_ba(100, m=3, seed=4),
    ):
        back = Graph.from_edge_index(graph.edge_index, n_nodes=graph.n_nodes)
        np.testing.assert_array_equal(back.edges, graph.edges)
    arcs = Graph([[0, 1], [1, 0], [2, 1]], n_nodes=3, directed=True)
    back = Graph.from_edge_index(arcs.edge_index, n_nodes=3, directed=True)
    np.testing.assert_array_equal(back.edges, arcs.edges)


def test_from_edge_index_accepts_any_column_order() -> None:
    """A PyG edge_index sorted by source still works; edges follow the u < v columns."""
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    graph = Graph.from_edge_index(edge_index, n_nodes=3)
    np.testing.assert_array_equal(graph.edges, [[0, 1], [1, 2]])


@pytest.mark.parametrize(
    ("edge_index", "match"),
    [
        (np.array([[0, 1], [1, 2]]), "both directions"),  # one direction only
        (np.array([[0, 0, 1], [1, 1, 0]]), "more than once"),  # repeated column
        (np.array([[0, 1, 2], [1, 0, 2]]), "self-loop"),
        (np.array([[0, 1, 2], [1, 0, 3]]), "out of range"),
    ],
)
def test_from_edge_index_needs_each_edge_once_each_way(edge_index: np.ndarray, match: str) -> None:
    """Undirected: every edge exactly once in each direction, nothing else."""
    with pytest.raises((ValueError, IndexError), match=match):
        Graph.from_edge_index(edge_index, n_nodes=3)


# ── Derived graphs, pickling, JSON ────────────────────────────────────────────


def test_edge_subgraph_keeps_nodes_and_drops_stale_metadata() -> None:
    """Every node stays; edge attributes follow the mask; measured metadata is dropped."""
    graph = Graph(
        EDGES,
        n_nodes=6,
        labels=[0, 0, 1, 1, 2, 2],
        metadata={"homophily": 0.5},
        edge_attrs={"w": [1.0, 2.0, 3.0, 4.0]},
    )
    keep = np.array([True, False, True, False])
    sub = graph.edge_subgraph(keep)
    assert sub.n_nodes == 6
    np.testing.assert_array_equal(sub.edges, EDGES[keep])
    np.testing.assert_array_equal(sub.labels, graph.labels)
    np.testing.assert_array_equal(sub.edge_attrs["w"], [1.0, 3.0])
    assert dict(sub.metadata) == {}
    with pytest.raises(ValueError, match="boolean"):
        graph.edge_subgraph(np.array([0, 2]))


def test_pickle_round_trip_keeps_everything_read_only() -> None:
    """A pickled Graph comes back equal, frozen, with its attributes."""
    graph = plekton.mixing_graph(60, mean_degree=4.0, strength={"mean": 0.5}, seed=1)
    back = pickle.loads(pickle.dumps(graph))  # noqa: S301 - our own bytes
    np.testing.assert_array_equal(back.edges, graph.edges)
    np.testing.assert_array_equal(back.edge_attrs["strength"], graph.edge_attrs["strength"])
    assert back.metadata["seed"] == graph.metadata["seed"]
    assert back.metadata["mixing"] == graph.metadata["mixing"]
    assert not back.edges.flags.writeable
    with pytest.raises(TypeError):
        back.metadata["clustering"]["target"] = 1.0


def test_from_json_round_trips_and_refuses_misnumbered_nodes() -> None:
    """to_json then from_json gives the same edges; node ids must be 0..n in order."""
    graph = plekton.generate_csbm(50, 2, 0.8, 4.0, 3, 1.0, seed=6)
    back = Graph.from_json(plekton.to_json(graph))
    np.testing.assert_array_equal(np.sort(back.edges, axis=0), np.sort(graph.edges, axis=0))
    assert back.n_nodes == graph.n_nodes
    document = {
        "metadata": {"schema": "plekton.graph/v1", "n_nodes": 2, "n_edges": 1, "homophily": None},
        "nodes": [{"id": 1, "community": 0}, {"id": 0, "community": 0}],
        "edges": [{"source": 0, "target": 1}],
    }
    with pytest.raises(ValueError, match="position 0 has id 1"):
        Graph.from_json(document)


def test_to_hetero_data_refuses_a_graph() -> None:
    """A homogeneous Graph has no types to map; the error names PyG's own conversion."""
    with pytest.raises(TypeError, match="to_heterogeneous"):
        to_hetero_data(Graph(EDGES, n_nodes=4))


# ── Seeds ─────────────────────────────────────────────────────────────────────


def test_the_recorded_seed_is_the_seed_that_ran() -> None:
    """An explicit seed is recorded as given; a drawn one regenerates the same graph."""
    assert plekton.generate_ba(50, seed=123).metadata["seed"] == 123
    drawn = plekton.generate_csbm(80, 2, 0.7, 4.0, 3, 1.0)
    replay = plekton.generate_csbm(80, 2, 0.7, 4.0, 3, 1.0, seed=drawn.metadata["seed"])
    np.testing.assert_array_equal(drawn.edges, replay.edges)
    np.testing.assert_array_equal(drawn.x, replay.x)


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5])
def test_an_invalid_seed_is_refused(seed: object) -> None:
    """A seed is an integer in [0, 2**64); bools and floats are not coerced."""
    with pytest.raises((TypeError, ValueError)):
        plekton.generate_sbm(20, 2, 0.5, 0.1, seed=seed)


# ── Every graph-consuming function refuses the step-1c format errors ──────────

_SCORES3 = np.ones(3)
_CALLERS: dict[str, Callable[[Any], object]] = {
    "metrics.homophily": lambda e: plekton.metrics.homophily(e, n_nodes=4, labels=[0, 0, 1, 1]),
    "metrics.attribute_mixing": lambda e: plekton.metrics.attribute_mixing(
        e,
        n_nodes=4,
        labels=[0, 0, 1, 1],
    ),
    "metrics.clustering": lambda e: plekton.metrics.clustering(e, n_nodes=4),
    "metrics.transitivity": lambda e: plekton.metrics.transitivity(e, n_nodes=4),
    "metrics.assortativity": lambda e: plekton.metrics.assortativity(e, n_nodes=4),
    "metrics.components": lambda e: plekton.metrics.components(e, n_nodes=4),
    "metrics.density": lambda e: plekton.metrics.density(e, n_nodes=4),
    "metrics.degree_summary": lambda e: plekton.metrics.degree_summary(e, n_nodes=4),
    "metrics.degree_ks": lambda e: plekton.metrics.degree_ks(e, EDGES, n_nodes_a=4, n_nodes_b=4),
    "fit.fit_blockmodel": lambda e: plekton.fit.fit_blockmodel(e, n_nodes=4, labels="detect"),
    "fit.label_propagation": lambda e: plekton.fit.label_propagation(e, n_nodes=4),
    "fit.overlapping_communities": lambda e: plekton.fit.overlapping_communities(e, n_nodes=4),
    "sparsify.similarity_scores": lambda e: plekton.sparsify.similarity_scores(
        e,
        "jaccard",
        n_nodes=4,
    ),
    "sparsify.approx_effective_resistance": lambda e: plekton.sparsify.approx_effective_resistance(
        e,
        n_nodes=4,
    ),
    "sparsify.select_mask": lambda e: plekton.sparsify.select_mask(
        e,
        _SCORES3,
        "threshold",
        0.5,
        n_nodes=4,
    ),
    "sparsify.metric_backbone_mask": lambda e: plekton.sparsify.metric_backbone_mask(
        e,
        _SCORES3,
        n_nodes=4,
    ),
}
_FORMAT_ERRORS: dict[str, tuple[np.ndarray, type[Exception], str]] = {
    "a (2, E) array": (np.array([[0, 1, 2], [1, 2, 3]]), ValueError, "from_edge_index"),
    "an edge in both directions": (
        np.array([[0, 1], [1, 2], [1, 0]]),
        ValueError,
        "both directions",
    ),
    "an id outside [0, n_nodes)": (np.array([[0, 1], [2, 7], [1, 2]]), IndexError, "out of range"),
    "a non-integer dtype": (np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 2.0]]), TypeError, "integer"),
}


@pytest.mark.parametrize("error", list(_FORMAT_ERRORS))
@pytest.mark.parametrize("caller", list(_CALLERS))
def test_every_function_refuses_a_malformed_edge_array(caller: str, error: str) -> None:
    """Each format error is refused with a clear message by every function, never absorbed."""
    edges, exc, match = _FORMAT_ERRORS[error]
    with pytest.raises(exc, match=match):
        _CALLERS[caller](edges)
