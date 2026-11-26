# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for the generators (each returns a ``plekton.Graph``) and the graph converters."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_N_NODES_CSBM = 120
_FEAT_DIM_CSBM = 16
_N_NODES_SBM = 200
_EDGE_INDEX_ROWS = 2
_N_COMMUNITIES_MAX = 5

try:
    import plekton

    _HAS_PLEKTON = True
except ImportError:
    _HAS_PLEKTON = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import torch
    from plekton.pyg import to_pyg
    from torch_geometric.data import Data  # noqa: F401 - availability probe

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

try:
    import networkx as nx
    from plekton.nx import networkx_to_json, to_networkx

    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

requires_plekton = pytest.mark.skipif(
    not _HAS_PLEKTON or not _HAS_NUMPY,
    reason="plekton extension or numpy not installed (run `maturin develop`)",
)
requires_pyg = pytest.mark.skipif(
    not (_HAS_PLEKTON and _HAS_PYG),
    reason="torch / torch_geometric (or the compiled extension) not installed",
)
requires_networkx = pytest.mark.skipif(
    not (_HAS_PLEKTON and _HAS_NETWORKX),
    reason="networkx (or the compiled extension) not installed",
)


def _sbm_json() -> str:
    """Build a small SBM graph (no node features) as JSON."""
    return plekton.to_json(plekton.generate_sbm(200, 4, 0.3, 0.02, seed=0))


def _csbm_json() -> str:
    """Build a small labelled cSBM graph as JSON."""
    return plekton.to_json(plekton.generate_csbm(120, 2, 0.85, 8.0, 16, 1.2, seed=42))


# ── generate_sbm / generate_ba / generate_csbm (plekton.Graph output) ─────────


@requires_plekton
class TestGenerateSBM:
    """The Rust generators hand their canonical edges over as a ``plekton.Graph``."""

    def test_returns_a_graph(self) -> None:
        """Returns a Graph with labels and the measured homophily."""
        graph = plekton.generate_sbm(200, 4, p_in=0.15, p_out=0.01, seed=0)
        assert isinstance(graph, plekton.Graph)
        assert graph.n_nodes == _N_NODES_SBM
        assert 0.0 <= graph.metadata["homophily"] <= 1.0
        assert graph.metadata["seed"] == 0

    def test_edges_are_canonical_and_edge_index_bidirectional(self) -> None:
        """Edges are (E, 2) with u < v; edge_index lists them, then their reverses."""
        graph = plekton.generate_sbm(200, 4, p_in=0.15, p_out=0.01, seed=0)
        edges = graph.edges
        assert edges.shape == (graph.n_edges, 2)
        assert edges.dtype == np.int64
        assert edges.flags.c_contiguous
        assert np.all(edges[:, 0] < edges[:, 1])
        edge_index = graph.edge_index
        assert edge_index.shape == (_EDGE_INDEX_ROWS, 2 * graph.n_edges)
        np.testing.assert_array_equal(edge_index[:, : graph.n_edges], edges.T)
        np.testing.assert_array_equal(edge_index[:, graph.n_edges :], edges[:, ::-1].T)

    def test_labels_shape_and_range(self) -> None:
        """Labels shape and range."""
        graph = plekton.generate_sbm(300, 5, p_in=0.2, p_out=0.01, seed=1)
        labels = graph.labels
        assert labels is not None
        assert labels.shape == (300,)
        assert int(labels.min()) >= 0
        assert int(labels.max()) < _N_COMMUNITIES_MAX

    def test_reproducibility_same_seed(self) -> None:
        """Reproducibility same seed."""
        g1 = plekton.generate_sbm(200, 4, p_in=0.15, p_out=0.01, seed=42)
        g2 = plekton.generate_sbm(200, 4, p_in=0.15, p_out=0.01, seed=42)
        np.testing.assert_array_equal(g1.edges, g2.edges)
        np.testing.assert_array_equal(g1.labels, g2.labels)
        assert g1.metadata == g2.metadata

    def test_a_drawn_seed_is_recorded_and_reproduces_the_graph(self) -> None:
        """seed=None draws a seed; the recorded one regenerates the same graph."""
        first = plekton.generate_sbm(200, 4, p_in=0.15, p_out=0.01)
        again = plekton.generate_sbm(200, 4, p_in=0.15, p_out=0.01, seed=first.metadata["seed"])
        np.testing.assert_array_equal(first.edges, again.edges)


@requires_plekton
class TestGenerateBA:
    """Barabási–Albert generator."""

    def test_ba_shape_and_reproducibility(self) -> None:
        """BA produces the expected edge count (1 + 98 * 2 = 197)."""
        g1 = plekton.generate_ba(100, m=2, m0=2, n_communities=4, seed=42)
        g2 = plekton.generate_ba(100, m=2, m0=2, n_communities=4, seed=42)
        assert g1.n_edges == 197
        assert g1.edge_index.shape == (2, 394)
        assert g1.labels is not None
        assert g1.labels.shape == (100,)
        np.testing.assert_array_equal(g1.edges, g2.edges)
        np.testing.assert_array_equal(g1.labels, g2.labels)

    def test_m1_leaves_no_node_isolated(self) -> None:
        """With m = 1 every node, the seed node included, gets an edge: a tree on n nodes."""
        graph = plekton.generate_ba(30, m=1, seed=0)
        assert graph.n_edges == 29
        degrees = np.bincount(graph.edges.ravel(), minlength=30)
        assert degrees.min() >= 1, f"isolated node in {degrees.tolist()}"


@requires_plekton
class TestGenerateCSBM:
    """The Rust cSBM bridge: edges, features and labels in one Graph."""

    def test_returns_a_graph_with_features(self) -> None:
        """Returns a Graph whose x has one row per node."""
        graph = plekton.generate_csbm(
            200,
            3,
            homophily=0.8,
            avg_degree=10.0,
            feat_dim=16,
            mu=1.0,
            seed=0,
        )
        assert isinstance(graph, plekton.Graph)
        assert graph.x is not None
        assert graph.x.shape == (200, 16)

    def test_feature_matrix_shape_and_finite(self) -> None:
        """Feature matrix shape and finite."""
        graph = plekton.generate_csbm(
            500,
            4,
            homophily=0.7,
            avg_degree=8.0,
            feat_dim=32,
            mu=1.5,
            seed=5,
        )
        x = graph.x
        assert x is not None
        assert x.shape == (500, 32)
        assert x.dtype == np.float32
        assert np.all(np.isfinite(x)), "cSBM features contain NaN or Inf"


@requires_plekton
class TestToJson:
    """``plekton.to_json`` serialises any undirected Graph."""

    def test_sbm_round_trip_and_file_write(self, tmp_path: Path) -> None:
        """Serializes a Graph to a JSON string and an optional file; from_json reads it back."""
        graph = plekton.generate_sbm(80, 4, 0.25, 0.02, seed=7)
        out = tmp_path / "sbm.json"
        doc_str = plekton.to_json(graph, path=out)
        assert out.read_text(encoding="utf-8") == doc_str
        doc = json.loads(doc_str)
        assert doc["metadata"]["schema"] == "plekton.graph/v1"
        assert doc["metadata"]["n_nodes"] == 80
        assert doc["metadata"]["n_edges"] == graph.n_edges
        back = plekton.Graph.from_json(doc_str)
        assert back.edge_index.shape == graph.edge_index.shape
        np.testing.assert_array_equal(back.labels, graph.labels)
        assert back.x is None
        assert back.metadata["schema"] == "plekton.graph/v1"

    def test_csbm_and_ba_serialize_uniformly(self) -> None:
        """Both cSBM (with features) and BA (without features) serialize cleanly."""
        csbm = plekton.generate_csbm(60, 3, 0.8, 6.0, 8, 1.0, seed=3)
        back = plekton.Graph.from_json(plekton.to_json(csbm))
        assert back.x is not None
        assert csbm.x is not None
        assert back.x.shape == csbm.x.shape
        np.testing.assert_allclose(back.x, csbm.x, atol=1e-6)
        np.testing.assert_array_equal(back.labels, csbm.labels)
        assert back.n_edges == csbm.n_edges

        ba = plekton.generate_ba(50, m=2, m0=2, n_communities=2, seed=9)
        ba_doc = json.loads(plekton.to_json(ba))
        assert ba_doc["metadata"]["n_nodes"] == 50
        assert ba_doc["metadata"]["n_edges"] == ba.n_edges

    def test_nothing_is_dropped_silently(self) -> None:
        """v1 is undirected and has no edge attributes: both refuse unless asked."""
        arcs = plekton.Graph([[0, 1], [1, 0]], n_nodes=2, directed=True)
        with pytest.raises(ValueError, match="undirected"):
            plekton.to_json(arcs)
        weighted = plekton.Graph([[0, 1]], n_nodes=2, edge_attrs={"strength": [0.5]})
        with pytest.raises(ValueError, match="drop_edge_attrs=True"):
            plekton.to_json(weighted)
        assert json.loads(plekton.to_json(weighted, drop_edge_attrs=True))["edges"] == [
            {"source": 0, "target": 1},
        ]

    def test_only_a_graph_is_serialised(self) -> None:
        """A raw array is refused with a pointer to plekton.Graph."""
        with pytest.raises(TypeError, match=r"plekton\.Graph"):
            plekton.to_json(np.array([[0, 1]]))


# ── to_pyg: the generic PyG dispatcher ─────────────────────────────────────────


@requires_pyg
class TestToPyg:
    """``to_pyg`` accepts a JSON string, a dict, or a ``.json`` path."""

    def test_from_json_string_sbm(self) -> None:
        """From JSON string sbm."""
        data = to_pyg(_sbm_json())
        assert data.edge_index.shape[0] == _EDGE_INDEX_ROWS
        assert data.edge_index.dtype == torch.long
        assert data.y.shape == (200,)
        assert data.num_nodes == _N_NODES_SBM
        assert data.x is None  # SBM has no node features

    def test_from_dict(self) -> None:
        """From dict."""
        data = to_pyg(json.loads(_sbm_json()))
        assert data.num_nodes == _N_NODES_SBM

    def test_from_json_path(self, tmp_path: Path) -> None:
        """From JSON path."""
        path = tmp_path / "graph.json"
        path.write_text(_sbm_json(), encoding="utf-8")
        data = to_pyg(str(path))
        assert data.num_nodes == _N_NODES_SBM

    def test_csbm_has_features(self) -> None:
        """Csbm has features."""
        data = to_pyg(_csbm_json())
        assert data.x is not None
        assert data.x.shape == (120, 16)
        assert data.x.dtype == torch.float32

    def test_reproducible(self) -> None:
        """Reproducible."""
        j = _sbm_json()
        d1, d2 = to_pyg(j), to_pyg(j)
        assert torch.equal(d1.edge_index, d2.edge_index)
        assert torch.equal(d1.y, d2.y)

    def test_from_a_graph_matches_its_json(self) -> None:
        """A Graph converts exactly like the JSON document it serialises to."""
        graph = plekton.generate_csbm(120, 2, 0.85, 8.0, 16, 1.2, seed=42)
        from_graph, from_json = to_pyg(graph), to_pyg(plekton.to_json(graph))
        assert torch.equal(from_graph.edge_index, from_json.edge_index)
        assert torch.equal(from_graph.y, from_json.y)
        assert torch.equal(from_graph.x, from_json.x)


# ── to_networkx: the generic NetworkX dispatcher ───────────────────────────────


@requires_networkx
class TestToNetworkx:
    """To networkx."""

    def test_node_count_and_attrs(self) -> None:
        """Node count and attrs."""
        graph = to_networkx(_csbm_json())
        assert isinstance(graph, nx.Graph)
        assert not graph.is_directed()
        assert graph.number_of_nodes() == _N_NODES_CSBM
        # Every node carries a community; cSBM nodes also carry features.
        assert all("community" in graph.nodes[n] for n in graph.nodes())
        assert len(graph.nodes[0]["features"]) == _FEAT_DIM_CSBM

    def test_metadata_on_graph(self) -> None:
        """Metadata on graph."""
        graph = to_networkx(_sbm_json())
        assert graph.graph["n_nodes"] == _N_NODES_SBM
        assert "homophily" in graph.graph

    def test_accepts_dict_and_path(self, tmp_path: Path) -> None:
        """Accepts dict and path."""
        assert to_networkx(json.loads(_sbm_json())).number_of_nodes() == _N_NODES_SBM
        path = tmp_path / "g.json"
        path.write_text(_sbm_json(), encoding="utf-8")
        assert to_networkx(str(path)).number_of_nodes() == _N_NODES_SBM

    def test_directed_doubles_edges(self) -> None:
        """Directed doubles edges."""
        j = _sbm_json()
        undirected = to_networkx(j)
        directed = to_networkx(j, directed=True)
        assert directed.is_directed()
        assert directed.number_of_edges() == 2 * undirected.number_of_edges()

    def test_from_a_graph_matches_its_json(self) -> None:
        """A Graph converts exactly like the JSON document it serialises to."""
        graph = plekton.generate_sbm(200, 4, 0.3, 0.02, seed=0)
        from_graph = to_networkx(graph)
        from_json = to_networkx(plekton.to_json(graph))
        assert sorted(from_graph.edges()) == sorted(from_json.edges())
        assert dict(from_graph.nodes(data=True)) == dict(from_json.nodes(data=True))
        assert from_graph.graph == from_json.graph


# ── Cross-converter consistency ────────────────────────────────────────────────


@requires_pyg
@requires_networkx
def test_pyg_and_networkx_edge_counts_agree() -> None:
    """PyG's bidirectional ``edge_index`` has exactly twice the undirected nx edges."""
    j = _sbm_json()
    data = to_pyg(j)
    graph = to_networkx(j)
    assert data.edge_index.shape[1] == 2 * graph.number_of_edges()


# ── networkx_to_json: the inverse ─────────────────────────────────────────────


@requires_networkx
class TestNetworkxRoundTrip:
    """Networkx round trip."""

    def test_round_trip_preserves_counts(self) -> None:
        """Round trip preserves counts."""
        graph = to_networkx(_csbm_json())
        graph2 = to_networkx(networkx_to_json(graph))
        assert graph2.number_of_nodes() == graph.number_of_nodes()
        assert graph2.number_of_edges() == graph.number_of_edges()

    def test_directed_round_trips_to_undirected_edge_set(self) -> None:
        """Directed round trips to undirected edge set."""
        j = _sbm_json()
        directed = to_networkx(j, directed=True)
        # The inverse canonicalises, collapsing (t,s) duplicates back to one edge.
        recovered = to_networkx(networkx_to_json(directed))
        assert recovered.number_of_edges() == to_networkx(j).number_of_edges()


def _metadata_of(graph: nx.Graph) -> dict[str, object]:
    """Return the ``metadata`` block ``networkx_to_json`` writes for ``graph``."""
    metadata: dict[str, object] = json.loads(networkx_to_json(graph))["metadata"]
    return metadata


@requires_networkx
class TestNetworkxToJsonMetadata:
    """``networkx_to_json`` writes measured metadata, or ``None`` with a reason — never a filler."""

    def test_components_are_counted(self) -> None:
        """Components are counted, an isolated node included."""
        graph = nx.Graph([(0, 1), (2, 3)])
        graph.add_node(4)
        assert _metadata_of(graph)["n_components"] == 3

    def test_a_connected_graph_has_one_component(self) -> None:
        """A connected graph has one component."""
        assert _metadata_of(nx.path_graph(3))["n_components"] == 1

    def test_directed_graphs_count_weak_components(self) -> None:
        """Directed graphs count weakly connected components."""
        # 0 -> 1 <- 2 is one weak component (but three strong ones).
        assert _metadata_of(nx.DiGraph([(0, 1), (2, 1)]))["n_components"] == 1

    def test_labelled_homophily_is_measured(self) -> None:
        """Labelled homophily is measured."""
        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, {0: 0, 1: 0, 2: 1}, "community")
        metadata = _metadata_of(graph)
        assert metadata["homophily"] == pytest.approx(0.5)
        assert "metrics_skipped" not in metadata

    def test_unlabelled_homophily_is_null_with_a_reason(self) -> None:
        """Unlabelled homophily is null, with a reason."""
        # Every node defaults to community 0: there is no label for an edge to share.
        metadata = _metadata_of(nx.path_graph(3))
        assert metadata["homophily"] is None
        assert metadata["metrics_skipped"] == [
            {
                "metric": "homophily",
                "reason": "undefined here: no node is labelled (every community is 0)",
            },
        ]

    def test_edgeless_homophily_is_null_with_a_reason(self) -> None:
        """Edgeless homophily is null, with a reason."""
        graph = nx.empty_graph(3)
        nx.set_node_attributes(graph, {0: 0, 1: 1, 2: 1}, "community")
        metadata = _metadata_of(graph)
        assert metadata["homophily"] is None
        assert metadata["metrics_skipped"] == [
            {"metric": "homophily", "reason": "undefined here: the graph has no edges"},
        ]

    def test_a_null_homophily_is_readable_by_the_rust_reader(self) -> None:
        """A null homophily is readable by the Rust reader."""
        # Graph.from_json (and so to_pyg) parses the metadata; a null must not break it.
        graph = plekton.Graph.from_json(networkx_to_json(nx.path_graph(3)))
        assert graph.edge_index.shape == (_EDGE_INDEX_ROWS, 4)
        assert graph.labels is not None
        assert graph.labels.shape == (3,)
        assert graph.x is None
        assert graph.metadata["homophily"] is None


@requires_plekton
def test_py_typed_and_stubs_match_extension() -> None:
    """PEP 561 marker and ``_plekton.pyi`` are packaged and cover every PyO3 symbol."""
    import ast  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import plekton._plekton as ext  # noqa: PLC0415

    pkg_dir = Path(plekton.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file(), "py.typed missing from installed wheel"
    stub_path = pkg_dir / "_plekton.pyi"
    assert stub_path.is_file(), "_plekton.pyi missing from installed wheel"

    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    top_defs: set[str] = set()
    sub_defs: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            top_defs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            top_defs.add(node.name)
            sub_defs[node.name] = {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            }

    ext_public = {name for name in dir(ext) if not name.startswith("_")}
    assert ext_public == top_defs, f"stub mismatch: ext={ext_public} vs pyi={top_defs}"
    for sub in ("fit", "metrics", "sparsify"):
        sub_mod = getattr(ext, sub)
        sub_public = {name for name in dir(sub_mod) if not name.startswith("_")}
        assert sub_public == sub_defs.get(sub, set()), (
            f"submodule '{sub}' stub mismatch: ext={sub_public} vs pyi={sub_defs.get(sub)}"
        )
