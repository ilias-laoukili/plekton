# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for the graph converters (``to_pyg``, ``to_networkx``) and NumPy generators."""

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
_SBM_TUPLE_LEN = 3
_CSBM_TUPLE_LEN = 4
_N_COMMUNITIES_MAX = 5

try:
    import plekton as gg

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
    """Build a small unlabeled SBM graph (no node features)."""
    edge_index, labels, _ = gg.generate_sbm(200, 4, 0.3, 0.02, seed=0)
    return gg.to_json(edge_index, labels)


def _csbm_json() -> str:
    """Build a small labelled cSBM graph."""
    edge_index, x, labels, _ = gg.generate_csbm(120, 2, 0.85, 8.0, 16, 1.2, seed=42)
    return gg.to_json(edge_index, labels, x)


# ── generate_sbm / generate_ba / generate_csbm (raw numpy output) ─────────────


@requires_plekton
class TestGenerateSBM:
    """The Rust zero-copy numpy bridge that ``to_pyg`` is built on."""

    def test_returns_three_arrays(self) -> None:
        """Returns three arrays."""
        result = gg.generate_sbm(n_nodes=200, n_communities=4, p_in=0.15, p_out=0.01, seed=0)
        assert len(result) == _SBM_TUPLE_LEN

    def test_edge_index_shape(self) -> None:
        """Edge index shape."""
        edge_index, _labels, _h = gg.generate_sbm(
            n_nodes=200,
            n_communities=4,
            p_in=0.15,
            p_out=0.01,
            seed=0,
        )
        assert edge_index.ndim == _EDGE_INDEX_ROWS
        assert edge_index.shape[0] == _EDGE_INDEX_ROWS
        assert edge_index.shape[1] % 2 == 0  # bidirectional

    def test_labels_shape_and_range(self) -> None:
        """Labels shape and range."""
        _edge_index, labels, _h = gg.generate_sbm(
            n_nodes=300,
            n_communities=5,
            p_in=0.2,
            p_out=0.01,
            seed=1,
        )
        assert labels.shape == (300,)
        assert int(labels.min()) >= 0
        assert int(labels.max()) < _N_COMMUNITIES_MAX

    def test_reproducibility_same_seed(self) -> None:
        """Reproducibility same seed."""
        r1 = gg.generate_sbm(n_nodes=200, n_communities=4, p_in=0.15, p_out=0.01, seed=42)
        r2 = gg.generate_sbm(n_nodes=200, n_communities=4, p_in=0.15, p_out=0.01, seed=42)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])
        assert r1[2] == r2[2]


@requires_plekton
class TestGenerateBA:
    """Zero-copy NumPy Barabási–Albert generator."""

    def test_ba_shape_and_reproducibility(self) -> None:
        """BA produces expected edge count (1 + 98*2 = 197 canonical -> 394 directed)."""
        ei1, labels1, _ = gg.generate_ba(n_nodes=100, m=2, m0=2, n_communities=4, seed=42)
        ei2, labels2, _ = gg.generate_ba(n_nodes=100, m=2, m0=2, n_communities=4, seed=42)
        assert ei1.shape == (2, 394)
        assert labels1.shape == (100,)
        np.testing.assert_array_equal(ei1, ei2)
        np.testing.assert_array_equal(labels1, labels2)


@requires_plekton
class TestGenerateCSBM:
    """The Rust cSBM bridge (edge_index, x, labels, effective_h)."""

    def test_returns_four_arrays(self) -> None:
        """Returns four arrays."""
        result = gg.generate_csbm(
            n_nodes=200,
            n_communities=3,
            homophily=0.8,
            avg_degree=10.0,
            feat_dim=16,
            mu=1.0,
            seed=0,
        )
        assert len(result) == _CSBM_TUPLE_LEN

    def test_feature_matrix_shape_and_finite(self) -> None:
        """Feature matrix shape and finite."""
        _edge_index, x, _labels, _h = gg.generate_csbm(
            n_nodes=500,
            n_communities=4,
            homophily=0.7,
            avg_degree=8.0,
            feat_dim=32,
            mu=1.5,
            seed=5,
        )
        assert x.shape == (500, 32)
        assert np.all(np.isfinite(x)), "cSBM features contain NaN or Inf"


@requires_plekton
class TestToJson:
    """Unified ``plekton.to_json`` serializer across all generators."""

    def test_sbm_round_trip_and_file_write(self, tmp_path: Path) -> None:
        """Serializes SBM arrays to JSON string and optional file path."""
        edge_index, labels, _ = gg.generate_sbm(80, 4, 0.25, 0.02, seed=7)
        out = tmp_path / "sbm.json"
        doc_str = gg.to_json(edge_index, labels, path=out)
        assert out.read_text(encoding="utf-8") == doc_str
        doc = json.loads(doc_str)
        assert doc["metadata"]["schema"] == "plekton.graph/v1"
        assert doc["metadata"]["n_nodes"] == 80
        assert doc["metadata"]["n_edges"] == edge_index.shape[1] // 2
        ei2, lbl2, x2 = gg.json_to_pyg(doc_str)
        assert ei2.shape == edge_index.shape
        np.testing.assert_array_equal(lbl2, labels)
        assert x2 is None

    def test_csbm_and_ba_serialize_uniformly(self) -> None:
        """Both cSBM (with features) and BA (without features) serialize cleanly."""
        ei_c, x_c, lbl_c, _ = gg.generate_csbm(60, 3, 0.8, 6.0, 8, 1.0, seed=3)
        ei2, lbl2, x2 = gg.json_to_pyg(gg.to_json(ei_c, lbl_c, x_c))
        assert x2 is not None
        assert x2.shape == x_c.shape
        np.testing.assert_allclose(x2, x_c, atol=1e-6)
        np.testing.assert_array_equal(lbl2, lbl_c)
        assert ei2.shape == ei_c.shape

        ei_b, lbl_b, _ = gg.generate_ba(50, m=2, m0=2, n_communities=2, seed=9)
        ba_doc = json.loads(gg.to_json(ei_b, lbl_b))
        assert ba_doc["metadata"]["n_nodes"] == 50
        assert ba_doc["metadata"]["n_edges"] == ei_b.shape[1] // 2


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


@requires_plekton
def test_py_typed_and_stubs_match_extension() -> None:
    """PEP 561 marker and ``_plekton.pyi`` are packaged and cover every PyO3 symbol."""
    import ast  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import plekton._plekton as ext  # noqa: PLC0415

    pkg_dir = Path(gg.__file__).resolve().parent
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
