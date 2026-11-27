# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Python face of ``plekton`` — compiled Rust engine with zero-copy NumPy arrays.

The base package is deliberately lean: ``import plekton`` pulls only ``numpy``
(never ``torch`` or ``torch_geometric``). Everything revolves around one type,
:class:`plekton.Graph`: ``n_nodes`` nodes and an ``(E, 2)`` int64 edge array,
plus optional labels, features, per-edge attributes and metadata. It exposes:

- ``Graph``, with ``Graph.from_edge_index`` (PyG) and ``Graph.from_json``.
- Synthetic graph generators, each returning a ``Graph``: ``generate_sbm``,
  ``generate_sbm_from_fit``, ``generate_csbm``, ``generate_ba``, ``mixing_graph``.
- Canonical JSON: ``to_json(graph)``, and the JSON-to-JSON contract entry points
  ``generate_from_config`` and ``generate_mixing_graph_json``.
- Blockmodel estimation & community detection: ``plekton.fit``.
- Structural graph metrics: ``plekton.metrics``.
- Edge scoring (similarities, approximate effective resistance) and score-based or
  metric-backbone sparsification: ``plekton.sparsify``.
- Leakage-free group & temporal ML splitters: ``plekton.splits``.

Every randomised function takes ``seed=None`` by default, which draws a seed; the
generators record the seed that ran in ``graph.metadata["seed"]``.

Optional extras::

    from plekton.pyg import to_pyg              # pip install plekton[pyg]
    from plekton.hetero import to_hetero_data   # pip install plekton[pyg]
    from plekton.nx import to_networkx          # pip install plekton[networkx]
    from plekton.presets import get_preset, list_presets
    from plekton.sweep import sweep
    from plekton.distributions import fit_from_dataset, generate_realistic

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plekton import fit, metrics, sparsify, splits
from plekton._generators import generate_ba, generate_csbm, generate_sbm, generate_sbm_from_fit
from plekton._graph import Graph
from plekton._plekton import generate_from_config
from plekton._plekton import (
    graph_to_json as _graph_to_json,
)
from plekton.mixing import generate_mixing_graph_json, mixing_graph

if TYPE_CHECKING:
    from pathlib import Path


def to_json(
    graph: Graph,
    *,
    path: str | Path | None = None,
    drop_edge_attrs: bool = False,
) -> str:
    """Serialise a :class:`Graph` into canonical ``plekton.graph/v1`` JSON.

    When *path* is given, the document is also written to that file. A graph
    without ``labels`` has every node written in community ``0``, since the field
    is required, and its ``homophily`` is ``null``, with the reason in
    ``metrics_skipped``: there is no labelling to measure.

    ``plekton.graph/v1`` is undirected and has no field for edge attributes, so
    nothing is dropped silently: a directed graph raises, and so does a graph
    with ``edge_attrs`` unless ``drop_edge_attrs=True`` says to write it without
    them.
    """
    if not isinstance(graph, Graph):
        msg = (
            f"to_json takes a plekton.Graph, got {type(graph).__name__}; build one with "
            "plekton.Graph(edges, n_nodes=...) or plekton.Graph.from_edge_index(...)"
        )
        raise TypeError(msg)
    if graph.directed:
        msg = (
            "plekton.graph/v1 is undirected: a directed plekton.Graph cannot be written "
            "without losing the direction of its arcs"
        )
        raise ValueError(msg)
    if graph.edge_attrs and not drop_edge_attrs:
        names = ", ".join(sorted(graph.edge_attrs))
        msg = (
            f"plekton.graph/v1 has no field for edge attributes ({names}); pass "
            "drop_edge_attrs=True to write the graph without them"
        )
        raise ValueError(msg)
    x = None if graph.x is None else np.ascontiguousarray(graph.x, dtype=np.float32)
    out_path = None if path is None else str(path)
    # `edges.T` is a (2, E) view: the extension reads it without a copy.
    return str(_graph_to_json(graph.edges.T, graph.n_nodes, graph.labels, x, out_path))


__all__ = [
    "Graph",
    "fit",
    "generate_ba",
    "generate_csbm",
    "generate_from_config",
    "generate_mixing_graph_json",
    "generate_sbm",
    "generate_sbm_from_fit",
    "metrics",
    "mixing_graph",
    "sparsify",
    "splits",
    "to_json",
]
