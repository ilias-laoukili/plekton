# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""NetworkX converters for ``plekton``.

:func:`to_networkx` turns any canonical ``plekton.graph/v1`` JSON document
into a NetworkX graph (without requiring PyTorch), and :func:`networkx_to_json`
is the inverse.

Requires::

    pip install plekton[networkx]

Quick start
-----------
>>> from plekton import generate_sbm, to_json
>>> from plekton.nx import to_networkx, networkx_to_json
>>> edge_index, labels, _ = generate_sbm(500, 5, 0.3, 0.02, seed=0)
>>> graph = to_networkx(to_json(edge_index, labels))
>>> graph_json = networkx_to_json(graph)
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

from plekton._graph_io import GraphInput, coerce_graph_json


def to_networkx(graph: GraphInput, *, directed: bool = False) -> nx.Graph:
    """Convert a ``plekton`` JSON graph into a NetworkX ``Graph`` or ``DiGraph``."""
    doc = json.loads(coerce_graph_json(graph))

    out: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    metadata = doc.get("metadata") or {}
    out.graph.update(metadata)

    for node in doc.get("nodes", []):
        attrs: dict[str, Any] = {"community": int(node.get("community", 0))}
        features = node.get("features")
        if features is not None:
            attrs["features"] = list(features)
        out.add_node(int(node["id"]), **attrs)

    for edge in doc.get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        out.add_edge(source, target)
        if directed:
            out.add_edge(target, source)

    return out


def networkx_to_json(graph: nx.Graph) -> str:
    """Serialise a NetworkX graph into canonical ``plekton.graph/v1`` JSON."""
    node_ids = sorted(int(n) for n in graph.nodes())
    n_nodes = len(node_ids)

    nodes = []
    communities: dict[int, int] = {}
    for node_id in node_ids:
        attrs = graph.nodes[node_id]
        community = int(attrs.get("community", 0))
        communities[node_id] = community
        record: dict[str, object] = {"id": node_id, "community": community}
        features = attrs.get("features")
        if features is not None:
            record["features"] = list(features)
        nodes.append(record)

    canonical_edges = sorted(
        {(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges() if int(u) != int(v)},
    )
    n_edges = len(canonical_edges)

    any_labeled = any(c != 0 for c in communities.values())
    if n_edges == 0 or not any_labeled:
        homophily = 0.0
    else:
        n_matching = sum(
            1 for (src, dst) in canonical_edges if communities[src] == communities[dst]
        )
        homophily = n_matching / n_edges

    avg_degree = 2.0 * n_edges / n_nodes if n_nodes > 0 else 0.0

    metadata: dict[str, Any] = {
        "schema": "plekton.graph/v1",
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "homophily": homophily,
        "avg_degree_actual": avg_degree,
        "n_components": 0,
    }
    schema = graph.graph.get("feature_schema")
    if schema is not None:
        metadata["feature_schema"] = schema

    graph_dict = {
        "metadata": metadata,
        "nodes": nodes,
        "edges": [{"source": src, "target": dst} for src, dst in canonical_edges],
    }
    return json.dumps(graph_dict)
