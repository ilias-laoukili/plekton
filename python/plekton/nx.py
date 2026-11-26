# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""NetworkX converters for ``plekton``.

:func:`to_networkx` turns a :class:`plekton.Graph`, or any canonical
``plekton.graph/v1`` JSON document, into a NetworkX graph (without requiring
PyTorch), and :func:`networkx_to_json` is the inverse: nodes whose ids are not
already ``0 .. n-1`` are numbered so, each keeping its NetworkX id in
``original_id``.

Requires::

    pip install plekton[networkx]

Quick start
-----------
>>> import plekton
>>> from plekton.nx import to_networkx, networkx_to_json
>>> graph = to_networkx(plekton.generate_sbm(500, 5, 0.3, 0.02, seed=0))
>>> graph_json = networkx_to_json(graph)
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np

import plekton as _rs
from plekton._graph import Graph
from plekton._graph_io import GraphInput, coerce_graph_json


def to_networkx(
    graph: Graph | GraphInput,
    *,
    directed: bool = False,
    drop_edge_attrs: bool = False,
) -> nx.Graph:
    """Convert a ``plekton.Graph`` or a ``plekton`` JSON graph into a NetworkX graph.

    A ``Graph`` goes through :func:`plekton.to_json` (so it must be undirected,
    and ``drop_edge_attrs=True`` is needed to leave its edge attributes behind).
    ``directed=True`` returns a ``DiGraph`` with every edge in both directions.
    A node's ``original_id``, written by :func:`networkx_to_json`, becomes a node
    attribute of the same name.
    """
    text = (
        _rs.to_json(graph, drop_edge_attrs=drop_edge_attrs)
        if isinstance(graph, Graph)
        else coerce_graph_json(graph)
    )
    doc = json.loads(text)

    out: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    metadata = doc.get("metadata") or {}
    out.graph.update(metadata)

    for node in doc.get("nodes", []):
        attrs: dict[str, Any] = {"community": int(node.get("community", 0))}
        features = node.get("features")
        if features is not None:
            attrs["features"] = list(features)
        original_id = node.get("original_id")
        if original_id is not None:
            attrs["original_id"] = original_id
        out.add_node(int(node["id"]), **attrs)

    for edge in doc.get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        out.add_edge(source, target)
        if directed:
            out.add_edge(target, source)

    return out


def _edge_homophily(
    canonical_edges: list[tuple[int, int]],
    communities: dict[int, int],
) -> tuple[float | None, str | None]:
    """Edge homophily, or ``None`` and the reason it is undefined.

    Same rule as :func:`plekton.metrics.homophily`: undefined without edges, and
    undefined when every node is in community ``0`` — the converter's default for a
    node without a ``community`` attribute, so nothing was labelled.
    """
    if not canonical_edges:
        return None, "undefined here: the graph has no edges"
    if not any(community != 0 for community in communities.values()):
        return None, "undefined here: no node is labelled (every community is 0)"
    n_matching = sum(1 for src, dst in canonical_edges if communities[src] == communities[dst])
    return n_matching / len(canonical_edges), None


def _original_id(node: object) -> int | str:
    """Return the JSON value that keeps ``node``'s NetworkX id: an ``int`` or a ``str``."""
    if isinstance(node, str):
        return str(node)
    if isinstance(node, (int, np.integer)):
        return int(node)
    msg = (
        f"networkx_to_json: node {node!r} is neither an integer nor a string, so the "
        "JSON cannot keep its id as original_id"
    )
    raise TypeError(msg)


def _id_order(original_id: int | str) -> tuple[bool, int | str]:
    """Sort key: integer ids in ascending order, then string ids."""
    return isinstance(original_id, str), original_id


def networkx_to_json(graph: nx.Graph) -> str:
    """Serialise a NetworkX graph into canonical ``plekton.graph/v1`` JSON.

    A graph whose ids are already exactly ``0 .. n-1`` keeps them, and no node
    carries ``original_id``. Otherwise every node is renumbered, integer ids in
    ascending order and then string ids (so the numbering does not depend on the
    order nodes were added in), and every node keeps its NetworkX id in
    ``original_id``: the field is on all nodes or on none, and without it ``id`` is
    the NetworkX id. :func:`to_networkx` brings ``original_id`` back as a node
    attribute, and ``nx.relabel_nodes`` with that attribute restores the ids. A
    node id that is neither an integer nor a string raises :class:`TypeError`.

    ``n_components`` counts connected components (weakly connected ones for a
    directed graph). ``homophily`` is the fraction of edges whose endpoints share a
    ``community``; it is ``None`` (JSON ``null``), with a ``metrics_skipped`` entry
    giving the reason, when the graph has no edges or no node is labelled.
    """
    ordered = sorted(
        ((_original_id(node), node) for node in graph.nodes()),
        key=lambda pair: _id_order(pair[0]),
    )
    # Ids that are exactly 0..n-1 sort into their own positions: nothing to renumber.
    renumbered = any(original_id != index for index, (original_id, _) in enumerate(ordered))
    position = {node: index for index, (_, node) in enumerate(ordered)}
    n_nodes = len(ordered)

    nodes = []
    communities: dict[int, int] = {}
    for index, (original_id, node) in enumerate(ordered):
        attrs = graph.nodes[node]
        community = int(attrs.get("community", 0))
        communities[index] = community
        record: dict[str, object] = {"id": index}
        if renumbered:
            record["original_id"] = original_id
        record["community"] = community
        features = attrs.get("features")
        if features is not None:
            record["features"] = list(features)
        nodes.append(record)

    canonical_edges = sorted(
        {
            (min(position[u], position[v]), max(position[u], position[v]))
            for u, v in graph.edges()
            if u != v
        },
    )
    n_edges = len(canonical_edges)

    homophily, homophily_skipped = _edge_homophily(canonical_edges, communities)
    n_components = (
        nx.number_weakly_connected_components(graph)
        if graph.is_directed()
        else nx.number_connected_components(graph)
    )

    avg_degree = 2.0 * n_edges / n_nodes if n_nodes > 0 else 0.0

    metadata: dict[str, Any] = {
        "schema": "plekton.graph/v1",
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "homophily": homophily,
        "avg_degree_actual": avg_degree,
        "n_components": int(n_components),
    }
    if homophily_skipped is not None:
        metadata["metrics_skipped"] = [{"metric": "homophily", "reason": homophily_skipped}]
    schema = graph.graph.get("feature_schema")
    if schema is not None:
        metadata["feature_schema"] = schema

    graph_dict = {
        "metadata": metadata,
        "nodes": nodes,
        "edges": [{"source": src, "target": dst} for src, dst in canonical_edges],
    }
    return json.dumps(graph_dict)
