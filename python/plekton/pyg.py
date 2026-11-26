# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""PyTorch Geometric converter for ``plekton``.

:func:`to_pyg` turns a :class:`plekton.Graph`, or any canonical
``plekton.graph/v1`` JSON document, into a ``torch_geometric.data.Data``.

Requires::

    pip install plekton[pyg]

Quick start
-----------
>>> import plekton
>>> from plekton.pyg import to_pyg
>>> graph = plekton.generate_sbm(500, 5, 0.3, 0.02, seed=0)
>>> data = to_pyg(graph)
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

import plekton as _rs
from plekton._graph import Graph
from plekton._graph_io import GraphInput, coerce_graph_json


def to_pyg(graph: Graph | GraphInput, *, drop_edge_attrs: bool = False) -> Data:
    """Convert a ``plekton.Graph`` or a ``plekton`` JSON graph into a PyG ``Data``.

    A ``Graph`` goes through :func:`plekton.to_json` (so it must be undirected,
    and ``drop_edge_attrs=True`` is needed to leave its edge attributes behind);
    a JSON string, parsed dict or ``.json`` path is read with
    :meth:`plekton.Graph.from_json`. ``edge_index`` lists every edge in both
    directions, ``y`` holds the communities, ``x`` the features if any.
    """
    doc = (
        _rs.to_json(graph, drop_edge_attrs=drop_edge_attrs)
        if isinstance(graph, Graph)
        else coerce_graph_json(graph)
    )
    parsed = Graph.from_json(doc)
    labels = parsed.labels
    if labels is None:  # pragma: no cover - a JSON document always carries communities
        msg = "internal: a plekton.graph/v1 document without communities"
        raise RuntimeError(msg)
    data = Data(
        edge_index=torch.from_numpy(parsed.edge_index),
        y=torch.from_numpy(labels.copy()),
        num_nodes=parsed.n_nodes,
    )
    if parsed.x is not None:
        data.x = torch.from_numpy(parsed.x.copy())
    schema = parsed.metadata.get("feature_schema")
    if schema is not None:
        data.feature_schema = schema
    return data
