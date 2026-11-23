# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""PyTorch Geometric converter for ``plekton``.

:func:`to_pyg` turns any canonical ``plekton.graph/v1`` JSON document into a
``torch_geometric.data.Data`` object via the zero-copy Rust bridge
(``_plekton.json_to_pyg``).

Requires::

    pip install plekton[pyg]

Quick start
-----------
>>> from plekton import generate_sbm, to_json
>>> from plekton.pyg import to_pyg
>>> edge_index, labels, _ = generate_sbm(500, 5, 0.3, 0.02, seed=0)
>>> data = to_pyg(to_json(edge_index, labels))
"""

from __future__ import annotations

import json

import torch
from torch_geometric.data import Data

import plekton as _rs
from plekton._graph_io import GraphInput, coerce_graph_json


def to_pyg(graph: GraphInput) -> Data:
    """Convert a ``plekton`` JSON graph into a PyG ``Data`` object."""
    json_str = coerce_graph_json(graph)
    edge_index_np, labels_np, x_np = _rs.json_to_pyg(json_str)
    data = Data(
        edge_index=torch.from_numpy(edge_index_np.copy()),
        y=torch.from_numpy(labels_np.copy()),
        num_nodes=int(labels_np.shape[0]),
    )
    if x_np is not None:
        data.x = torch.from_numpy(x_np.copy())
    metadata = json.loads(json_str).get("metadata") or {}
    schema = metadata.get("feature_schema")
    if schema is not None:
        data.feature_schema = schema
    return data
