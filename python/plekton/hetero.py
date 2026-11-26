# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""``plekton.hetero-graph/v1`` → a PyTorch Geometric :class:`HeteroData`.

The Python half of the bridge ``plekton_entity::hetero`` writes. Node types are
entity kinds, edge types are PyG's canonical ``(source type, relation,
destination type)`` triples, and the indices are already local to their own node
type — so this is a transcription, not a conversion.

**No node features are invented, and that is the whole reason this module is
small.** The artifact states its feature stance explicitly; a loader that found
no ``x`` and quietly used ones, or degrees, would be handing a model a
representation nobody chose and reporting the score as if somebody had. When the
artifact says ``none``, ``HeteroData`` gets ``num_nodes`` per type and no ``x``,
which is what PyG is designed to accept for a structure-only graph.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ._graph import Graph
from ._graph_io import GraphInput, coerce_graph_json

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from torch_geometric.data import HeteroData

SCHEMA = "plekton.hetero-graph/v1"

__all__ = ["SCHEMA", "FeaturesNotProvidedError", "to_hetero_data"]


class FeaturesNotProvidedError(RuntimeError):
    """Raised when a caller asks for features an artifact does not carry.

    Deliberately an error rather than a silent zero tensor. An entity graph's
    attributes are an open string map; turning one into a feature vector is a
    modelling decision, and making it here would put it somewhere nobody looks.
    """


def to_hetero_data(graph: GraphInput, *, require_features: bool = False) -> HeteroData:
    """Build a :class:`HeteroData` from a heterogeneous graph artifact.

    Parameters
    ----------
    graph
        A path, a JSON string, or a parsed dict, as elsewhere in this package.
    require_features
        Raise :class:`FeaturesNotProvidedError` rather than returning a
        structure-only graph. Off by default: a structure-only graph is the
        honest product of an entity graph, and a caller who needs features
        should have to say so.

    Raises
    ------
    TypeError
        If ``graph`` is a :class:`plekton.Graph`: that type is homogeneous, with no
        node or relation types to map; ``to_pyg(graph).to_heterogeneous()`` is
        PyG's own conversion.
    ValueError
        If the document is not ``plekton.hetero-graph/v1``.
    FeaturesNotProvidedError
        If ``require_features`` and the artifact carries none.

    """
    if isinstance(graph, Graph):
        msg = (
            "to_hetero_data reads plekton.hetero-graph/v1 documents (typed entity graphs); "
            "a plekton.Graph is homogeneous. Use plekton.pyg.to_pyg(graph).to_heterogeneous()"
        )
        raise TypeError(msg)
    import torch  # noqa: PLC0415 - torch is an optional extra, not a base-package dependency
    from torch_geometric.data import (  # noqa: PLC0415 - torch_geometric is an optional extra
        HeteroData,
    )

    document: dict[str, Any] = json.loads(coerce_graph_json(graph))
    schema = document.get("schema")
    if schema != SCHEMA:
        msg = (
            f"this reads {SCHEMA!r}; that document says {schema!r}. "
            "`plekton_entity::hetero::to_hetero` writes this format."
        )
        raise ValueError(msg)

    stance = (document.get("features") or {}).get("stance")
    if require_features and stance != "provided":
        why = (document.get("features") or {}).get("why", "")
        msg = (
            f"the artifact carries no node features (stance={stance!r}). {why} "
            "Pass require_features=False to work with the structure alone."
        )
        raise FeaturesNotProvidedError(msg)

    data = HeteroData()
    for node_type, ids in (document.get("nodes") or {}).items():
        # `num_nodes` and no `x`: PyG's own representation of a structure-only
        # node type. Setting an `x` of ones here is the exact substitution this
        # module exists to refuse.
        data[node_type].num_nodes = len(ids)
        data[node_type].ids = list(ids)

    for edge in document.get("edges") or []:
        key = (edge["src_type"], edge["relation"], edge["dst_type"])
        index = edge["edge_index"]
        data[key].edge_index = torch.tensor(index, dtype=torch.long)

    return data
