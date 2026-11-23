# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""The heterogeneous bridge: what it builds, and what it refuses to invent."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from plekton.hetero import SCHEMA, FeaturesNotProvidedError, to_hetero_data

_N_USERS = 2


def artifact() -> dict[str, object]:
    """Build a miniature ``plekton.hetero-graph/v1`` document."""
    return {
        "schema": SCHEMA,
        "nodes": {
            "user": ["shop:user:alice", "shop:user:bob"],
            "item": ["shop:item:book1"],
        },
        "edges": [
            {
                "src_type": "user",
                "relation": "bought",
                "dst_type": "item",
                "edge_index": [[0, 1], [0, 0]],
            },
        ],
        "features": {"stance": "none", "why": "an entity graph's attributes are a string map"},
    }


def test_canonical_triples_survive_the_crossing() -> None:
    """Canonical triples survive the crossing."""
    data = to_hetero_data(artifact())
    assert set(data.node_types) == {"user", "item"}
    assert ("user", "bought", "item") in data.edge_types
    assert data["user"].num_nodes == _N_USERS
    assert data[("user", "bought", "item")].edge_index.shape == (2, 2)


def test_no_features_are_invented() -> None:
    """A structure-only graph must not acquire an ``x`` on the way through."""
    data = to_hetero_data(artifact())
    for node_type in data.node_types:
        assert "x" not in data[node_type], f"{node_type} acquired features"


def test_a_caller_who_needs_features_is_refused_rather_than_given_zeros() -> None:
    """A caller who needs features is refused rather than given zeros."""
    with pytest.raises(FeaturesNotProvidedError) as raised:
        to_hetero_data(artifact(), require_features=True)
    assert "carries no node features" in str(raised.value)


def test_another_schema_is_refused_by_name() -> None:
    """Another schema is refused by name."""
    document = artifact() | {"schema": "plekton.graph/v1"}
    with pytest.raises(ValueError, match=SCHEMA) as raised:
        to_hetero_data(document)
    assert SCHEMA in str(raised.value)


def test_the_ids_travel_so_a_prediction_can_be_named() -> None:
    """Original entity ids are preserved on each node type."""
    data = to_hetero_data(artifact())
    assert data["user"].ids[1] == "shop:user:bob"
