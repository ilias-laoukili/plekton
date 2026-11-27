# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""The JSON of ordinary graphs, frozen byte for byte.

Each SHA-256 below was computed on ``main`` (bc3b398) before metrics that are
undefined became ``None``: that change rewrites what is written for a graph with
no node, a single node, no edge or no labels, and must leave a graph with nodes,
edges and labels exactly as it was. A digest that moves means the JSON contract
moved; re-freeze it only deliberately, saying why.
"""

from __future__ import annotations

import hashlib
import json

import pytest

plekton = pytest.importorskip("plekton")


def _sbm(seed: int, **extra: object) -> dict[str, object]:
    """Return an SBM config (DC-SBM with ``theta_exponent``) of 60 nodes in 3 blocks."""
    parameters: dict[str, object] = {
        "n_nodes": 60,
        "n_communities": 3,
        "homophily": 0.8,
        "avg_degree": 6.0,
    }
    parameters.update(extra)
    return {"model_type": "sbm", "seed": seed, "parameters": parameters}


def _ba(seed: int) -> dict[str, object]:
    """Return a Barabási–Albert config of 60 nodes, 2 edges per new node, 3 label blocks."""
    return {
        "model_type": "ba",
        "seed": seed,
        "parameters": {"n_nodes": 60, "n_communities": 3, "m": 2},
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_FROM_CONFIG = {
    "sbm-seed3": (
        _sbm(3),
        "b8ad144c828f6b3630bc29b158f2e65676fe50ee9866ca62d44f7bc0f4e87720",
    ),
    "sbm-seed11": (
        _sbm(11),
        "4c48bd49477c306a45947eee602fae901b6f405a9035a6833b40da26b296ba2f",
    ),
    "dcsbm-seed3": (
        _sbm(3, theta_exponent=2.5),
        "96f052c37f635f012c7871da01653b5a122d90416ee50f9cb3136f3ebcd74c26",
    ),
    "dcsbm-seed11": (
        _sbm(11, theta_exponent=2.5),
        "2945778254e4d504d263d7676d4170d44f6026c819bf5624cd74125b191ce394",
    ),
    "ba-seed3": (
        _ba(3),
        "882717a360c65d32144c1dd86e6e4241c5a9cf07243af9915f6b266fa9f8f521",
    ),
    "ba-seed11": (
        _ba(11),
        "50a48a132b114febc099e5d8a84480fd39c911e77cb5c5ddd87a914031d413a0",
    ),
}

_TO_JSON = {
    3: "8147f22ab24f1bf40a747267c28339509c1cc8d08e450a461058fa5625342865",
    11: "d44f190a61941986de1e0c6fa2604702d2be7ab9ad92a98598f97f90d15e5f99",
}


@pytest.mark.parametrize("name", sorted(_FROM_CONFIG))
def test_generate_from_config_writes_the_same_bytes(name: str) -> None:
    """``generate_from_config`` writes an ordinary graph exactly as before."""
    config, digest = _FROM_CONFIG[name]
    assert _sha256(plekton.generate_from_config(json.dumps(config))) == digest


@pytest.mark.parametrize("seed", sorted(_TO_JSON))
def test_to_json_writes_the_same_bytes(seed: int) -> None:
    """``to_json`` writes an ordinary generated graph exactly as before."""
    graph = plekton.generate_sbm(60, 3, p_in=0.2, p_out=0.02, seed=seed)
    assert _sha256(plekton.to_json(graph)) == _TO_JSON[seed]
