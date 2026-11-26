# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Benchmark presets for ``plekton``.

Each preset is a ready-to-use config dict that can be passed directly to
``plekton.generate_from_config``.

Quick start
-----------
>>> import json
>>> import plekton
>>> from plekton.presets import get_preset, list_presets
>>> cfg = get_preset("sbm-balanced")
>>> graph = plekton.Graph.from_json(plekton.generate_from_config(json.dumps(cfg)))
>>> graph.metadata["homophily"]
"""

from __future__ import annotations

import copy
from typing import Any

_PRESETS: dict[str, dict[str, Any]] = {
    "csbm-easy": {
        "model_type": "csbm",
        "parameters": {
            "n_nodes": 1000,
            "n_communities": 5,
            "homophily": 0.8,
            "avg_degree": 10.0,
            "features_dim": 32,
            "mu": 2.0,
        },
    },
    "csbm-hard": {
        "model_type": "csbm",
        "parameters": {
            "n_nodes": 1000,
            "n_communities": 5,
            "homophily": 0.2,
            "avg_degree": 5.0,
            "features_dim": 32,
            "mu": 0.5,
        },
    },
    "hetero": {
        "model_type": "csbm",
        "parameters": {
            "n_nodes": 2000,
            "n_communities": 10,
            "homophily": 0.05,
            "avg_degree": 15.0,
            "features_dim": 64,
            "mu": 0.3,
        },
    },
    "sbm-balanced": {
        "model_type": "sbm_classique",
        "parameters": {
            "n_nodes": 500,
            "n_communities": 4,
            "homophily": 0.85,
            "avg_degree": 8.0,
        },
    },
    "dcsbm-heavy": {
        "model_type": "sbm",
        "parameters": {
            "n_nodes": 1000,
            "n_communities": 5,
            "homophily": 0.7,
            "avg_degree": 8.0,
            "theta_exponent": 2.5,
        },
    },
    "ba-scale-free": {
        "model_type": "ba",
        "parameters": {
            "n_nodes": 1000,
            "n_communities": 5,
            "homophily": 0.5,
            "avg_degree": 0.0,
            "m": 2,
            "m0": 2,
        },
    },
    "csbm-large": {
        "model_type": "csbm",
        "parameters": {
            "n_nodes": 10000,
            "n_communities": 8,
            "homophily": 0.7,
            "avg_degree": 15.0,
            "features_dim": 64,
            "mu": 1.5,
        },
    },
    "sbm-many-comm": {
        "model_type": "sbm_classique",
        "parameters": {
            "n_nodes": 2000,
            "n_communities": 20,
            "homophily": 0.75,
            "avg_degree": 10.0,
        },
    },
    "csbm-sparse": {
        "model_type": "csbm",
        "parameters": {
            "n_nodes": 1000,
            "n_communities": 5,
            "homophily": 0.8,
            "avg_degree": 2.0,
            "features_dim": 16,
            "mu": 1.5,
        },
    },
}


def get_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of the named preset config dict.

    Parameters
    ----------
    name:
        Preset name (see ``list_presets()`` for available names).

    Raises
    ------
    KeyError
        If the preset name is not recognised.

    """
    if name not in _PRESETS:
        msg = f"Unknown preset '{name}'. Available: {', '.join(_PRESETS)}"
        raise KeyError(msg)
    return copy.deepcopy(_PRESETS[name])


def list_presets() -> list[str]:
    """Return a list of all available preset names."""
    return list(_PRESETS.keys())
