# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.presets`` — the built-in benchmark preset catalogue.

Verifies that every advertised preset name resolves, that ``get_preset`` returns
an independent deep copy, and that each preset config is a faithful JSON
round-trip.
"""

from __future__ import annotations

import json

import pytest

_SENTINEL = -999

try:
    from plekton.presets import get_preset, list_presets

    _HAS_PLEKTON = True
except ImportError:
    _HAS_PLEKTON = False

requires_plekton = pytest.mark.skipif(
    not _HAS_PLEKTON,
    reason="plekton extension not installed (run `maturin develop`)",
)


@requires_plekton
def test_list_presets_is_non_empty_and_unique() -> None:
    """List presets is non empty and unique."""
    names = list_presets()
    assert names, "expected at least one preset"
    assert all(isinstance(name, str) for name in names)
    assert len(names) == len(set(names)), "preset names must be unique"


@requires_plekton
def test_every_listed_preset_resolves() -> None:
    """Each advertised name resolves to a well-formed config dict."""
    for name in list_presets():
        cfg = get_preset(name)
        assert isinstance(cfg, dict)
        assert isinstance(cfg["model_type"], str)
        assert cfg["model_type"]
        assert isinstance(cfg["parameters"], dict)
        assert cfg["parameters"]
        assert "n_nodes" in cfg["parameters"]


@requires_plekton
def test_get_preset_unknown_raises_keyerror() -> None:
    """Get preset unknown raises keyerror."""
    with pytest.raises(KeyError):
        get_preset("no-such-preset")


@requires_plekton
def test_get_preset_returns_independent_deep_copy() -> None:
    """Mutating a returned preset must not bleed into later lookups."""
    first = get_preset("sbm-balanced")
    first["parameters"]["n_nodes"] = -999
    second = get_preset("sbm-balanced")
    assert second["parameters"]["n_nodes"] != _SENTINEL


@requires_plekton
def test_preset_config_json_round_trips() -> None:
    """Every preset survives a JSON dump/load unchanged."""
    for name in list_presets():
        cfg = get_preset(name)
        assert json.loads(json.dumps(cfg)) == cfg
