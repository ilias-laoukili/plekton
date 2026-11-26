# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.sweep`` — parameter sweep utility.

All tests require the compiled Rust extension (``maturin develop``) and are
guarded with a ``requires_plekton`` skip mark.

Run with:  python -m pytest python/tests/test_sweep.py -v
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest
from plekton.sweep import sweep

_THIRD_SEED = 2
_N_RESULTS = 6

if TYPE_CHECKING:
    from pathlib import Path

try:
    import plekton as _rs

    _HAS_PLEKTON = True
except ImportError:
    _HAS_PLEKTON = False

requires_plekton = pytest.mark.skipif(
    not _HAS_PLEKTON,
    reason="plekton Rust extension not installed (run `maturin develop`)",
)

# ─── Shared fixture ───────────────────────────────────────────────────────────

_BASE_CONFIG: dict[str, object] = {
    "model_type": "sbm_classique",
    "parameters": {
        "n_nodes": 50,
        "n_communities": 2,
        "homophily": 0.7,
        "avg_degree": 4.0,
    },
}

# ─── Tests ────────────────────────────────────────────────────────────────────


@requires_plekton
def test_sweep_returns_correct_entry_count() -> None:
    """3 homophily values × 2 replicas per config = 6 result entries."""
    results = sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.2, 0.5, 0.8]},
        count_per_config=2,
        base_seed=0,
    )
    assert len(results) == _N_RESULTS, f"expected _N_RESULTS entries, got {len(results)}"


@requires_plekton
def test_base_dict_is_not_mutated() -> None:
    """The ``base`` dict must remain unchanged after the sweep completes."""
    base = copy.deepcopy(_BASE_CONFIG)
    original_base = copy.deepcopy(base)

    sweep(
        base=base,
        sweep_params={"homophily": [0.3, 0.7]},
        count_per_config=1,
    )
    assert base == original_base, "sweep() must not mutate the caller's base dict"


@requires_plekton
def test_seed_management_is_deterministic() -> None:
    """With ``base_seed=0``, the first entry uses seed 0 and the second uses seed 1."""
    results = sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.5]},
        count_per_config=3,
        base_seed=0,
    )
    assert results[0]["config"]["seed"] == 0, "first entry must use seed 0"
    assert results[1]["config"]["seed"] == 1, "second entry must use seed 1"
    assert results[_THIRD_SEED]["config"]["seed"] == _THIRD_SEED, (
        "third entry must use seed _THIRD_SEED"
    )


def test_count_per_config_zero_raises() -> None:
    """``count_per_config=0`` must raise ``ValueError`` immediately."""
    with pytest.raises(ValueError, match="count_per_config"):
        sweep(
            base=copy.deepcopy(_BASE_CONFIG),
            sweep_params={"homophily": [0.5]},
            count_per_config=0,
        )


@requires_plekton
def test_jsonl_output_written_per_entry(tmp_path: Path) -> None:
    """With a ``.jsonl`` output path, each entry must appear as one line in the file."""
    output_file = tmp_path / "results.jsonl"
    results = sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.3, 0.6]},
        count_per_config=1,
        base_seed=7,
        output_path=str(output_file),
    )

    assert output_file.exists(), "JSONL output file must be created"
    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results), (
        f"JSONL file must have {len(results)} lines, got {len(lines)}"
    )
    # Every line must be valid JSON.
    for line_number, line in enumerate(lines):
        parsed = json.loads(line)
        assert "config" in parsed, f"line {line_number} must have a 'config' key"
        assert "graph" in parsed, f"line {line_number} must have a 'graph' key"


@requires_plekton
@pytest.mark.parametrize("name", ["results.jsonl", "results.json"])
def test_output_file_has_no_carriage_return(tmp_path: Path, name: str) -> None:
    r"""Lines end in ``\n`` on every platform: the written bytes hold no ``\r``."""
    # Text mode writes os.linesep, so only a Windows run can catch a missing newline="\n".
    output_file = tmp_path / name
    sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.3, 0.6]},
        count_per_config=1,
        base_seed=7,
        output_path=str(output_file),
    )
    data = output_file.read_bytes()
    assert b"\n" in data
    assert b"\r" not in data


@requires_plekton
def test_without_base_seed_every_graph_records_the_seed_it_ran_with() -> None:
    """With ``base_seed=None`` a seed is drawn per graph, recorded, and replays the graph."""
    results = sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.5]},
        count_per_config=2,
    )
    seeds = [entry["config"]["seed"] for entry in results]
    assert all(isinstance(seed, int) for seed in seeds), seeds
    replay = json.loads(_rs.generate_from_config(json.dumps(results[0]["config"])))
    assert replay == results[0]["graph"]
