# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.sweep`` — parameter sweep utility.

All tests require the compiled Rust extension (``maturin develop``) and are
guarded with a ``requires_plekton`` skip mark.

Run with:  python -m pytest python/tests/test_sweep.py -v
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from plekton._seed import derive_seed
from plekton.sweep import sweep

_N_RESULTS = 6
_N_REPLICAS = 3

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


def _seeds_by_entry(results: list[dict[str, Any]], count_per_config: int) -> dict[Any, int]:
    """Map ``(homophily, replica)`` to the seed each entry of a sweep ran with."""
    return {
        (entry["config"]["parameters"]["homophily"], index % count_per_config): entry["config"][
            "seed"
        ]
        for index, entry in enumerate(results)
    }


@requires_plekton
def test_base_seed_gives_each_entry_its_derived_seed() -> None:
    """Each seed comes from the base seed, the entry's config and its replica: all differ."""
    results = sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.5]},
        count_per_config=_N_REPLICAS,
        base_seed=0,
    )
    seeds = [entry["config"]["seed"] for entry in results]
    assert seeds == [
        derive_seed(0, entry["config"], replica) for replica, entry in enumerate(results)
    ]
    assert len(set(seeds)) == _N_REPLICAS


@requires_plekton
def test_adding_a_grid_value_keeps_every_existing_seed() -> None:
    """A value inserted in the middle of the grid shifts no other entry's seed."""
    before = sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.3, 0.6]},
        count_per_config=2,
        base_seed=7,
    )
    after = sweep(
        base=copy.deepcopy(_BASE_CONFIG),
        sweep_params={"homophily": [0.3, 0.45, 0.6]},
        count_per_config=2,
        base_seed=7,
    )
    seeds_before = _seeds_by_entry(before, 2)
    seeds_after = _seeds_by_entry(after, 2)
    assert seeds_before.items() <= seeds_after.items()
    # Same seeds, so the entries both sweeps share are the same graphs.
    kept = [entry for entry in after if entry["config"]["parameters"]["homophily"] != 0.45]
    assert [entry["graph"] for entry in kept] == [entry["graph"] for entry in before]


def test_a_seed_is_the_same_in_two_processes() -> None:
    """The derivation is a fixed function of bytes, not Python's per-process ``hash()``."""
    config = {"model_type": "sbm", "parameters": {"n_nodes": 50, "homophily": 0.4}}
    script = (
        "import json, sys\n"
        "from plekton._seed import derive_seed\n"
        "print(derive_seed(7, json.loads(sys.argv[1]), 1))\n"
    )
    seeds = set()
    for hash_seed in ("1", "2"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        run = subprocess.run(  # noqa: S603 - our own interpreter and script
            [sys.executable, "-c", script, json.dumps(config)],
            capture_output=True,
            check=True,
            env=environment,
            text=True,
        )
        seeds.add(int(run.stdout))
    assert seeds == {derive_seed(7, config, 1)}


def test_the_seed_is_the_sha256_of_the_canonical_entry() -> None:
    """The hashed bytes, written out: canonical JSON of base seed, config, replica."""
    config = {
        "model_type": "sbm_classique",
        "seed": 123,  # left out of the hash: it is what the hash produces
        "parameters": {"n_nodes": 50, "n_communities": 2, "homophily": 0.5, "avg_degree": 4.0},
    }
    payload = (
        b'[7,{"model_type":"sbm_classique","parameters":'
        b'{"avg_degree":4.0,"homophily":0.5,"n_communities":2,"n_nodes":50}},1]'
    )
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert derive_seed(7, config, 1) == expected == 7_923_242_528_039_645_024


def test_the_whole_config_counts() -> None:
    """Another model type, another key: another seed. ``4`` and ``4.0`` are not the same."""
    parameters = {"n_nodes": 50, "avg_degree": 4.0}
    sbm = derive_seed(7, {"model_type": "sbm", "parameters": parameters}, 0)
    assert sbm != derive_seed(7, {"model_type": "csbm", "parameters": parameters}, 0)
    assert sbm != derive_seed(7, {"model_type": "sbm", "parameters": parameters, "x": 1}, 0)
    assert sbm != derive_seed(
        7,
        {"model_type": "sbm", "parameters": {"n_nodes": 50, "avg_degree": 4}},
        0,
    )
    with pytest.raises(ValueError, match="JSON"):
        derive_seed(7, {"parameters": {"mu": float("nan")}}, 0)


@requires_plekton
def test_a_seed_above_two_to_the_63_runs() -> None:
    """A derived seed spans [0, 2**64); Rust takes the top of the range as a u64."""
    config = {**copy.deepcopy(_BASE_CONFIG), "seed": 2**64 - 1}
    graph = json.loads(_rs.generate_from_config(json.dumps(config)))
    assert graph["metadata"]["n_nodes"] == 50


@requires_plekton
def test_a_seed_in_the_base_config_is_the_base_seed() -> None:
    """``base["seed"]`` without ``base_seed`` derives distinct seeds, not copies of one graph."""
    base = {**copy.deepcopy(_BASE_CONFIG), "seed": 5}
    results = sweep(base=base, sweep_params={"homophily": [0.5]}, count_per_config=_N_REPLICAS)
    seeds = [entry["config"]["seed"] for entry in results]
    assert len(set(seeds)) == _N_REPLICAS
    assert seeds == [
        derive_seed(5, entry["config"], replica) for replica, entry in enumerate(results)
    ]
    # The same seed given both ways is no contradiction.
    agreed = sweep(base=base, sweep_params={"homophily": [0.5]}, count_per_config=1, base_seed=5)
    assert agreed[0]["config"]["seed"] == seeds[0]


def test_two_different_base_seeds_are_refused() -> None:
    """``base_seed`` and ``base["seed"]`` that differ leave no seed to trust: refused."""
    base = {**copy.deepcopy(_BASE_CONFIG), "seed": 5}
    with pytest.raises(ValueError, match="contradicts"):
        sweep(base=base, sweep_params={"homophily": [0.5]}, base_seed=6)


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
