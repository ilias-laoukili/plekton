# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Parameter sweep utility for ``plekton``.

Generates graphs over a cross-product of parameter values by calling
the compiled Rust engine in a loop. Every graph gets an explicit seed, recorded
in its entry's ``config["seed"]``: derived from ``base_seed`` when one is given,
drawn otherwise, so any graph of a sweep can be regenerated.

Streaming output
----------------
Pass ``output_path`` to write results to disk as they are generated, which
lets you recover partial results if the sweep is interrupted:

- Files ending in ``.jsonl`` or ``.ndjson`` → one JSON object per line
  (JSON Lines / newline-delimited JSON).
- All other extensions (e.g. ``.json``) → a JSON array written when the
  sweep completes.

Quick start
-----------
>>> from plekton.sweep import sweep
>>> results = sweep(
...     base={"model_type": "csbm", "parameters": {
...         "n_nodes": 500, "n_communities": 4, "homophily": 0.5,
...         "avg_degree": 8.0, "features_dim": 16, "mu": 1.0,
...     }},
...     sweep_params={"homophily": [0.2, 0.5, 0.8]},
...     count_per_config=10,
...     base_seed=42,
... )
>>> len(results)
30
>>> results[0].keys()
dict_keys(['config', 'graph'])
"""

from __future__ import annotations

import contextlib
import itertools
import json
from pathlib import Path
from typing import Any

import plekton as _rs
from plekton._seed import resolve_seed


def _assign_seed(config: dict[str, Any], base_seed: int | None, index: int) -> None:
    """Give ``config`` the explicit seed it will run with, so the entry records it."""
    if base_seed is not None:
        config["seed"] = base_seed + index
    elif config.get("seed") is None:
        config["seed"] = resolve_seed(None)


def sweep(
    base: dict[str, Any],
    sweep_params: dict[str, list[Any]],
    count_per_config: int = 1,
    base_seed: int | None = None,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    """Generate graphs over a cross-product of parameter values.

    Parameters
    ----------
    base:
        Base config dict (``model_type`` + ``parameters`` keys).
        Copied for each combination — the original is never mutated.
    sweep_params:
        Mapping of parameter name → list of values to sweep over.
        Keys must match fields inside ``base["parameters"]``.
        The cross-product of all lists is enumerated.
    count_per_config:
        Number of graphs to generate for each parameter combination
        (useful for averaging over random seeds).
    base_seed:
        If provided, each graph receives a deterministic seed derived as
        ``base_seed + result_index``, ensuring full reproducibility. If ``None``,
        a graph whose config carries no ``"seed"`` gets a freshly drawn one. Either
        way the seed that ran is in the entry's ``config["seed"]``.
    output_path:
        Optional path to write results to disk.

        - ``.jsonl`` / ``.ndjson`` — one compact JSON object per line,
          written immediately after each graph is generated.  Safe to
          resume after interruption (append the existing file to recover
          partial results).
        - Any other extension (e.g. ``.json``) — a pretty-printed JSON
          array written atomically when the sweep finishes.

        In both cases the in-memory list is also returned.

    Returns
    -------
    list[dict]
        One entry per generated graph: ``{"config": {...}, "graph": {...}}``.
        ``config`` reflects the actual parameters used (including seed).
        ``graph`` is the parsed JSON output from the Rust engine.

    Raises
    ------
    ValueError
        If ``count_per_config < 1``.

    Examples
    --------
    Stream results to a JSON Lines file for large sweeps:

    >>> results = sweep(
    ...     base=get_preset("csbm-easy"),
    ...     sweep_params={"homophily": [0.2, 0.5, 0.8]},
    ...     count_per_config=5,
    ...     base_seed=42,
    ...     output_path="results.jsonl",
    ... )
    >>> len(results)
    15

    """
    if count_per_config < 1:
        msg = "count_per_config must be >= 1"
        raise ValueError(msg)

    keys = sorted(sweep_params.keys())
    combos = list(itertools.product(*[sweep_params[key] for key in keys]))
    results: list[dict[str, Any]] = []

    # ── Determine output mode ─────────────────────────────────────────────────
    _jsonl_mode = False
    _out_path: Path | None = None
    _jsonl_file = None  # open file handle for streaming mode

    if output_path is not None:
        _out_path = Path(output_path)
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        _jsonl_mode = _out_path.suffix.lower() in {".jsonl", ".ndjson"}

    with contextlib.ExitStack() as _stack:
        if _jsonl_mode and _out_path is not None:
            _jsonl_file = _stack.enter_context(_out_path.open("w", encoding="utf-8"))
        for combo in combos:
            for _ in range(count_per_config):
                config = json.loads(json.dumps(base))
                config.setdefault("parameters", {})
                for k, v in zip(keys, combo, strict=True):
                    config["parameters"][k] = v
                _assign_seed(config, base_seed, len(results))

                graph_json = _rs.generate_from_config(json.dumps(config))
                entry = {"config": config, "graph": json.loads(graph_json)}
                results.append(entry)

                # ── Stream to JSONL immediately ───────────────────────────────
                if _jsonl_file is not None:
                    _jsonl_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
                    _jsonl_file.flush()

    # ── Write JSON array (non-streaming mode) ─────────────────────────────────
    if _out_path is not None and not _jsonl_mode:
        _out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results
