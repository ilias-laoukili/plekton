# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Parameter sweep utility for ``plekton``.

Generates graphs over a cross-product of parameter values by calling
the compiled Rust engine in a loop. Every graph gets an explicit seed, recorded
in its entry's ``config["seed"]``, so any graph of a sweep can be regenerated.

With a base seed — ``base_seed``, or else the ``"seed"`` of ``base`` — each
graph's seed is derived from that base seed, the graph's own configuration
(without its seed) and its replica number, through SHA-256 (see
:func:`plekton._seed.derive_seed`). Adding a value to the grid therefore leaves
the seed, and the graph, of every other entry unchanged, and the same entry gets
the same seed in every process. Without a base seed, each graph draws a fresh one.

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
from plekton._seed import derive_seed, resolve_seed


def _base_seed(base: dict[str, Any], base_seed: int | None) -> int | None:
    """Return the seed every graph's seed derives from, or ``None`` to draw each one.

    ``base_seed`` comes first; when it is ``None`` the ``"seed"`` of ``base`` stands
    in for it, so a seeded base config gives distinct, reproducible replicas rather
    than copies of a single graph. Both given and different is ambiguous: refused.
    """
    in_base = base.get("seed")
    if base_seed is None:
        return None if in_base is None else resolve_seed(in_base)
    chosen = resolve_seed(base_seed)
    if in_base is not None and resolve_seed(in_base) != chosen:
        msg = f"base_seed={chosen} contradicts base['seed']={in_base!r}; give only one of them"
        raise ValueError(msg)
    return chosen


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
        The seed every graph's seed derives from, together with the graph's
        configuration (without its seed) and its replica number — see the module
        docstring. ``None`` uses the ``"seed"`` of ``base`` instead when it has one,
        and otherwise draws a fresh seed for every graph. Either way the seed that
        ran is in the entry's ``config["seed"]``.
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
        If ``count_per_config < 1``, if ``base_seed`` and the ``"seed"`` of
        ``base`` are both given and differ, or if a seeded configuration holds a
        NaN or an infinity, which has no JSON form to derive a seed from.

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
    root = _base_seed(base, base_seed)

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

    # newline="\n" on every write: text mode would otherwise write \r\n on Windows.
    with contextlib.ExitStack() as _stack:
        if _jsonl_mode and _out_path is not None:
            _jsonl_file = _stack.enter_context(
                _out_path.open("w", encoding="utf-8", newline="\n"),
            )
        for combo in combos:
            for replica in range(count_per_config):
                config = json.loads(json.dumps(base))
                config.setdefault("parameters", {})
                for k, v in zip(keys, combo, strict=True):
                    config["parameters"][k] = v
                config["seed"] = (
                    resolve_seed(None) if root is None else derive_seed(root, config, replica)
                )

                graph_json = _rs.generate_from_config(json.dumps(config))
                entry = {"config": config, "graph": json.loads(graph_json)}
                results.append(entry)

                # ── Stream to JSONL immediately ───────────────────────────────
                if _jsonl_file is not None:
                    _jsonl_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
                    _jsonl_file.flush()

    # ── Write JSON array (non-streaming mode) ─────────────────────────────────
    if _out_path is not None and not _jsonl_mode:
        _out_path.write_text(json.dumps(results, indent=2), encoding="utf-8", newline="\n")

    return results
