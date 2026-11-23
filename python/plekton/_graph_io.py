# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Shared input coercion for the graph converters.

Both :func:`plekton.pyg.to_pyg` and :func:`plekton.nx.to_networkx` accept
the same three input forms and normalise them to a single canonical graph-JSON
string before parsing. That normalisation lives here — in a dependency-light
module — so ``pyg.py`` (torch) and ``nx.py`` (networkx) can share it without
importing each other or pulling in a heavy dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

GraphInput = Union[str, "os.PathLike[str]", dict[str, object]]


def coerce_graph_json(graph: GraphInput) -> str:
    """Normalise a graph argument to a canonical graph-JSON string.

    Accepted forms:

    * ``dict`` — a parsed ``{ metadata, nodes, edges }`` document → re-serialised.
    * ``os.PathLike``, or a ``str`` ending in ``.json`` that exists on disk → the
      file is read and returned verbatim.
    * any other ``str`` — assumed to already be JSON content and returned as-is.

    Parameters
    ----------
    graph:
        A JSON string, a parsed ``dict``, or a path to a ``.json`` file produced
        by ``generate_*_json`` or ``generate_from_config``.

    Returns
    -------
    str
        Canonical graph JSON: ``{ metadata, nodes:[{id, community, features?}],
        edges:[{source, target}] }``.

    """
    if isinstance(graph, dict):
        return json.dumps(graph)

    if isinstance(graph, os.PathLike):
        with Path(graph).open(encoding="utf-8") as fh:
            return fh.read()

    if isinstance(graph, str):
        if graph.endswith(".json") and Path(graph).is_file():
            with Path(graph).open(encoding="utf-8") as fh:
                return fh.read()
        return graph

    msg = f"graph must be a JSON string, a dict, or a .json path; got {type(graph).__name__}"
    raise TypeError(msg)
