# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Python face of ``plekton`` — compiled Rust engine with zero-copy NumPy arrays.

The base package is deliberately lean: ``import plekton`` pulls only ``numpy``
(never ``torch`` or ``torch_geometric``). It exposes:

- Synthetic graph generators: ``generate_sbm``, ``generate_sbm_from_fit``,
  ``generate_csbm``, ``generate_ba``, ``mixing_graph``, ``generate_from_config``.
- Canonical JSON serialization: ``to_json``, ``json_to_pyg``.
- Blockmodel estimation & overlapping communities: ``plekton.fit``,
  ``overlapping_communities``.
- Structural graph metrics: ``plekton.metrics``.
- Edge scoring & spectral/backbone sparsification: ``plekton.sparsify``.
- Leakage-free group & temporal ML splitters: ``plekton.splits``.

Optional extras::

    from plekton.pyg import to_pyg              # pip install plekton[pyg]
    from plekton.hetero import to_hetero_data   # pip install plekton[pyg]
    from plekton.nx import to_networkx          # pip install plekton[networkx]
    from plekton.presets import get_preset, list_presets
    from plekton.sweep import sweep
    from plekton.distributions import fit_from_dataset, generate_realistic

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plekton import fit, metrics, sparsify, splits
from plekton._plekton import (
    generate_ba,
    generate_csbm,
    generate_from_config,
    generate_sbm,
    generate_sbm_from_fit,
    json_to_pyg,
    overlapping_communities,
)
from plekton._plekton import (
    graph_to_json as _graph_to_json,
)
from plekton.mixing import generate_mixing_graph_json, mixing_graph

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike


def to_json(
    edge_index: ArrayLike,
    labels: ArrayLike,
    x: ArrayLike | None = None,
    *,
    path: str | Path | None = None,
) -> str:
    """Serialise generator NumPy arrays into canonical ``plekton.graph/v1`` JSON.

    Works uniformly with outputs from :func:`generate_sbm`,
    :func:`generate_sbm_from_fit`, :func:`generate_csbm`, and
    :func:`generate_ba`. Accepts either bidirectional ``[2, 2E]`` or canonical
    ``[2, E]`` ``edge_index`` arrays. When *path* is given, the JSON document
    is also written to that file.
    """
    ei = np.ascontiguousarray(edge_index, dtype=np.int64)
    lbl = np.ascontiguousarray(labels, dtype=np.int64)
    feats = None if x is None else np.ascontiguousarray(x, dtype=np.float32)
    out_path = None if path is None else str(path)
    return str(_graph_to_json(ei, lbl, feats, out_path))


__all__ = [
    "fit",
    "generate_ba",
    "generate_csbm",
    "generate_from_config",
    "generate_mixing_graph_json",
    "generate_sbm",
    "generate_sbm_from_fit",
    "json_to_pyg",
    "metrics",
    "mixing_graph",
    "overlapping_communities",
    "sparsify",
    "splits",
    "to_json",
]
