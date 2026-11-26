# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Blockmodel estimation and community detection (wrapping ``plekton-fit``).

Every function takes an undirected :class:`plekton.Graph` first, or an ``(E, 2)``
edge array with ``n_nodes=`` by keyword. Example::

    import plekton

    graph = plekton.generate_sbm(200, 4, 0.2, 0.01, seed=0)
    fitted = plekton.fit.fit_blockmodel(graph)            # partition: graph.labels
    regenerated = plekton.generate_sbm_from_fit(
        200, fitted["block_sizes"], fitted["b_matrix"], theta=fitted["theta"], seed=1
    )
    detected = plekton.fit.fit_blockmodel(graph, labels="detect", seed=2)

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from plekton._graph import Graph, as_graph, node_labels
from plekton._plekton import fit as _ext
from plekton._seed import resolve_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

__all__ = [
    "fit_blockmodel",
    "label_propagation",
    "overlapping_communities",
]

# The detector seed is unused for a declared partition; the binding still takes one.
_UNUSED_SEED = 0


def fit_blockmodel(
    graph: Graph | npt.ArrayLike,
    *,
    labels: npt.ArrayLike | Literal["detect"] | None = None,
    attribute: str = "labels",
    seed: int | None = None,
    max_rounds: int = 30,
    n_nodes: int | None = None,
) -> dict[str, Any]:
    """Estimate DC-SBM parameters (block sizes, K×K mixing matrix, per-node θ).

    The partition is never inferred silently:

    * ``labels=None`` (default) fits ``graph.labels``, a declared partition, and
      raises if the graph carries none;
    * an array fits that declared partition (``attribute`` names its source);
    * ``labels="detect"`` runs seeded label propagation first and records
      ``provenance="detected"``. ``seed=None`` draws a seed, and the result's
      ``"seed"`` is the one that ran.

    ``seed`` only applies to ``labels="detect"``.
    """
    g = as_graph(graph, n_nodes, caller="fit.fit_blockmodel")
    if isinstance(labels, str):
        if labels != "detect":
            msg = f"labels must be an array, None (use graph.labels) or 'detect', got {labels!r}"
            raise ValueError(msg)
        used = resolve_seed(seed)
        detected: dict[str, Any] = _ext.fit_blockmodel(
            g.edges,
            g.n_nodes,
            None,
            attribute,
            used,
            max_rounds,
        )
        detected["seed"] = used
        return detected
    if seed is not None:
        msg = "fit.fit_blockmodel: seed only applies to labels='detect'"
        raise ValueError(msg)
    declared = node_labels(
        g,
        labels,
        caller="fit.fit_blockmodel",
        hint="pass labels=..., or labels='detect' to run label propagation",
    )
    result: dict[str, Any] = _ext.fit_blockmodel(
        g.edges,
        g.n_nodes,
        declared,
        attribute,
        _UNUSED_SEED,
        max_rounds,
    )
    result["seed"] = None
    return result


def label_propagation(
    graph: Graph | npt.ArrayLike,
    *,
    seed: int | None = None,
    max_rounds: int = 30,
    n_nodes: int | None = None,
) -> tuple[np.ndarray, bool]:
    """Asynchronous seeded label propagation; returns ``(labels [N], converged)``.

    ``seed=None`` draws a seed; pass one explicitly to reproduce the result.
    """
    g = as_graph(graph, n_nodes, caller="fit.label_propagation")
    lbls, converged = _ext.label_propagation(g.edges, g.n_nodes, resolve_seed(seed), max_rounds)
    return np.asarray(lbls), bool(converged)


def overlapping_communities(
    graph: Graph | npt.ArrayLike,
    *,
    seed: int | None = None,
    rounds: int = 100,
    threshold: float = 0.2,
    n_nodes: int | None = None,
) -> list[list[int]]:
    """Detect overlapping communities via SLPA (Xie, Szymanski & Liu, 2011).

    ``seed=None`` draws a seed; pass one explicitly to reproduce the result.
    """
    g = as_graph(graph, n_nodes, caller="fit.overlapping_communities")
    result: list[list[int]] = _ext.overlapping_communities(
        g.edges,
        g.n_nodes,
        resolve_seed(seed),
        rounds,
        threshold,
    )
    return result
