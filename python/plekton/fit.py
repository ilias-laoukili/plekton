# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Blockmodel estimation and community detection (wrapping ``plekton-fit``).

Example::

    import plekton as gg

    edge_index, labels, _ = gg.generate_sbm(200, 4, 0.2, 0.01, seed=0)
    # Canonical undirected edges [E, 2]
    mask = edge_index[0] < edge_index[1]
    edges = edge_index[:, mask].T
    fitted = gg.fit.fit_blockmodel(edges, n_nodes=200, labels=labels)
    edge_index2, labels2, _ = gg.generate_sbm_from_fit(
        200, fitted["block_sizes"], fitted["b_matrix"], fitted["theta"], seed=1
    )

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from plekton._plekton import fit as _ext

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

_EDGE_PAIR_NDIM = 2

__all__ = [
    "fit_blockmodel",
    "label_propagation",
    "overlapping_communities",
]


def _edges(edges: npt.ArrayLike) -> np.ndarray:
    """Coerce ``edges`` to a contiguous ``int64`` array of shape ``[E, 2]``."""
    arr = np.ascontiguousarray(edges, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, _EDGE_PAIR_NDIM), dtype=np.int64)
    if arr.ndim != _EDGE_PAIR_NDIM or arr.shape[1] != _EDGE_PAIR_NDIM:
        msg = f"edges must have shape [E, 2], got {arr.shape}"
        raise ValueError(msg)
    return arr


def fit_blockmodel(
    edges: npt.ArrayLike,
    n_nodes: int,
    labels: npt.ArrayLike | None = None,
    *,
    attribute: str = "labels",
    seed: int = 0,
    max_rounds: int = 30,
) -> dict[str, Any]:
    """Estimate DC-SBM parameters (block sizes, K×K mixing matrix, per-node θ).

    When ``labels`` is supplied, fits conditional on that declared partition.
    When ``labels`` is ``None``, detects a partition via seeded label propagation
    first and records ``provenance="detected"``.
    """
    lbl_arr = None if labels is None else np.ascontiguousarray(labels, dtype=np.int64)
    result: dict[str, Any] = _ext.fit_blockmodel(
        _edges(edges),
        n_nodes,
        lbl_arr,
        attribute,
        seed,
        max_rounds,
    )
    return result


def label_propagation(
    edges: npt.ArrayLike,
    n_nodes: int,
    *,
    seed: int = 0,
    max_rounds: int = 30,
) -> tuple[np.ndarray, bool]:
    """Asynchronous seeded label propagation; returns ``(labels [N], converged)``."""
    lbls, converged = _ext.label_propagation(_edges(edges), n_nodes, seed, max_rounds)
    return np.asarray(lbls), bool(converged)


def overlapping_communities(
    n_nodes: int,
    edges: npt.ArrayLike,
    *,
    seed: int = 0,
    rounds: int = 100,
    threshold: float = 0.2,
) -> list[list[int]]:
    """Detect overlapping communities via SLPA (Xie, Szymanski & Liu, 2011)."""
    pairs = [(int(u), int(v)) for u, v in _edges(edges).tolist()]
    result: list[list[int]] = _ext.overlapping_communities(n_nodes, pairs, seed, rounds, threshold)
    return result
