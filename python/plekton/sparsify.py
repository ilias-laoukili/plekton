# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Edge scoring and graph sparsification (the ``sparsify`` engine).

Pure-Rust, multi-threaded (``rayon``), zero-copy over NumPy. Always available::

    import numpy as np
    from plekton import sparsify

    edges = np.array([[0, 1], [0, 2], [1, 2], [0, 3]], dtype=np.int64)  # [E, 2], u < v
    scores = sparsify.similarity_scores(edges, n_nodes=4, metric="jaccard")
    keep = sparsify.select_mask(edges, 4, scores, "threshold", retention_ratio=0.5)
    sparse_edges = edges[keep]

The functions accept any array-like ``edges`` (coerced to a contiguous
``int64[E, 2]``), ``features`` (``float32[n_nodes, F]``), and score/distance
vectors (``float64[E]``). Endpoints must lie in ``[0, n_nodes)`` or an
``IndexError`` is raised. See ``plekton-metrics`` / ``plekton-sparsify`` (Rust)
for the algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plekton._plekton import sparsify as _ext

# An edge list is (n_edges, 2) — two dimensions and two endpoints.
_EDGE_PAIR_NDIM = 2

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

__all__ = [
    "approx_effective_resistance",
    "metric_backbone_mask",
    "select_mask",
    "similarity_scores",
]


def _edges(edges: npt.ArrayLike) -> np.ndarray:
    """Coerce ``edges`` to a contiguous ``int64`` array of shape ``[E, 2]``."""
    arr = np.ascontiguousarray(edges, dtype=np.int64)
    if arr.ndim != _EDGE_PAIR_NDIM or arr.shape[1] != _EDGE_PAIR_NDIM:
        msg = f"edges must have shape [E, 2], got {arr.shape}"
        raise ValueError(msg)
    return arr


def _f64(values: npt.ArrayLike) -> np.ndarray:
    """Coerce a per-edge score/distance vector to contiguous ``float64``."""
    return np.ascontiguousarray(values, dtype=np.float64)


def similarity_scores(
    edges: npt.ArrayLike,
    n_nodes: int,
    metric: str,
    features: npt.ArrayLike | None = None,
    *,
    ignore_self_loops: bool = True,
) -> np.ndarray:
    """Per-edge similarity scores for ``metric``.

    ``metric`` is one of ``"jaccard"``, ``"adamic_adar"``, ``"degree_product"``,
    or ``"feature_cosine"``. ``feature_cosine`` requires ``features`` (a
    ``float32[n_nodes, F]`` matrix); the structural metrics ignore it.
    ``ignore_self_loops`` defaults to ``True`` so a self-loop never makes a node
    its own neighbour.
    Returns ``float64[E]`` aligned with ``edges``.
    """
    feats = None
    if features is not None:
        feats = np.ascontiguousarray(features, dtype=np.float32)
    return np.asarray(
        _ext.similarity_scores(_edges(edges), n_nodes, metric, feats, ignore_self_loops),
    )


def approx_effective_resistance(
    edges: npt.ArrayLike,
    n_nodes: int,
    epsilon: float = 0.3,
    seed: int = 42,
    max_dim: int | None = None,
) -> np.ndarray:
    """Per-edge approximate effective resistance (Spielman-Srivastava).

    ``epsilon`` controls the Johnson-Lindenstrauss approximation error;
    ``max_dim`` optionally caps the projection dimension so the method stays
    tractable on large subgraphs. Returns ``float64[E]`` (higher = more critical
    edge). Raises ``ValueError`` if ``epsilon`` is not a finite number ``> 0``,
    or if it is so small that the projection cannot be allocated.
    """
    return np.asarray(
        _ext.approx_effective_resistance(_edges(edges), n_nodes, epsilon, seed, max_dim),
    )


def select_mask(  # noqa: PLR0913 - the edge set, the score, and the selection policy, which vary independently
    edges: npt.ArrayLike,
    n_nodes: int,
    scores: npt.ArrayLike,
    strategy: str,
    retention_ratio: float,
    *,
    keep_lowest: bool = False,
    min_per_node: int = 1,
    seed: int = 42,
    preserve_connectivity: bool = True,
) -> np.ndarray:
    """Select edges under ``strategy``; return a ``bool[E]`` keep-mask.

    ``strategy`` is one of ``"threshold"``, ``"degree_aware"``, or ``"sampled"``.
    ``scores`` are per-edge weights (from :func:`similarity_scores` or
    :func:`approx_effective_resistance`). Per-strategy parameters: ``keep_lowest``
    (threshold), ``min_per_node`` (degree_aware), ``seed`` (sampled). When
    ``preserve_connectivity`` is ``True`` no originally-connected node is
    isolated. Apply the mask with ``edges[mask]``.
    """
    return np.asarray(
        _ext.select_mask(
            _edges(edges),
            n_nodes,
            _f64(scores),
            strategy,
            retention_ratio,
            keep_lowest,
            min_per_node,
            seed,
            preserve_connectivity,
        ),
    )


def metric_backbone_mask(
    edges: npt.ArrayLike,
    n_nodes: int,
    distances: npt.ArrayLike,
    epsilon: float = 1e-9,
    *,
    distances_are_similarity: bool = False,
) -> np.ndarray:
    """Metric-backbone selection; return a ``bool[E]`` keep-mask.

    Keeps only shortest-path-preserving edges: the metric backbone of Simas, Correia
    & Rocha, *J. Complex Netw.* 9(6), cnab021 (2021),
    https://doi.org/10.1093/comnet/cnab021. ``distances``
    are per-edge weights; pass ``distances_are_similarity=True`` to feed raw
    similarity scores (they are inverted to distances internally). The retained
    fraction is determined by graph structure, not a ratio.
    """
    return np.asarray(
        _ext.metric_backbone_mask(
            _edges(edges),
            n_nodes,
            _f64(distances),
            epsilon,
            distances_are_similarity,
        ),
    )
