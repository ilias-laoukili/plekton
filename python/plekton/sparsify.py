# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Edge scoring and graph sparsification (the ``sparsify`` engine).

Pure-Rust, multi-threaded (``rayon``), zero-copy over NumPy. Always available::

    import plekton
    from plekton import sparsify

    graph = plekton.generate_sbm(1_000, 4, 0.05, 0.002, seed=0)
    scores = sparsify.similarity_scores(graph, "jaccard")
    keep = sparsify.select_mask(graph, scores, "threshold", retention_ratio=0.5)
    sparse = graph.edge_subgraph(keep)   # same nodes, half the edges

Every function takes an undirected :class:`plekton.Graph` first, or an ``(E, 2)``
edge array with ``n_nodes=`` by keyword. Scores and masks are aligned with
``graph.edges``. See ``plekton-metrics`` / ``plekton-sparsify`` (Rust) for the
algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plekton._graph import Graph, as_graph
from plekton._plekton import sparsify as _ext
from plekton._seed import resolve_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

__all__ = [
    "approx_effective_resistance",
    "metric_backbone_mask",
    "select_mask",
    "similarity_scores",
]


def _f64(values: npt.ArrayLike) -> np.ndarray:
    """Coerce a per-edge score/distance vector to contiguous ``float64``."""
    return np.ascontiguousarray(values, dtype=np.float64)


def similarity_scores(
    graph: Graph | npt.ArrayLike,
    metric: str,
    *,
    features: npt.ArrayLike | None = None,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Per-edge similarity scores for ``metric``.

    ``metric`` is one of ``"jaccard"``, ``"adamic_adar"``, ``"degree_product"``,
    or ``"feature_cosine"``. ``feature_cosine`` uses ``features`` (a
    ``(n_nodes, F)`` matrix), by default ``graph.x``; the structural metrics
    ignore features. Returns ``float64[E]`` aligned with ``graph.edges``.
    """
    g = as_graph(graph, n_nodes, caller="sparsify.similarity_scores")
    feats = None
    if metric == "feature_cosine":
        source = g.x if features is None else features
        if source is None:
            msg = "sparsify.similarity_scores: feature_cosine needs features; the graph has no x"
            raise ValueError(msg)
        feats = np.ascontiguousarray(source, dtype=np.float32)
    return np.asarray(_ext.similarity_scores(g.edges, g.n_nodes, metric, feats))


def approx_effective_resistance(
    graph: Graph | npt.ArrayLike,
    *,
    epsilon: float = 0.3,
    seed: int | None = None,
    max_dim: int | None = None,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Per-edge approximate effective resistance (Spielman-Srivastava).

    ``epsilon`` controls the Johnson-Lindenstrauss approximation error;
    ``max_dim`` optionally caps the projection dimension so the method stays
    tractable on large subgraphs. Returns ``float64[E]`` (higher = more critical
    edge). ``seed=None`` draws a seed; pass one to reproduce the scores. Raises
    ``ValueError`` if ``epsilon`` is not a finite number ``> 0``, or if it is so
    small that the projection cannot be allocated.
    """
    g = as_graph(graph, n_nodes, caller="sparsify.approx_effective_resistance")
    return np.asarray(
        _ext.approx_effective_resistance(g.edges, g.n_nodes, epsilon, resolve_seed(seed), max_dim),
    )


def select_mask(  # noqa: PLR0913 - the graph, the score, and the selection policy, which vary independently
    graph: Graph | npt.ArrayLike,
    scores: npt.ArrayLike,
    strategy: str,
    retention_ratio: float,
    *,
    keep_lowest: bool = False,
    min_per_node: int = 1,
    seed: int | None = None,
    preserve_connectivity: bool = True,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Select edges under ``strategy``; return a ``bool[E]`` keep-mask.

    ``strategy`` is one of ``"threshold"``, ``"degree_aware"``, or ``"sampled"``.
    ``scores`` are per-edge weights (from :func:`similarity_scores` or
    :func:`approx_effective_resistance`). Per-strategy parameters: ``keep_lowest``
    (threshold), ``min_per_node`` (degree_aware), ``seed`` (sampled; ``None``
    draws one). When ``preserve_connectivity`` is ``True`` no originally-connected
    node is isolated. Apply the mask with ``graph.edge_subgraph(mask)``.
    """
    g = as_graph(graph, n_nodes, caller="sparsify.select_mask")
    return np.asarray(
        _ext.select_mask(
            g.edges,
            g.n_nodes,
            _f64(scores),
            strategy,
            retention_ratio,
            keep_lowest,
            min_per_node,
            resolve_seed(seed),
            preserve_connectivity,
        ),
    )


def metric_backbone_mask(
    graph: Graph | npt.ArrayLike,
    distances: npt.ArrayLike,
    *,
    epsilon: float = 1e-9,
    distances_are_similarity: bool = False,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Metric-backbone selection; return a ``bool[E]`` keep-mask.

    Keeps only shortest-path-preserving edges: the metric backbone of Simas, Correia
    & Rocha, *J. Complex Netw.* 9(6), cnab021 (2021),
    https://doi.org/10.1093/comnet/cnab021. ``distances``
    are per-edge weights; pass ``distances_are_similarity=True`` to feed raw
    similarity scores (they are inverted to distances internally). The retained
    fraction is determined by graph structure, not a ratio.
    """
    g = as_graph(graph, n_nodes, caller="sparsify.metric_backbone_mask")
    return np.asarray(
        _ext.metric_backbone_mask(
            g.edges,
            g.n_nodes,
            _f64(distances),
            epsilon,
            distances_are_similarity,
        ),
    )
