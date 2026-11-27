# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Synthetic graph generators. Each one returns a :class:`plekton.Graph`.

The Rust engine hands over the canonical ``(E, 2)`` edges, the labels and, for
the cSBM, the features without copying. ``Graph.metadata`` records the measured
``homophily`` and the ``seed`` that produced the graph: when ``seed`` is
``None`` one is drawn, used and recorded, so every graph can be regenerated.
A graph drawn without any edge has no homophily: ``metadata["homophily"]`` is
then ``None``, and ``metadata["metrics_skipped"]`` says why, as in the JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plekton import _plekton as _ext
from plekton._graph import Graph, trusted_graph
from plekton._seed import resolve_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = ["generate_ba", "generate_csbm", "generate_sbm", "generate_sbm_from_fit"]

# The reason the Rust JSON writer records for a graph without edges, word for word.
HOMOPHILY_WITHOUT_EDGES = "undefined here: the graph has no edges"


def _metadata(homophily: float | None, seed: int) -> dict[str, object]:
    """``{"homophily", "seed"}``, plus ``metrics_skipped`` when the homophily is undefined."""
    metadata: dict[str, object] = {"homophily": homophily, "seed": seed}
    if homophily is None:
        metadata["metrics_skipped"] = [{"metric": "homophily", "reason": HOMOPHILY_WITHOUT_EDGES}]
    return metadata


def generate_sbm(
    n_nodes: int,
    n_communities: int,
    p_in: float,
    p_out: float,
    *,
    theta_exponent: float | None = None,
    seed: int | None = None,
) -> Graph:
    """Stochastic block model, or exact O(N + E) DC-SBM when ``theta_exponent`` is set.

    ``n_nodes`` nodes in ``n_communities`` balanced blocks, intra-block edge
    probability ``p_in`` and inter-block ``p_out``; ``theta_exponent`` (> 2)
    draws Pareto degree weights. Returns an undirected ``Graph`` with
    ``labels`` and ``metadata = {"homophily", "seed"}``.
    """
    used = resolve_seed(seed)
    edges, labels, homophily = _ext.generate_sbm(
        n_nodes,
        n_communities,
        p_in,
        p_out,
        theta_exponent,
        used,
    )
    return trusted_graph(
        edges,
        n_nodes,
        labels=labels,
        metadata=_metadata(homophily, used),
    )


def generate_sbm_from_fit(
    n_nodes: int,
    block_sizes: Sequence[int],
    b_matrix: npt.ArrayLike,
    *,
    theta: npt.ArrayLike | None = None,
    seed: int | None = None,
) -> Graph:
    """Regenerate a graph from a fitted blockmodel ``(block_sizes, b_matrix, theta)``.

    Takes the output of :func:`plekton.fit.fit_blockmodel`. Returns an undirected
    ``Graph`` with ``labels`` and ``metadata = {"homophily", "seed"}``.
    """
    used = resolve_seed(seed)
    b = np.ascontiguousarray(b_matrix, dtype=np.float64)
    weights = None if theta is None else np.ascontiguousarray(theta, dtype=np.float64)
    edges, labels, homophily = _ext.generate_sbm_from_fit(
        n_nodes,
        list(block_sizes),
        b,
        weights,
        used,
    )
    return trusted_graph(
        edges,
        n_nodes,
        labels=labels,
        metadata=_metadata(homophily, used),
    )


def generate_ba(
    n_nodes: int,
    *,
    m: int = 2,
    m0: int | None = None,
    n_communities: int = 1,
    seed: int | None = None,
) -> Graph:
    """Barabási–Albert preferential attachment: ``m`` edges per new node.

    Growth starts from a clique of ``m0`` nodes (default ``m``). Labels are
    sequential blocks of ``n_nodes / n_communities`` nodes. Returns an undirected
    ``Graph`` with ``labels`` and ``metadata = {"homophily", "seed"}``.
    """
    used = resolve_seed(seed)
    edges, labels, homophily = _ext.generate_ba(n_nodes, m, m0, n_communities, used)
    return trusted_graph(
        edges,
        n_nodes,
        labels=labels,
        metadata=_metadata(homophily, used),
    )


def generate_csbm(  # noqa: PLR0913 - the cSBM keyword surface, one argument per knob
    n_nodes: int,
    n_communities: int,
    homophily: float,
    avg_degree: float,
    feat_dim: int,
    mu: float,
    *,
    feature_dist: str = "gaussian",
    class_weights: Sequence[float] | None = None,
    theta_exponent: float | None = None,
    p_triangle: float = 0.0,
    feat_noise_ratio: float = 0.0,
    ensure_connected: bool = False,
    p_in_override: float | None = None,
    p_out_override: float | None = None,
    b_matrix: Sequence[float] | None = None,
    centroids: Sequence[float] | None = None,
    covariance: Sequence[float] | None = None,
    seed: int | None = None,
) -> Graph:
    """Contextual SBM: block structure plus correlated ``feat_dim``-dimensional features.

    Class centroids are scaled by ``mu``; ``feature_dist`` is ``"gaussian"``,
    ``"uniform"`` or ``"laplacian"``. Returns an undirected ``Graph`` with
    ``labels``, ``x`` (float32 ``(n_nodes, feat_dim)``) and
    ``metadata = {"homophily", "seed"}``.
    """
    used = resolve_seed(seed)
    edges, x, labels, effective_h = _ext.generate_csbm(
        n_nodes,
        n_communities,
        homophily,
        avg_degree,
        feat_dim,
        mu,
        feature_dist,
        None if class_weights is None else list(class_weights),
        theta_exponent,
        p_triangle,
        feat_noise_ratio,
        ensure_connected,
        p_in_override,
        p_out_override,
        None if b_matrix is None else list(b_matrix),
        None if centroids is None else list(centroids),
        None if covariance is None else list(covariance),
        used,
    )
    return trusted_graph(
        edges,
        n_nodes,
        labels=labels,
        x=x,
        metadata=_metadata(effective_h, used),
    )
