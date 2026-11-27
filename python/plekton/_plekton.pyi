# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Type stubs for the compiled Rust extension ``plekton._plekton``.

Low-level surface: every argument is explicit (seeds included) and edges are
``int64[E, 2]`` arrays. The public API in ``plekton`` wraps these into
:class:`plekton.Graph`.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ─── Synthetic graph generators ──────────────────────────────────────────────

def generate_sbm(
    n_nodes: int,
    n_communities: int,
    p_in: float,
    p_out: float,
    theta_exponent: float | None,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    """Generate an SBM or exact O(N + E) DC-SBM graph.

    Returns ``(edges [E, 2], labels [N], effective_h)``.
    """

def generate_sbm_from_fit(
    n_nodes: int,
    block_sizes: Sequence[int],
    b_matrix: NDArray[np.float64],
    theta: NDArray[np.float64] | None,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    """Generate a graph from an arbitrary fitted blockmodel ``(block_sizes, b_matrix, theta)``.

    Returns ``(edges [E, 2], labels [N], effective_h)``.
    """

def generate_ba(
    n_nodes: int,
    m: int,
    m0: int | None,
    n_communities: int,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], float]:
    """Generate a Barabási–Albert preferential-attachment graph.

    Returns ``(edges [E, 2], labels [N], effective_h)``.
    """

def generate_csbm(
    n_nodes: int,
    n_communities: int,
    homophily: float,
    avg_degree: float,
    feat_dim: int,
    mu: float,
    feature_dist: str,
    class_weights: Sequence[float] | None,
    theta_exponent: float | None,
    p_triangle: float,
    feat_noise_ratio: float,
    ensure_connected: bool,
    p_in_override: float | None,
    p_out_override: float | None,
    b_matrix: Sequence[float] | None,
    centroids: Sequence[float] | None,
    covariance: Sequence[float] | None,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.int64], float]:
    """Generate a contextual SBM graph with correlated node features.

    Returns ``(edges [E, 2], x [N, F], labels [N], effective_h)``.
    """

# ─── JSON serialization & config dispatch ────────────────────────────────────

def graph_to_json(
    edge_index: NDArray[np.int64],
    labels: NDArray[np.int64],
    x: NDArray[np.float32] | None = None,
    path: str | None = None,
) -> str:
    """Serialise ``(edge_index [2, M], labels, optional x)`` to canonical ``plekton.graph/v1`` JSON."""

def json_to_graph(
    json_str: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32] | None, str]:
    """Parse ``plekton.graph/v1`` JSON into ``(edges [E, 2], labels [N], x?, metadata JSON)``."""

def generate_from_config(config_json: str) -> str:
    """Generate a graph from a JSON configuration contract and return ``plekton.graph/v1`` JSON."""

# ─── Multi-attribute mixing graphs ───────────────────────────────────────────

def mixing_graph_native(
    nodes: int,
    attributes: dict[str, list[str]] | None,
    groups: list[str | None] | None,
    degrees: list[int] | None,
    mean_degree: float | None,
    degree_spread: float,
    max_degree: int | None,
    mixing: dict[str, dict[str, Any]] | None,
    clustering: float | None,
    strength: dict[str, Any] | None,
    directed: bool,
    popularity_spread: float,
    seed: int,
) -> dict[str, Any]:
    """Generate a multi-attribute mixing graph with zero-copy NumPy arrays."""

def generate_mixing_graph_json_py(request_json: str) -> str:
    """JSON-in / JSON-out entry point for ``mixing_graph``."""

# ─── Submodules ──────────────────────────────────────────────────────────────

class fit:  # noqa: N801 - mirrors PyO3 submodule name
    """DC-SBM blockmodel estimation and community detection bindings."""

    @staticmethod
    def fit_blockmodel(
        edges: NDArray[np.int64],
        n_nodes: int,
        labels: NDArray[np.int64] | None,
        attribute: str,
        seed: int,
        max_rounds: int,
    ) -> dict[str, Any]:
        """Estimate DC-SBM block sizes, K×K mixing matrix, and per-node degree weights θ."""

    @staticmethod
    def label_propagation(
        edges: NDArray[np.int64],
        n_nodes: int,
        seed: int,
        max_rounds: int,
    ) -> tuple[NDArray[np.int64], bool]:
        """Run asynchronous seeded label propagation; returns ``(labels [N], converged)``."""

    @staticmethod
    def overlapping_communities(
        edges: NDArray[np.int64],
        n_nodes: int,
        seed: int,
        rounds: int,
        threshold: float,
    ) -> list[list[int]]:
        """Detect overlapping communities via SLPA."""

class metrics:  # noqa: N801 - mirrors PyO3 submodule name
    """Structural graph metrics over ``[E, 2]`` canonical edge lists."""

    @staticmethod
    def homophily(edges: NDArray[np.int64], labels: NDArray[np.int64]) -> float | None:
        """Edge homophily (fraction of intra-community edges), or ``None`` if undefined."""

    @staticmethod
    def clustering(n_nodes: int, edges: NDArray[np.int64]) -> float | None:
        """Mean local clustering coefficient, or ``None`` above threshold / undefined."""

    @staticmethod
    def transitivity(n_nodes: int, edges: NDArray[np.int64]) -> float | None:
        """Global transitivity (3 * triangles / open_triplets), or ``None`` if undefined."""

    @staticmethod
    def assortativity(n_nodes: int, edges: NDArray[np.int64]) -> float | None:
        """Newman's degree assortativity r, or ``None`` if edgeless or regular."""

    @staticmethod
    def components(n_nodes: int, edges: NDArray[np.int64]) -> int:
        """Count connected components."""

    @staticmethod
    def density(n_nodes: int, n_edges: int) -> float:
        """Undirected graph density: ``2E / (N * (N - 1))``."""

    @staticmethod
    def attribute_mixing(
        edges: NDArray[np.int64],
        labels: NDArray[np.int64],
    ) -> dict[str, Any] | None:
        """Attribute mixing report ``{"within", "random", "r"}``, or ``None`` if edgeless."""

    @staticmethod
    def degree_summary(n_nodes: int, edges: NDArray[np.int64]) -> dict[str, Any] | None:
        """Degree summary ``{"mean", "max", "p50", "p90", "p99", "gini"}``."""

    @staticmethod
    def degree_ks(
        n_nodes_a: int,
        edges_a: NDArray[np.int64],
        n_nodes_b: int,
        edges_b: NDArray[np.int64],
    ) -> float | None:
        """Two-sample KS distance in ``[0, 1]`` between two graphs' degree distributions."""

class sparsify:  # noqa: N801 - mirrors PyO3 submodule name
    """Edge scoring, approximate effective resistance, and sparsification bindings."""

    @staticmethod
    def similarity_scores(
        edges: NDArray[np.int64],
        n_nodes: int,
        metric: str,
        features: NDArray[np.float32] | None,
    ) -> NDArray[np.float64]:
        """Compute per-edge similarity scores (``jaccard``, ``adamic_adar``, ``degree_product``, ``feature_cosine``)."""

    @staticmethod
    def approx_effective_resistance(
        edges: NDArray[np.int64],
        n_nodes: int,
        epsilon: float,
        seed: int,
        max_dim: int | None,
    ) -> NDArray[np.float64]:
        """Compute per-edge Spielman–Srivastava approximate effective resistance."""

    @staticmethod
    def select_mask(
        edges: NDArray[np.int64],
        n_nodes: int,
        scores: NDArray[np.float64],
        strategy: str,
        retention_ratio: float,
        keep_lowest: bool,
        min_per_node: int,
        seed: int,
        guard: str,
    ) -> NDArray[np.bool_]:
        """Select edges under ``threshold``, ``degree_aware``, or ``sampled``; guard ``none`` or ``isolates``."""

    @staticmethod
    def metric_backbone_mask(
        edges: NDArray[np.int64],
        n_nodes: int,
        distances: NDArray[np.float64],
        epsilon: float,
        distances_are_similarity: bool,
    ) -> NDArray[np.bool_]:
        """Keep shortest-path-preserving metric-backbone edges."""
