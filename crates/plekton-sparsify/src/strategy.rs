// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Sparsification strategy, guard and edge-scoring metric parameter types.

/// The edge-selection policy applied by [`crate::sparsify`].
///
/// Every variant carries an explicit `retention_ratio` in `(0, 1]` — the target
/// fraction of edges to keep. The stochastic [`Strategy::Sampled`] variant also
/// carries an explicit `seed` for full reproducibility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strategy {
    /// Keep the highest-scoring `retention_ratio` fraction of edges (a hard
    /// top-k cut). When `keep_lowest` is `true`, keep the lowest-scoring edges
    /// instead — useful as a control experiment.
    ///
    /// This is the **Jaccard-Threshold / AA-Threshold** family. Note it is a
    /// distinct (and, on homophilous graphs, weaker) method than the metric
    /// backbone: a raw top-k on similarity scores, not a shortest-path filter.
    Threshold {
        /// Target fraction of edges to keep, in `(0, 1]`.
        retention_ratio: f64,
        /// Keep the lowest-scoring edges instead of the highest.
        keep_lowest: bool,
    },

    /// Global top-k selection with a per-node minimum-degree guarantee.
    ///
    /// Each node first keeps its top `min_per_node` incident edges by score;
    /// the remaining budget is filled from the global ranking. This prevents
    /// low-degree nodes from being isolated at aggressive retention ratios.
    DegreeAware {
        /// Target fraction of edges to keep, in `(0, 1]`.
        retention_ratio: f64,
        /// Minimum incident edges guaranteed per node (before the global fill).
        min_per_node: usize,
    },

    /// Weighted sampling **without replacement** (Efraimidis-Spirakis keys): like
    /// successive draws that each pick a remaining edge with probability
    /// proportional to its score. An edge's chance of being kept grows with its
    /// score but is not proportional to it; it saturates towards 1 for heavy
    /// edges.
    ///
    /// Unlike [`Strategy::Threshold`], every edge has a non-zero chance of being
    /// kept, so low-scoring bridge edges are not guaranteed to be cut. The
    /// `seed` drives stream 1 of `ChaCha8Rng::seed_from_u64(seed)` — not stream
    /// 0, which the effective-resistance projection reads — so the same seed
    /// reproduces the same selection (not promised across minor versions of
    /// `rand`).
    Sampled {
        /// Target fraction of edges to keep, in `(0, 1]`.
        retention_ratio: f64,
        /// RNG seed for reproducible sampling.
        seed: u64,
    },
}

impl Strategy {
    /// The `retention_ratio` carried by this strategy.
    #[must_use]
    pub const fn retention_ratio(&self) -> f64 {
        match *self {
            Self::Threshold {
                retention_ratio, ..
            }
            | Self::DegreeAware {
                retention_ratio, ..
            }
            | Self::Sampled {
                retention_ratio, ..
            } => retention_ratio,
        }
    }
}

/// What [`crate::sparsify`] repairs once the [`Strategy`] has chosen its edges.
///
/// Non-exhaustive so that further guards can be added without breaking callers
/// that match on it.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guard {
    /// Keep the selection exactly as the strategy made it.
    Off,
    /// No node loses its last edge: a node that had at least one edge in the
    /// input but none in the selection gets its highest-scoring incident edge
    /// back, as GNNs that divide by node degree need.
    ///
    /// This is **not** a connectivity guarantee. Every node keeps an edge, yet
    /// the kept edges can still split a connected graph into many components: a
    /// 40-node path whose threshold keeps every other edge comes out as 20
    /// disjoint pairs.
    ///
    /// The restored edges come on top of the strategy's target, so the result
    /// can hold more edges than `retention_ratio` asks for — 20 instead of
    /// ⌊39 · 0.5⌋ = 19 on that path. Compare methods at an equal edge budget
    /// with [`Guard::Off`], or count the edges actually kept.
    Isolates,
}

/// A structural (or feature) edge-scoring metric, dispatched to
/// [`plekton_metrics::similarity`].
///
/// This is a convenience so callers can request scoring and selection in one
/// [`crate::sparsify_with_metric`] call. Callers that already hold scores can
/// use [`crate::sparsify`] directly and skip this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Jaccard neighbourhood overlap.
    Jaccard,
    /// Adamic-Adar hub-penalised overlap.
    AdamicAdar,
    /// Degree product `deg(u) · deg(v)`.
    DegreeProduct,
    /// Feature cosine similarity (requires node features).
    FeatureCosine,
}
