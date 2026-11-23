// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Sparsification strategy and edge-scoring metric parameter types.

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

    /// Weighted sampling **without replacement**, with selection probability
    /// proportional to edge score (Efraimidis-Spirakis keys).
    ///
    /// Unlike [`Strategy::Threshold`], every edge has a non-zero chance of being
    /// kept, so low-scoring bridge edges are not guaranteed to be cut. The
    /// `seed` drives a `ChaCha8Rng` for cross-platform, cross-version
    /// reproducibility (bit-identical on a laptop and in CI).
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
