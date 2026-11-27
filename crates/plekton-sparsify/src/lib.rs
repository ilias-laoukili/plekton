// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! # plekton-sparsify
//!
//! Principled edge-sparsification strategies for undirected graphs.
//!
//! Given an undirected graph and a per-edge importance score, this crate selects
//! a subset of edges under a target retention ratio. Scores come from
//! [`plekton_metrics::similarity`] (Jaccard, Adamic-Adar, feature cosine, degree
//! product); the selection strategy decides *which* scored edges to keep while
//! preserving the structural properties a GNN relies on.
//!
//! ## Strategies
//!
//! | [`Strategy`] | Selection rule |
//! |---|---|
//! | [`Strategy::Threshold`]   | hard top-k (or bottom-k) by score |
//! | [`Strategy::DegreeAware`] | top-k with a per-node minimum-degree guarantee |
//! | [`Strategy::Sampled`]     | weighted sampling without replacement (`ChaCha8Rng`) |
//!
//! Two further methods have their own modules because they compute their own
//! edge scores rather than taking a [`Strategy`] over supplied scores:
//!
//! - [`approx_er`] — approximate effective resistance (the Johnson–Lindenstrauss
//!   estimate of Spielman and Srivastava), combined with weighted sampling via
//!   [`sparsify_resistance_sampled`]. The sampled edges are not reweighted, so
//!   the result carries no spectral guarantee.
//! - [`backbone`] — the shortest-path-preserving metric backbone, via
//!   [`sparsify_metric_backbone`].
//!
//! ## Quick start
//!
//! ```
//! use plekton_sparsify::{sparsify_with_metric, Guard, Metric, Strategy};
//!
//! // A triangle plus one weak bridge edge (3)-(0).
//! let edges = [(0, 1), (0, 2), (1, 2), (0, 3)];
//! let strategy = Strategy::Threshold {
//!     retention_ratio: 0.75,
//!     keep_lowest: false,
//! };
//!
//! let result = sparsify_with_metric(
//!     4,
//!     &edges,
//!     None,
//!     Metric::Jaccard,
//!     strategy,
//!     /* ignore_self_loops = */ true,
//!     Guard::Isolates,
//! )
//! .expect("valid inputs");
//!
//! // Node 3 keeps its only edge: the guard gives it back.
//! assert!(result.kept_edges.contains(&(0, 3)));
//! ```

// Public API surface: every exported strategy, metric, and error variant is
// documented so integrators understand the selection semantics.
#![warn(missing_docs)]

pub mod approx_er;
pub mod backbone;
pub mod strategy;

mod ops;

pub use approx_er::{approx_er_scores, ApproxErConfig};
pub use backbone::{metric_backbone, proximity_to_distance};
pub use strategy::{Guard, Metric, Strategy};

/// The outcome of a sparsification call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsifyResult {
    /// The retained edges, a subset of the input in the same order.
    pub kept_edges: Vec<(usize, usize)>,
    /// Boolean keep-mask aligned with the input `edges` slice.
    pub kept_mask: Vec<bool>,
}

/// Errors returned by [`sparsify`] and [`sparsify_with_metric`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SparsifyError {
    /// `retention_ratio` was outside the half-open interval `(0, 1]`.
    #[error("retention_ratio must be in (0, 1], got {0}")]
    InvalidRatio(f64),
    /// The `scores` slice length did not match the number of edges.
    #[error("scores length ({scores}) does not match edge count ({edges})")]
    LengthMismatch {
        /// Number of scores supplied.
        scores: usize,
        /// Number of edges supplied.
        edges: usize,
    },
    /// [`Metric::FeatureCosine`] was requested without node features.
    #[error("feature_cosine requires node features")]
    MissingFeatures,
    /// A feature dimension of zero was supplied.
    #[error("feat_dim must be non-zero")]
    ZeroFeatureDim,
    /// The approximate-effective-resistance `epsilon` was not a finite number
    /// `> 0`.
    #[error("epsilon must be a finite number > 0, got {0}")]
    InvalidEpsilon(f64),
    /// The `n_nodes × k` Johnson-Lindenstrauss projection cannot be allocated.
    /// `k` grows as `1/ε²`, so a vanishing `epsilon` overflows it.
    #[error("the {n_nodes} x {k} JL projection cannot be allocated; raise epsilon or set max_dim")]
    ProjectionTooLarge {
        /// Number of nodes (rows of the projection).
        n_nodes: usize,
        /// JL dimension (columns of the projection).
        k: usize,
    },
}

/// `usize` count -> `f64`, exact for every count below 2^53 (it used to go
/// through `u32` and saturate at `u32::MAX`).
#[inline]
#[allow(
    clippy::cast_precision_loss,
    reason = "exact below 2^53, which no edge count gets near"
)]
const fn count_to_f64(n: usize) -> f64 {
    n as f64
}

/// Number of edges to keep for a retention ratio: `⌊n_edges · ratio⌋`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "ratio ∈ (0,1] so the floored product is ≤ n_edges and non-negative"
)]
fn num_to_keep(n_edges: usize, retention_ratio: f64) -> usize {
    (count_to_f64(n_edges) * retention_ratio).floor() as usize
}

/// Compute per-edge scores with a [`plekton_metrics::similarity`] metric.
///
/// `features` is `Some((flat_row_major, feat_dim))` for [`Metric::FeatureCosine`]
/// and may be `None` for the structural metrics. `ignore_self_loops` is normally
/// `true`, so a self-loop never makes a node its own neighbour (see
/// [`plekton_metrics::similarity`]).
///
/// # Errors
/// Returns [`SparsifyError::MissingFeatures`] / [`SparsifyError::ZeroFeatureDim`]
/// when [`Metric::FeatureCosine`] is requested without valid features.
pub fn score_edges(
    metric: Metric,
    n_nodes: usize,
    edges: &[(usize, usize)],
    features: Option<(&[f32], usize)>,
    ignore_self_loops: bool,
) -> Result<Vec<f64>, SparsifyError> {
    use plekton_metrics::similarity;
    let scores = match metric {
        Metric::Jaccard => similarity::jaccard_scores(n_nodes, edges, ignore_self_loops),
        Metric::AdamicAdar => similarity::adamic_adar_scores(n_nodes, edges, ignore_self_loops),
        Metric::DegreeProduct => {
            similarity::degree_product_scores(n_nodes, edges, ignore_self_loops)
        }
        Metric::FeatureCosine => {
            let (feature_rows, feat_dim) = features.ok_or(SparsifyError::MissingFeatures)?;
            if feat_dim == 0 {
                return Err(SparsifyError::ZeroFeatureDim);
            }
            similarity::feature_cosine_scores(edges, feature_rows, feat_dim)
        }
    };
    Ok(scores)
}

/// Select a subset of edges under the given [`Strategy`] and pre-computed
/// scores, then apply `guard`.
///
/// `scores` must be aligned with `edges` (one score per edge). With
/// [`Guard::Isolates`], every node that had at least one edge in the input keeps
/// at least one edge in the output (its highest-scoring incident edge is
/// restored if the selection dropped them all) — required for GNNs that divide
/// by node degree. The guard does not keep the graph connected; see [`Guard`].
///
/// # Errors
/// - [`SparsifyError::InvalidRatio`] if the strategy's `retention_ratio` is not
///   in `(0, 1]`.
/// - [`SparsifyError::LengthMismatch`] if `scores.len() != edges.len()`.
pub fn sparsify(
    n_nodes: usize,
    edges: &[(usize, usize)],
    scores: &[f64],
    strategy: &Strategy,
    guard: Guard,
) -> Result<SparsifyResult, SparsifyError> {
    let retention_ratio = strategy.retention_ratio();
    if retention_ratio <= 0.0 || retention_ratio > 1.0 {
        return Err(SparsifyError::InvalidRatio(retention_ratio));
    }
    if scores.len() != edges.len() {
        return Err(SparsifyError::LengthMismatch {
            scores: scores.len(),
            edges: edges.len(),
        });
    }

    let num_keep = num_to_keep(edges.len(), retention_ratio);

    let mut mask = match *strategy {
        Strategy::Threshold { keep_lowest, .. } => {
            ops::threshold::select(scores, num_keep, keep_lowest)
        }
        Strategy::DegreeAware { min_per_node, .. } => {
            ops::degree_aware::select(n_nodes, edges, scores, num_keep, min_per_node)
        }
        Strategy::Sampled { seed, .. } => ops::sampled::select(scores, num_keep, seed),
    };

    match guard {
        Guard::Off => {}
        Guard::Isolates => restore_isolates(n_nodes, edges, scores, &mut mask),
    }

    let kept_edges = edges
        .iter()
        .zip(&mask)
        .filter_map(|(&edge, &keep)| keep.then_some(edge))
        .collect();

    Ok(SparsifyResult {
        kept_edges,
        kept_mask: mask,
    })
}

/// Score edges with `metric` and then [`sparsify`] them in one call.
///
/// # Errors
/// Propagates errors from [`score_edges`] and [`sparsify`].
pub fn sparsify_with_metric(
    n_nodes: usize,
    edges: &[(usize, usize)],
    features: Option<(&[f32], usize)>,
    metric: Metric,
    strategy: Strategy,
    ignore_self_loops: bool,
    guard: Guard,
) -> Result<SparsifyResult, SparsifyError> {
    let scores = score_edges(metric, n_nodes, edges, features, ignore_self_loops)?;
    sparsify(n_nodes, edges, &scores, &strategy, guard)
}

/// Keep a `retention_ratio` share of the edges, sampled by approximate
/// effective resistance.
///
/// [`approx_er_scores`] scores the edges, then [`Strategy::Sampled`] keeps
/// `⌊retention_ratio · E⌋` of them by weighted sampling without replacement, so
/// high-resistance (bridge-like) edges are likely to stay.
///
/// This is **not** spectral sparsification. Spielman and Srivastava draw `q`
/// edges independently with replacement, edge `e` with probability `p_e`
/// proportional to `w_e · R_e`, and give each draw the weight `w_e / (q · p_e)`;
/// that reweighting is what preserves the Laplacian quadratic form. Here the
/// kept edges keep their unit weight, so no spectral bound applies.
///
/// `config.seed` drives both steps without sharing draws: the projection reads
/// stream 0 of `ChaCha8Rng::seed_from_u64(seed)` and the sampling stream 1.
///
/// # Errors
/// Propagates errors from [`approx_er_scores`] (e.g. an invalid `epsilon`) and
/// from [`sparsify`] (e.g. an out-of-range `retention_ratio`).
pub fn sparsify_resistance_sampled(
    n_nodes: usize,
    edges: &[(usize, usize)],
    retention_ratio: f64,
    config: &ApproxErConfig,
    guard: Guard,
) -> Result<SparsifyResult, SparsifyError> {
    let scores = approx_er_scores(n_nodes, edges, config)?;
    let strategy = Strategy::Sampled {
        retention_ratio,
        seed: config.seed,
    };
    sparsify(n_nodes, edges, &scores, &strategy, guard)
}

/// Metric-backbone sparsification driven by a [`plekton_metrics::similarity`]
/// metric: score edges, invert the (similarity) scores to distances, then keep
/// only shortest-path-preserving edges.
///
/// The retained fraction is determined by graph structure, not a ratio (see
/// [`backbone::metric_backbone`]). All [`Metric`] variants are similarities, so
/// the [`proximity_to_distance`] inversion uses the similarity form.
///
/// # Errors
/// Propagates errors from [`score_edges`] and [`backbone::metric_backbone`].
pub fn sparsify_metric_backbone(
    n_nodes: usize,
    edges: &[(usize, usize)],
    features: Option<(&[f32], usize)>,
    metric: Metric,
    epsilon: f64,
    ignore_self_loops: bool,
) -> Result<SparsifyResult, SparsifyError> {
    let scores = score_edges(metric, n_nodes, edges, features, ignore_self_loops)?;
    let distances = proximity_to_distance(&scores, false);
    metric_backbone(n_nodes, edges, &distances, epsilon)
}

/// [`Guard::Isolates`]: restore one incident edge for every node that the
/// selection would isolate.
///
/// A node that had `≥ 1` edge in the input but `0` in the current mask has its
/// highest-scoring incident edge re-added. Adding an edge covers both of its
/// endpoints, so nodes are processed once and the running kept-degree is
/// updated as edges are restored.
fn restore_isolates(n_nodes: usize, edges: &[(usize, usize)], scores: &[f64], mask: &mut [bool]) {
    let mut had_edge = vec![false; n_nodes];
    let mut kept_degree = vec![0usize; n_nodes];
    let mut best_incident: Vec<Option<(f64, usize)>> = vec![None; n_nodes];

    let mut note = |node: usize, edge_index: usize, kept: bool| {
        had_edge[node] = true;
        if kept {
            kept_degree[node] += 1;
        }
        let candidate = (scores[edge_index], edge_index);
        best_incident[node] = match best_incident[node] {
            Some(current) if current.0 >= candidate.0 => Some(current),
            _ => Some(candidate),
        };
    };

    for (edge_index, &(u, v)) in edges.iter().enumerate() {
        note(u, edge_index, mask[edge_index]);
        if v != u {
            note(v, edge_index, mask[edge_index]);
        }
    }

    for node in 0..n_nodes {
        if had_edge[node] && kept_degree[node] == 0 {
            if let Some((_, edge_index)) = best_incident[node] {
                if !mask[edge_index] {
                    mask[edge_index] = true;
                    let (u, v) = edges[edge_index];
                    kept_degree[u] += 1;
                    if v != u {
                        kept_degree[v] += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "unwrap keeps test assertions terse")]
mod tests {
    use super::*;

    // Triangle (0,1,2) with a weak pendant bridge (0,3): node 3 has degree 1.
    const EDGES: [(usize, usize); 4] = [(0, 1), (0, 2), (1, 2), (0, 3)];

    fn jaccard(edges: &[(usize, usize)], n: usize) -> Vec<f64> {
        plekton_metrics::similarity::jaccard_scores(n, edges, true)
    }

    #[test]
    fn threshold_keeps_highest_scoring() {
        // Jaccard: triangle edges share a neighbour (>0); bridge (0,3) scores 0.
        let scores = jaccard(&EDGES, 4);
        let strategy = Strategy::Threshold {
            retention_ratio: 0.75,
            keep_lowest: false,
        };
        // Without the guard, the 3 triangle edges are kept.
        let result = sparsify(4, &EDGES, &scores, &strategy, Guard::Off).unwrap();
        assert_eq!(result.kept_edges, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn invalid_ratio_is_rejected() {
        let scores = jaccard(&EDGES, 4);
        for bad in [0.0, -0.1, 1.5] {
            let strategy = Strategy::Threshold {
                retention_ratio: bad,
                keep_lowest: false,
            };
            assert_eq!(
                sparsify(4, &EDGES, &scores, &strategy, Guard::Off),
                Err(SparsifyError::InvalidRatio(bad)),
            );
        }
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let strategy = Strategy::Threshold {
            retention_ratio: 0.5,
            keep_lowest: false,
        };
        let err = sparsify(4, &EDGES, &[0.0, 0.0], &strategy, Guard::Off).unwrap_err();
        assert_eq!(
            err,
            SparsifyError::LengthMismatch {
                scores: 2,
                edges: 4
            }
        );
    }

    #[test]
    fn isolates_guard_saves_the_pendant_node() {
        // Aggressive threshold would drop the low-Jaccard bridge (0,3), isolating
        // node 3. The isolates guard must restore it.
        let scores = jaccard(&EDGES, 4);
        let strategy = Strategy::Threshold {
            retention_ratio: 0.5, // keep 2 edges
            keep_lowest: false,
        };
        let without = sparsify(4, &EDGES, &scores, &strategy, Guard::Off).unwrap();
        assert!(!without.kept_edges.contains(&(0, 3)));

        let with = sparsify(4, &EDGES, &scores, &strategy, Guard::Isolates).unwrap();
        assert!(with.kept_edges.contains(&(0, 3)));
        // Every node ends with at least one incident edge.
        let mut degree = [0usize; 4];
        for &(u, v) in &with.kept_edges {
            degree[u] += 1;
            degree[v] += 1;
        }
        assert!(degree.iter().all(|&d| d >= 1));
    }

    #[test]
    fn isolates_guard_does_not_keep_a_path_connected() {
        // A 40-node path whose scores alternate high and low: the threshold keeps
        // 19 of the 20 high edges (0,1), (2,3), …, (38,39) — ties go to the later
        // edges, so (0,1) is the one left out — and the guard gives (0,1) back to
        // the two nodes it would have stranded. No node is isolated, yet the path
        // falls into 20 disjoint pairs: the guard is not a connectivity guarantee.
        let path: Vec<(usize, usize)> = (0..39).map(|i| (i, i + 1)).collect();
        let scores: Vec<f64> = (0..39)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let strategy = Strategy::Threshold {
            retention_ratio: 0.5, // ⌊39 · 0.5⌋ = 19 edges
            keep_lowest: false,
        };
        let result = sparsify(40, &path, &scores, &strategy, Guard::Isolates).unwrap();

        let mut degree = [0usize; 40];
        for &(u, v) in &result.kept_edges {
            degree[u] += 1;
            degree[v] += 1;
        }
        assert!(degree.iter().all(|&d| d >= 1), "a node was left isolated");
        assert_eq!(
            plekton_metrics::components::count_components(40, &result.kept_edges),
            20
        );
        // The restored edge comes on top of the 19 the threshold aimed for.
        assert_eq!(result.kept_edges.len(), 20);

        // Without the guard, nodes 0 and 1 lose their only edges.
        let unguarded = sparsify(40, &path, &scores, &strategy, Guard::Off).unwrap();
        assert_eq!(unguarded.kept_edges.len(), 19);
        assert!(!unguarded.kept_edges.contains(&(0, 1)));
        assert!(unguarded.kept_edges.iter().all(|&(u, v)| u > 1 && v > 1));
    }

    #[test]
    fn degree_aware_guarantees_min_edge_per_node() {
        let scores = jaccard(&EDGES, 4);
        let strategy = Strategy::DegreeAware {
            retention_ratio: 0.25, // budget of 1, but every node must keep one
            min_per_node: 1,
        };
        let result = sparsify(4, &EDGES, &scores, &strategy, Guard::Off).unwrap();
        let mut degree = [0usize; 4];
        for &(u, v) in &result.kept_edges {
            degree[u] += 1;
            degree[v] += 1;
        }
        assert!(degree.iter().all(|&d| d >= 1));
    }

    #[test]
    fn sampled_is_deterministic_per_seed() {
        let scores = jaccard(&EDGES, 4);
        let strategy = Strategy::Sampled {
            retention_ratio: 0.5,
            seed: 7,
        };
        let a = sparsify(4, &EDGES, &scores, &strategy, Guard::Off).unwrap();
        let b = sparsify(4, &EDGES, &scores, &strategy, Guard::Off).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ratio_one_keeps_all_edges() {
        let scores = jaccard(&EDGES, 4);
        let strategy = Strategy::Threshold {
            retention_ratio: 1.0,
            keep_lowest: false,
        };
        let result = sparsify(4, &EDGES, &scores, &strategy, Guard::Off).unwrap();
        assert_eq!(result.kept_edges, EDGES.to_vec());
    }

    #[test]
    fn feature_cosine_missing_features_errors() {
        let err = score_edges(Metric::FeatureCosine, 4, &EDGES, None, true).unwrap_err();
        assert_eq!(err, SparsifyError::MissingFeatures);
    }
}
