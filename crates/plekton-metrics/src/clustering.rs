// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Mean local clustering-coefficient computation.

use hashbrown::HashSet;
use rayon::prelude::*;

/// Maximum number of nodes for which the clustering-coefficient computation
/// is attempted.
///
/// The local clustering coefficient requires iterating over every node's
/// neighbour set and counting shared neighbours, which has cost
/// `O(E · max_degree)`. For graphs larger than this threshold the computation
/// is skipped and [`compute_clustering_coefficient`] returns `None`.
pub const CLUSTERING_THRESHOLD: usize = 10_000;

/// Compute the mean local clustering coefficient of the graph.
///
/// For a node `v` with degree `d_v ≥ 2`, the local clustering coefficient is:
///
/// ```text
/// c_v = |{ edges (u,w) : u,w ∈ N(v) }| / (d_v * (d_v - 1) / 2)
/// ```
///
/// where `N(v)` is the neighbour set of `v`. The mean is taken over all nodes;
/// nodes with degree 0 or 1 contribute `0.0` to the sum.
///
/// Returns `None` when `n_nodes > CLUSTERING_THRESHOLD` (the adjacency
/// intersection is `O(E · max_degree)` and becomes prohibitive for large graphs),
/// and `None` for a graph with no node, where the mean would be over nothing.
/// An edgeless graph with at least one node gives `Some(0.0)`: every node has
/// degree below 2 and contributes `0.0`, as above.
///
/// # Arguments
/// * `n_nodes` – Total number of nodes.
/// * `edges`   – Canonical edge list; each pair satisfies `source < target`.
///
/// # Examples
/// ```
/// use plekton_metrics::clustering::compute_clustering_coefficient;
///
/// // Complete triangle: every node sees both neighbours connected → coefficient 1.0
/// assert_eq!(
///     compute_clustering_coefficient(3, &[(0, 1), (0, 2), (1, 2)]),
///     Some(1.0),
/// );
///
/// // No node: no mean to take. Nodes without edges: each counts 0.
/// assert_eq!(compute_clustering_coefficient(0, &[]), None);
/// assert_eq!(compute_clustering_coefficient(5, &[]), Some(0.0));
/// ```
#[must_use]
pub fn compute_clustering_coefficient(n_nodes: usize, edges: &[(usize, usize)]) -> Option<f64> {
    if n_nodes > CLUSTERING_THRESHOLD || n_nodes == 0 {
        return None;
    }
    if edges.is_empty() {
        return Some(0.0);
    }

    // Build an adjacency list and per-node neighbour HashSets for O(1) lookup.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(source, target) in edges {
        adjacency[source].push(target);
        adjacency[target].push(source);
    }

    // Pre-build neighbour sets for fast intersection queries.
    let neighbour_sets: Vec<HashSet<usize>> = adjacency
        .iter()
        .map(|neighbours| neighbours.iter().copied().collect())
        .collect();

    // Each node's local coefficient is an independent O(degree²) adjacency
    // intersection — the dominant cost — so we compute them across rayon's
    // thread pool. We collect into a node-ordered Vec and sum sequentially
    // afterwards so the floating-point reduction is bit-identical to the
    // original sequential left-fold (parallel f64 sums reorder additions).
    let per_node_coefficients: Vec<f64> = (0..n_nodes)
        .into_par_iter()
        .map(|node_id| {
            let degree = adjacency[node_id].len();
            if degree < 2 {
                return 0.0;
            }
            let max_possible_triangle_pairs = (degree * (degree - 1)) / 2;
            // Count pairs of neighbours that are themselves connected.
            // Iterating only over (neighbour, later_neighbour) avoids double-counting.
            let triangle_pair_count: usize = adjacency[node_id]
                .iter()
                .enumerate()
                .map(|(neighbour_rank, &neighbour)| {
                    adjacency[node_id][neighbour_rank + 1..]
                        .iter()
                        .filter(|&&later_neighbour| {
                            neighbour_sets[neighbour].contains(&later_neighbour)
                        })
                        .count()
                })
                .sum();
            crate::count_to_f64(triangle_pair_count)
                / crate::count_to_f64(max_possible_triangle_pairs)
        })
        .collect();

    let total_coefficient: f64 = per_node_coefficients.iter().sum();

    Some(total_coefficient / crate::count_to_f64(n_nodes))
}
