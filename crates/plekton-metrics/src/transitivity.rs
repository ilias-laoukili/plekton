// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Global transitivity (global clustering coefficient) computation.
//!
//! This is distinct from the *mean local clustering coefficient* in
//! [`crate::clustering`]: transitivity weights by node degree and counts
//! closed triangles globally, rather than averaging per-node coefficients.
//!
//! | Metric | Formula |
//! |---|---|
//! | Mean local CC | `(1/N) Σ_v c_v` where `c_v = triangles_v / possible_v` |
//! | Global transitivity | `3 × triangles / open_triplets` |

use hashbrown::HashSet;

/// Maximum number of nodes for which transitivity is computed.
///
/// Cost is `O(E · max_degree)`. Beyond this threshold the function returns `None`
/// (same convention as [`crate::clustering::CLUSTERING_THRESHOLD`]).
pub const TRANSITIVITY_THRESHOLD: usize = 20_000;

/// Compute the global transitivity (global clustering coefficient) of the graph.
///
/// ```text
/// T = 3 × |triangles| / |open_triplets|
/// ```
///
/// where:
/// - A **triangle** is a set of three mutually adjacent nodes.
/// - An **open triplet** is a path of length 2 (one node connected to two others);
///   equivalently `Σ_v d_v * (d_v - 1) / 2`.
///
/// Returns `None` rather than `0.0` for every case in which the quantity does not
/// exist, because `0.0` is also a real answer — a triangle-free graph with open
/// triplets genuinely has `T = 0`, and a reader given a bare `0.0` cannot tell it
/// from a graph nobody measured:
///
/// | case | before | now |
/// |---|---|---|
/// | no nodes, or no edges | `0.0` | `None` |
/// | `n_nodes > TRANSITIVITY_THRESHOLD` | `0.0` | `None` |
/// | no open triplets (no node of degree ≥ 2) | `0.0` | `None` |
/// | triplets exist, none closed | `0.0` | `Some(0.0)` |
///
/// Pair the `None` with a [`crate::skipped::SkippedMetric`] to say which case it was.
///
/// # Arguments
/// * `n_nodes` – Total number of nodes.
/// * `edges`   – Canonical edge list; each pair satisfies `source < target`.
///
/// # Examples
/// ```
/// use plekton_metrics::transitivity::transitivity;
///
/// // Perfect triangle: every open triplet is closed → T = 1.0
/// assert_eq!(transitivity(3, &[(0, 1), (0, 2), (1, 2)]), Some(1.0));
///
/// // Path 0–1–2: one open triplet, zero triangles → T = 0.0, which is an answer
/// assert_eq!(transitivity(3, &[(0, 1), (1, 2)]), Some(0.0));
///
/// // A single edge has no open triplet, so there is nothing to be a fraction of
/// assert_eq!(transitivity(2, &[(0, 1)]), None);
/// ```
#[must_use]
pub fn transitivity(n_nodes: usize, edges: &[(usize, usize)]) -> Option<f64> {
    if n_nodes == 0 || edges.is_empty() {
        return None;
    }
    if n_nodes > TRANSITIVITY_THRESHOLD {
        return None;
    }

    // Build adjacency list and per-node sets — sets enable O(min(deg_u, deg_v))
    // intersection per edge during triangle counting.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(source, target) in edges {
        adjacency[source].push(target);
        adjacency[target].push(source);
    }

    let adjacency_sets: Vec<HashSet<usize>> = adjacency
        .iter()
        .map(|neighbours| neighbours.iter().copied().collect())
        .collect();

    // For each canonical edge (u, v), |adj[u] ∩ adj[v]| counts shared
    // neighbours i.e. triangles through that edge. Each triangle is counted
    // 3× (once per edge), so divide by 3.
    let triangle_count: usize = edges
        .iter()
        .map(|&(source, target)| {
            adjacency_sets[source]
                .intersection(&adjacency_sets[target])
                .count()
        })
        .sum::<usize>()
        / 3;

    // Open triplets: `Σ_v deg(v) · (deg(v) - 1) / 2`.
    let open_triplet_count: usize = adjacency
        .iter()
        .map(|neighbours| {
            let degree = neighbours.len();
            degree.saturating_sub(1) * degree / 2
        })
        .sum();

    if open_triplet_count == 0 {
        // No node has two neighbours, so there is no denominator. That is not a
        // transitivity of zero; it is a graph the ratio does not apply to.
        None
    } else {
        Some(3.0 * crate::count_to_f64(triangle_count) / crate::count_to_f64(open_triplet_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_is_one() {
        // Single triangle: 1 triangle, 3 open triplets → T = 3/3 = 1.0
        assert_eq!(transitivity(3, &[(0, 1), (0, 2), (1, 2)]), Some(1.0));
    }

    #[test]
    fn a_path_is_a_real_zero() {
        // Path 0-1-2: 0 triangles, 1 open triplet. The denominator exists, so
        // this is an answer and not an absence — the one case that must stay
        // `Some(0.0)` for the distinction to be worth anything.
        assert_eq!(transitivity(3, &[(0, 1), (1, 2)]), Some(0.0));
    }

    #[test]
    fn an_empty_graph_has_nothing_to_report() {
        assert_eq!(transitivity(5, &[]), None);
    }

    #[test]
    fn isolated_nodes_have_nothing_to_report() {
        assert_eq!(transitivity(10, &[]), None);
    }

    #[test]
    fn a_graph_with_no_open_triplet_has_no_denominator() {
        // One edge: no node has two neighbours, so there is no triplet to close.
        assert_eq!(transitivity(2, &[(0, 1)]), None);
    }

    #[test]
    fn above_the_threshold_it_refuses_rather_than_answering_zero() {
        let n = TRANSITIVITY_THRESHOLD + 1;
        assert_eq!(transitivity(n, &[(0, 1), (0, 2), (1, 2)]), None);
    }

    #[test]
    fn complete_graph_k4() {
        // K4: 4 triangles, 12 open triplets → T = 12/12 = 1.0
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        assert_eq!(transitivity(4, &edges), Some(1.0));
    }

    #[test]
    fn two_triangles_sharing_edge() {
        // Nodes 0-1-2-3, edges: (0,1),(0,2),(1,2),(1,3),(2,3)
        // Triangles: {0,1,2} and {1,2,3} → 2 triangles
        // Open triplets: d0=2→1, d1=3→3, d2=3→3, d3=2→1 → 8
        // T = 6/8 = 0.75
        let edges = vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)];
        let transitivity_value = transitivity(4, &edges);
        assert!(
            transitivity_value.is_some_and(|t| (t - 0.75).abs() < 1e-12),
            "expected 0.75, got {transitivity_value:?}",
        );
    }
}
