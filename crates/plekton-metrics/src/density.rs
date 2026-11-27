// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Graph density computation.

/// Compute the density of an undirected simple graph.
///
/// Density is defined as the ratio of existing edges to the maximum possible
/// number of edges in a simple undirected graph with `n_nodes` nodes:
///
/// ```text
/// density = 2E / (N * (N - 1))
/// ```
///
/// # Edge cases
/// - Returns `None` for graphs with fewer than 2 nodes: no pair of nodes can be
///   an edge, so there is nothing for the edge count to be a fraction of. Pair
///   the `None` with a [`crate::skipped::SkippedMetric`].
/// - Returns `Some(0.0)` for an edgeless graph of 2 nodes or more, a real answer.
/// - Returns `Some(1.0)` for a complete graph `Kₙ` (`n_edges = n_nodes*(n_nodes-1)/2`).
///
/// # Arguments
/// * `n_nodes` – Total number of nodes.
/// * `n_edges` – Number of canonical undirected edges.
///
/// # Examples
/// ```
/// use plekton_metrics::density::density;
///
/// // Complete graph K4: 6 edges, density = 2*6 / (4*3) = 1.0
/// assert_eq!(density(4, 6), Some(1.0));
///
/// // Empty graph: 0 edges → density 0.0
/// assert_eq!(density(100, 0), Some(0.0));
///
/// // Single node: no pair to be an edge → no density
/// assert_eq!(density(1, 0), None);
/// ```
#[must_use]
pub const fn density(n_nodes: usize, n_edges: usize) -> Option<f64> {
    if n_nodes < 2 {
        return None;
    }
    let max_edges = crate::count_to_f64(n_nodes) * (crate::count_to_f64(n_nodes) - 1.0) / 2.0;
    Some(crate::count_to_f64(n_edges) / max_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(value: Option<f64>) -> Option<u64> {
        value.map(f64::to_bits)
    }

    #[test]
    fn complete_graph_k4() {
        // K4 has 4*3/2 = 6 edges → density 1.0
        assert_eq!(bits(density(4, 6)), Some(1.0_f64.to_bits()));
    }

    #[test]
    fn empty_graph() {
        assert_eq!(bits(density(100, 0)), Some(0.0_f64.to_bits()));
    }

    #[test]
    fn single_node_has_no_density() {
        assert_eq!(density(1, 0), None);
    }

    #[test]
    fn zero_nodes_have_no_density() {
        assert_eq!(density(0, 0), None);
    }

    #[test]
    fn sparse_graph() {
        // N=4, E=1 → density = 1/6 ≈ 0.1667
        assert!(density(4, 1).is_some_and(|d| (d - 1.0 / 6.0).abs() < 1e-12));
    }

    #[test]
    fn path_graph_10() {
        // Path on 10 nodes: 9 edges, max = 45 → density = 0.2
        assert!(density(10, 9).is_some_and(|d| (d - 0.2).abs() < 1e-12));
    }
}
