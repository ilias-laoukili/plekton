// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Degree assortativity (Newman's r) computation.
//!
//! Degree assortativity measures the tendency of nodes to connect to others of
//! similar degree. It is the Pearson correlation coefficient of the degrees at
//! each end of every edge.
//!
//! # Reference
//! Newman, M. E. J. (2002). Assortative mixing in networks.
//! *Physical Review Letters*, 89(20), 208701.

/// Compute the degree assortativity coefficient of the graph.
///
/// For an undirected graph, assortativity is:
///
/// ```text
/// r = [|E|⁻¹ Σ_{(u,v)∈E} dᵤ·d_v − (|E|⁻¹ Σ_{(u,v)∈E} (dᵤ+d_v)/2)²]
///     ─────────────────────────────────────────────────────────────────────
///     [|E|⁻¹ Σ_{(u,v)∈E} (dᵤ²+d_v²)/2 − (|E|⁻¹ Σ_{(u,v)∈E} (dᵤ+d_v)/2)²]
/// ```
///
/// The result lies in `[-1, 1]`:
/// - **> 0** — high-degree nodes tend to connect to high-degree nodes
///   (assortative, e.g. social networks)
/// - **≈ 0** — no degree–degree correlation
/// - **< 0** — high-degree nodes tend to connect to low-degree nodes
///   (disassortative, e.g. star graphs, biological networks)
///
/// Returns `None` rather than `0.0` where the correlation does not exist, because
/// `0.0` is also a real answer — "no degree–degree correlation" — and a reader
/// given a bare `0.0` cannot tell it from a graph nobody measured:
///
/// | case | before | now |
/// |---|---|---|
/// | no nodes, or no edges | `0.0` | `None` |
/// | zero variance in the degree pairs (a regular graph) | `0.0` | `None` |
/// | variance exists, degrees uncorrelated | `0.0` | `Some(0.0)` |
///
/// Pair the `None` with a [`crate::skipped::SkippedMetric`] to say which case it was.
///
/// # Arguments
/// * `n_nodes` – Total number of nodes; must equal the length of the implied
///   degree sequence.
/// * `edges`   – Canonical edge list; each pair satisfies `source < target`.
///
/// # Examples
/// ```
/// use plekton_metrics::assortativity::degree_assortativity;
///
/// // Star K_{1,3}: hub (degree 3) only connects to leaves (degree 1) → r < 0
/// let star_edges = vec![(0, 1), (0, 2), (0, 3)];
/// assert!(degree_assortativity(4, &star_edges).is_some_and(|r| r < 0.0));
///
/// // Path of length 3 (0-1-2-3): mixed → r < 0 (endpoints are degree-1)
/// let path_edges = vec![(0, 1), (1, 2), (2, 3)];
/// assert!(degree_assortativity(4, &path_edges).is_some_and(|r| r < 0.0));
///
/// // A single edge: both ends have degree 1, so there is no variance to correlate
/// assert_eq!(degree_assortativity(2, &[(0, 1)]), None);
/// ```
// Newman (2002) r: `sum_half_degree²` legitimately appears in BOTH the numerator
#[must_use]
pub fn degree_assortativity(n_nodes: usize, edges: &[(usize, usize)]) -> Option<f64> {
    if n_nodes == 0 || edges.is_empty() {
        return None;
    }

    // Per-node degree — accumulated from the canonical undirected edge list.
    let mut degree = vec![0u64; n_nodes];
    #[allow(
        clippy::suboptimal_flops,
        reason = "exact IEEE rounding is preserved intentionally without fused mul_add"
    )]
    for &(source, target) in edges {
        degree[source] += 1;
        degree[target] += 1;
    }

    let edge_count = f64::from(u32::try_from(edges.len()).unwrap_or(u32::MAX));

    // Newman (2002) formula for undirected graphs:
    //   j_i, k_i               = degrees at each end of edge i
    //   sum_degree_product     = Σ_i j_i · k_i
    //   sum_half_degree        = Σ_i (j_i + k_i) / 2
    //   sum_half_degree_squared = Σ_i (j_i² + k_i²) / 2
    //
    // r = (m · sum_jk − sum_half²) / (m · sum_sq_half − sum_half²)
    //
    // The arithmetic is spelled out through named intermediates on purpose: the
    // published formula's exact IEEE rounding is preserved (no fused `mul_add`,
    // no `f64::midpoint`), so computed metric values stay reproducible.
    let (mut sum_degree_product, mut sum_half_degree, mut sum_half_degree_squared) =
        (0.0f64, 0.0f64, 0.0f64);
    #[allow(
        clippy::suboptimal_flops,
        reason = "exact IEEE rounding is preserved intentionally without fused mul_add"
    )]
    for &(source, target) in edges {
        let degree_source = f64::from(u32::try_from(degree[source]).unwrap_or(u32::MAX));
        let degree_target = f64::from(u32::try_from(degree[target]).unwrap_or(u32::MAX));
        sum_degree_product += degree_source * degree_target;
        let degree_sum = degree_source + degree_target;
        sum_half_degree += degree_sum / 2.0;
        let source_squared = degree_source * degree_source;
        let target_squared = degree_target * degree_target;
        let squared_sum = source_squared + target_squared;
        sum_half_degree_squared += squared_sum / 2.0;
    }

    let sum_half_squared = sum_half_degree * sum_half_degree;
    let numerator_product = edge_count * sum_degree_product;
    let numerator = numerator_product - sum_half_squared;
    let denominator_product = edge_count * sum_half_degree_squared;
    let denominator = denominator_product - sum_half_squared;

    if denominator.abs() < f64::EPSILON {
        // Every edge joins the same degree pair, so the Pearson denominator is
        // zero. That is not "no correlation"; it is a graph with nothing to
        // correlate.
        None
    } else {
        Some(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_graph_has_no_correlation_to_report() {
        assert_eq!(degree_assortativity(5, &[]), None);
    }

    #[test]
    fn star_is_disassortative() {
        // Star K_{1,4}: hub degree=4 connects only to leaves degree=1 → r < 0
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let assortativity_value = degree_assortativity(5, &edges);
        assert!(
            assortativity_value.is_some_and(|r| r < 0.0),
            "star should be disassortative, got r={assortativity_value:?}",
        );
    }

    #[test]
    fn a_regular_graph_has_no_variance_to_correlate() {
        // Cycle C4: every node has degree 2, so every edge joins the same degree
        // pair and the Pearson denominator is zero. That is not "r = 0"; it is a
        // graph the coefficient does not apply to, and the previous 0.0 was
        // indistinguishable from a real uncorrelated answer.
        let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
        assert_eq!(degree_assortativity(4, &edges), None);
    }

    #[test]
    fn complete_graph_k4_has_no_variance_either() {
        let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        assert_eq!(degree_assortativity(4, &edges), None);
    }

    #[test]
    fn path_is_disassortative() {
        // Path 0-1-2-3-4: endpoint nodes (deg 1) connect to interior (deg 2)
        // → disassortative
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        let assortativity_value = degree_assortativity(5, &edges);
        assert!(
            assortativity_value.is_some_and(|r| r < 0.0),
            "path should be disassortative, got r={assortativity_value:?}",
        );
    }

    #[test]
    fn result_in_range() {
        // Arbitrary graph: result should be in [-1, 1]
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)];
        let assortativity_value = degree_assortativity(5, &edges);
        assert!(
            assortativity_value.is_some_and(|r| (-1.0..=1.0).contains(&r)),
            "r={assortativity_value:?} outside [-1, 1]",
        );
    }
}
