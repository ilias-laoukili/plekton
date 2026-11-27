// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Degree-aware selection with a per-node minimum-edge guarantee.

use super::indices_by_score_desc;

/// Select edges keeping each node's top `min_per_node` incident edges, then
/// filling the remaining budget from the global score ranking.
///
/// Phase 1 guarantees every node with degree `≥ 1` keeps at least
/// `min(min_per_node, degree)` of its highest-scoring incident edges — so no
/// node that had an edge is left without one. Phase 2 tops the selection up to
/// `num_keep` from the global descending ranking. If the guarantee phase alone
/// already meets or exceeds `num_keep`, all guaranteed edges are kept (the
/// result may then hold slightly more than `num_keep` edges — leaving no node
/// isolated takes precedence over the budget). Like [`crate::Guard::Isolates`],
/// this does not keep the graph connected.
pub fn select(
    n_nodes: usize,
    edges: &[(usize, usize)],
    scores: &[f64],
    num_keep: usize,
    min_per_node: usize,
) -> Vec<bool> {
    let n_edges = edges.len();
    let mut mask = vec![false; n_edges];

    // Incident edge indices per node (both endpoints; self-loops counted once).
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (edge_index, &(u, v)) in edges.iter().enumerate() {
        incident[u].push(edge_index);
        if v != u {
            incident[v].push(edge_index);
        }
    }

    // Phase 1 — guarantee the top `min_per_node` incident edges of each node.
    for edges_of_node in &incident {
        if edges_of_node.is_empty() {
            continue;
        }
        let keep = min_per_node.min(edges_of_node.len());
        let mut ranked = edges_of_node.clone();
        ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        for &edge_index in &ranked[..keep] {
            mask[edge_index] = true;
        }
    }

    // Phase 2 — fill the remaining budget from the global ranking.
    let mut kept = mask.iter().filter(|&&m| m).count();
    if kept < num_keep {
        for edge_index in indices_by_score_desc(scores) {
            if kept >= num_keep {
                break;
            }
            if !mask[edge_index] {
                mask[edge_index] = true;
                kept += 1;
            }
        }
    }

    mask
}
