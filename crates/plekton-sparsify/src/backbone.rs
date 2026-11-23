// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! The **Metric Backbone** — shortest-path-preserving sparsification.
//!
//! Unlike [`Strategy::Threshold`](crate::Strategy), which keeps a raw top-k of
//! similarity scores, the metric backbone keeps an edge `(u, v)` **iff its
//! direct distance is (within `ε` of) the shortest-path distance** between `u`
//! and `v`. Redundant edges — those short-circuited by a strictly shorter
//! multi-hop path — are dropped. The retained set exactly preserves all
//! geodesic distances, and its size is determined by the graph structure, not a
//! target ratio.
//!
//! This is the method the research paper calls the *metric backbone* (Serrano
//! et al., PNAS 2009), and it is **not** the same as thresholding raw Jaccard /
//! Adamic-Adar scores. Two steps are required:
//!
//! 1. **Proximity → distance.** Similarity scores must be inverted into a proper
//!    distance so the triangle inequality can prune edges
//!    ([`proximity_to_distance`]).
//! 2. **APSP filter.** Keep `(u, v)` iff `w(u,v) ≤ d_G(u, v) + ε`, where
//!    `d_G` is the shortest-path distance (Dijkstra from each source).
//!
//! Complexity is `O(n · (m log n))` time — fast on moderate-sized graphs and
//! subgraphs, heavier on very large graphs (where spectral sampling is faster).

use std::{cmp::Ordering, collections::BinaryHeap};

use rayon::prelude::*;

use crate::{SparsifyError, SparsifyResult};

/// Convert per-edge similarity (or distance) scores into distances in `[0, 1]`.
///
/// For a **similarity** metric (Jaccard, Adamic-Adar, feature cosine — higher
/// means closer), the distance is `d = 1 − s / s_max`. For a metric that is
/// **already a distance** (e.g. effective resistance — higher means farther),
/// the distance is `d = s / s_max`. When every score is `≤ 0`, all distances
/// are `1.0`. This mirrors the reference `_scores_to_cost` transform, whose
/// `1 − J` form is a proven proper metric for Jaccard.
#[must_use]
pub fn proximity_to_distance(scores: &[f64], is_distance_metric: bool) -> Vec<f64> {
    let s_max = scores.iter().copied().fold(0.0_f64, f64::max);
    if s_max <= 0.0 {
        return vec![1.0; scores.len()];
    }
    scores
        .iter()
        .map(|&s| {
            if is_distance_metric {
                s / s_max
            } else {
                1.0 - s / s_max
            }
        })
        .collect()
}

/// Compute the metric backbone of a weighted graph.
///
/// `distances` are per-edge weights aligned with `edges` (use
/// [`proximity_to_distance`] to derive them from similarity scores). An edge is
/// kept iff its weight is within `epsilon` of the shortest-path distance between
/// its endpoints. Self-loops are always dropped.
///
/// # Errors
/// Returns [`SparsifyError::LengthMismatch`] if `distances.len() != edges.len()`.
pub fn metric_backbone(
    n_nodes: usize,
    edges: &[(usize, usize)],
    distances: &[f64],
    epsilon: f64,
) -> Result<SparsifyResult, SparsifyError> {
    if distances.len() != edges.len() {
        return Err(SparsifyError::LengthMismatch {
            scores: distances.len(),
            edges: edges.len(),
        });
    }

    // Weighted adjacency for Dijkstra; and edge indices grouped by canonical
    // source so each source runs Dijkstra at most once.
    let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_nodes];
    let mut edges_of_source: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (edge_index, &(u, v)) in edges.iter().enumerate() {
        if u == v {
            continue;
        }
        let weight = distances[edge_index];
        adjacency[u].push((v, weight));
        adjacency[v].push((u, weight));
        edges_of_source[u].push(edge_index);
    }

    // For each source with outgoing edges, run Dijkstra once and test its edges.
    let decisions: Vec<(usize, bool)> = (0..n_nodes)
        .into_par_iter()
        .filter(|&source| !edges_of_source[source].is_empty())
        .flat_map_iter(|source| {
            let dist = dijkstra(&adjacency, source, n_nodes);
            edges_of_source[source].iter().map(move |&edge_index| {
                let (_, target) = edges[edge_index];
                let keep = distances[edge_index] <= dist[target] + epsilon;
                (edge_index, keep)
            })
        })
        .collect();

    let mut mask = vec![false; edges.len()];
    for (edge_index, keep) in decisions {
        mask[edge_index] = keep;
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

/// Single-source shortest paths over a non-negative weighted graph (Dijkstra
/// with a binary heap). Unreachable nodes retain `f64::INFINITY`.
fn dijkstra(adjacency: &[Vec<(usize, f64)>], source: usize, n_nodes: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; n_nodes];
    dist[source] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        node: source,
    });

    while let Some(HeapEntry {
        dist: current,
        node,
    }) = heap.pop()
    {
        if current > dist[node] {
            continue;
        }
        for &(neighbour, weight) in &adjacency[node] {
            let candidate = current + weight;
            if candidate < dist[neighbour] {
                dist[neighbour] = candidate;
                heap.push(HeapEntry {
                    dist: candidate,
                    node: neighbour,
                });
            }
        }
    }

    dist
}

/// Min-heap entry for Dijkstra (ordered so the smallest distance pops first).
#[derive(PartialEq)]
struct HeapEntry {
    dist: f64,
    node: usize,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse distance order (min-heap); node id breaks ties deterministically.
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "unwrap/float compares keep test assertions terse"
)]
mod tests {
    use super::*;

    #[test]
    fn drops_the_redundant_long_edge_of_a_triangle() {
        // Triangle with weights: (0,1)=1, (1,2)=1, (0,2)=3. The direct 0-2 edge
        // (3) is short-circuited by 0-1-2 (2), so it is redundant → dropped.
        let edges = [(0, 1), (0, 2), (1, 2)];
        let distances = [1.0, 3.0, 1.0];
        let result = metric_backbone(3, &edges, &distances, 1e-9).unwrap();
        assert_eq!(result.kept_edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn keeps_all_edges_when_none_are_redundant() {
        // Equal-weight triangle: no shorter alternative path exists.
        let edges = [(0, 1), (0, 2), (1, 2)];
        let distances = [1.0, 1.0, 1.0];
        let result = metric_backbone(3, &edges, &distances, 1e-9).unwrap();
        assert_eq!(result.kept_edges, edges.to_vec());
    }

    #[test]
    fn length_mismatch_errors() {
        let err = metric_backbone(3, &[(0, 1), (1, 2)], &[1.0], 0.0).unwrap_err();
        assert_eq!(
            err,
            SparsifyError::LengthMismatch {
                scores: 1,
                edges: 2
            }
        );
    }

    #[test]
    fn proximity_to_distance_inverts_similarity() {
        // s_max = 1.0 → distances are 1 − s.
        let d = proximity_to_distance(&[1.0, 0.0, 0.5], false);
        assert_eq!(d, vec![0.0, 1.0, 0.5]);
        // As a distance metric, distances are s / s_max.
        let d = proximity_to_distance(&[2.0, 0.0, 1.0], true);
        assert_eq!(d, vec![1.0, 0.0, 0.5]);
    }
}
