// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Barabási–Albert preferential-attachment backbone generator.

use std::collections::HashSet;

use rand::Rng;

use crate::{
    sampling::{effective_homophily, seeded_rng},
    types::SbmCoreResult,
    SynthGraphError,
};

// Barabási–Albert core

/// Generates a Barabási–Albert preferential-attachment graph.
///
/// `m`: edges added per new node (≥ 1).
/// `m0`: initial clique size (≥ m, < `n_nodes`). A one-node seed (`m0 = 1`,
/// hence `m = 1`) has no edge to attach to, so node 1 joins node 0 and growth
/// continues from node 2: with `m = 1` the result is a tree on all `n_nodes`.
/// Community labels are assigned sequentially: `label[i] = i * n_communities / n_nodes`.
///
/// # Errors
/// Returns [`SynthGraphError`] if the parameters are invalid (e.g. `m0 < m`,
/// `m0 >= n_nodes`, or `m == 0`).
pub fn generate_ba_core(
    n_nodes: usize,
    m: usize,
    m0: usize,
    n_communities: usize,
    seed: Option<u64>,
) -> Result<SbmCoreResult, SynthGraphError> {
    if n_nodes == 0 {
        return Err(SynthGraphError::InvalidParameter(
            "n_nodes must be > 0".into(),
        ));
    }
    if m == 0 {
        return Err(SynthGraphError::InvalidParameter("m must be ≥ 1".into()));
    }
    if m0 < m {
        return Err(SynthGraphError::InvalidParameter(format!(
            "m0 ({m0}) must be ≥ m ({m})"
        )));
    }
    if m0 >= n_nodes {
        return Err(SynthGraphError::InvalidParameter(format!(
            "m0 ({m0}) must be < n_nodes ({n_nodes})"
        )));
    }
    if n_communities == 0 {
        return Err(SynthGraphError::InvalidParameter(
            "n_communities must be > 0".into(),
        ));
    }
    if n_communities > n_nodes {
        return Err(SynthGraphError::InvalidParameter(format!(
            "n_communities ({n_communities}) must be ≤ n_nodes ({n_nodes})"
        )));
    }

    let mut rng = seeded_rng(seed);

    let mut edges: Vec<(i64, i64)> = Vec::new();
    let mut degree: Vec<usize> = vec![0usize; n_nodes];

    // Seed clique on the first `m0` nodes — every pair is connected.
    for seed_node in 0..m0 {
        for other_seed_node in (seed_node + 1)..m0 {
            edges.push((
                i64::try_from(seed_node).unwrap_or(i64::MAX),
                i64::try_from(other_seed_node).unwrap_or(i64::MAX),
            ));
            degree[seed_node] += 1;
            degree[other_seed_node] += 1;
        }
    }

    // A one-node seed (m0 = 1, hence m = 1) is a clique without an edge, so it
    // contributes no stub and node 0 could never be drawn: it would stay
    // isolated. Node 1 has exactly one possible neighbour, so attach it to node 0
    // without an RNG draw and grow preferentially from node 2. (`m0 < n_nodes`
    // guarantees node 1 exists.) Every m0 ≥ 2 path is untouched.
    let first_grown_node = if m0 == 1 {
        edges.push((0, 1));
        degree[0] = 1;
        degree[1] = 1;
        2
    } else {
        m0
    };

    // Preferential-attachment "stubs" — each node appears `degree[node]` times.
    // Sampling a stub uniformly at random reproduces `P(node) ∝ deg(node)`.
    // The pool is never empty from here on: the seed has at least one edge.
    let mut stubs: Vec<usize> = (0..first_grown_node)
        .flat_map(|seed_node| std::iter::repeat_n(seed_node, degree[seed_node]))
        .collect();

    // Grow the graph one node at a time.
    for new_node in first_grown_node..n_nodes {
        let mut chosen_neighbours: Vec<usize> = Vec::with_capacity(m);
        let mut chosen_neighbour_set: HashSet<usize> = HashSet::with_capacity(m);
        // `m × 20` retries before giving up; matches `m=1` saturation budget
        // when the stub pool is small.
        let max_attempts = (m * 20).max(100);
        let mut attempts = 0usize;
        while chosen_neighbours.len() < m && attempts < max_attempts {
            attempts += 1;
            if stubs.is_empty() {
                break;
            }
            let candidate_node = stubs[rng.gen_range(0..stubs.len())];
            if candidate_node != new_node && chosen_neighbour_set.insert(candidate_node) {
                chosen_neighbours.push(candidate_node);
            }
        }

        for &existing_node in &chosen_neighbours {
            let (canonical_src, canonical_dst) = if existing_node < new_node {
                (existing_node, new_node)
            } else {
                (new_node, existing_node)
            };
            edges.push((
                i64::try_from(canonical_src).unwrap_or(i64::MAX),
                i64::try_from(canonical_dst).unwrap_or(i64::MAX),
            ));
            degree[existing_node] += 1;
            degree[new_node] += 1;
            stubs.push(existing_node);
            stubs.push(new_node);
        }
    }

    // Sequential community labels — block-of-N/K layout.
    let labels: Vec<usize> = (0..n_nodes)
        .map(|node_idx| node_idx * n_communities / n_nodes)
        .collect();

    let effective_h = effective_homophily(&edges, &labels);

    Ok(SbmCoreResult {
        canonical_edges: edges,
        labels,
        effective_h,
        n_nodes,
    })
}
