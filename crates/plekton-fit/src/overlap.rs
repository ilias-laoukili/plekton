// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Overlapping communities by speaker-listener label propagation (SLPA).
//!
//! In a social graph a person belongs to several groups at once: a family, a
//! workshop, two circles of friends. [`crate::partition::detect_label_propagation`]
//! gives each node one label. SLPA (Xie, Szymanski & Liu, 2011) gives each node
//! a memory of the labels it heard, and keeps every label heard often enough.
//!
//! Each round, every node (in an order drawn from the seed) listens: each
//! neighbour speaks one label drawn from its memory in proportion to how often
//! it holds it, and the listener stores the label spoken most (ties drawn from
//! the seed). After `rounds` rounds, a node belongs to every community
//! whose label makes up at least `threshold` of its memory. Communities of one
//! node, duplicates and communities nested in a larger one are dropped.
//!
//! Determinism: memories are ordered maps and the visiting order comes from the
//! seed, so the same seed and the same edges give the same communities.

use std::collections::BTreeMap;

use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};

/// Overlapping communities of a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overlapping {
    /// Each community as its sorted node ids; sorted by their first node.
    pub communities: Vec<Vec<usize>>,
    /// For every node, the indices of the communities it belongs to (possibly
    /// several, possibly none for an isolated node).
    pub memberships: Vec<Vec<usize>>,
}

/// Detects overlapping communities with SLPA.
///
/// `threshold` is the share of a node's memory a label needs for the node to
/// belong to that community, in `(0, 1]`: `0.5` allows at most one community
/// per node, `0.1` up to ten. It is clamped into that range.
#[must_use]
pub fn detect_overlapping(
    edges: &[(usize, usize)],
    n_nodes: usize,
    seed: u64,
    rounds: usize,
    threshold: f64,
) -> Overlapping {
    let threshold = threshold.clamp(f64::MIN_POSITIVE, 1.0);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(source, target) in edges {
        if source < n_nodes && target < n_nodes && source != target {
            adjacency[source].push(target);
            adjacency[target].push(source);
        }
    }

    // Every node starts by holding its own label once.
    let mut memory: Vec<BTreeMap<usize, usize>> = (0..n_nodes)
        .map(|node| BTreeMap::from([(node, 1)]))
        .collect();
    let mut held: Vec<usize> = vec![1; n_nodes];
    let mut order: Vec<usize> = (0..n_nodes).collect();
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..rounds {
        order.shuffle(&mut rng);
        for &listener in &order {
            if adjacency[listener].is_empty() {
                continue;
            }
            let mut heard: BTreeMap<usize, usize> = BTreeMap::new();
            for &speaker in &adjacency[listener] {
                let label = speak(&memory[speaker], held[speaker], &mut rng);
                *heard.entry(label).or_insert(0) += 1;
            }
            // Most spoken label; a tie is broken by a draw from the seed, so a
            // node heard equally by two groups remembers both.
            let top = heard.values().copied().max().unwrap_or(0);
            let tied: Vec<usize> = heard
                .iter()
                .filter(|&(_, &count)| count == top)
                .map(|(&label, _)| label)
                .collect();
            if let Some(&label) = tied.choose(&mut rng) {
                *memory[listener].entry(label).or_insert(0) += 1;
                held[listener] += 1;
            }
        }
    }

    // A node joins every label that is frequent enough in its memory.
    let mut by_label: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (node, labels) in memory.iter().enumerate() {
        for (&label, &count) in labels {
            if share(count, held[node]) >= threshold {
                by_label.entry(label).or_default().push(node);
            }
        }
    }
    let mut communities: Vec<Vec<usize>> = by_label
        .into_values()
        .filter(|members| members.len() > 1)
        .collect();
    communities.sort();
    communities.dedup();
    let nested: Vec<bool> = communities
        .iter()
        .enumerate()
        .map(|(i, c)| {
            communities
                .iter()
                .enumerate()
                .any(|(j, other)| i != j && other.len() > c.len() && is_subset(c, other))
        })
        .collect();
    let communities: Vec<Vec<usize>> = communities
        .into_iter()
        .zip(nested)
        .filter_map(|(c, inside)| (!inside).then_some(c))
        .collect();

    let mut memberships = vec![Vec::new(); n_nodes];
    for (index, members) in communities.iter().enumerate() {
        for &node in members {
            memberships[node].push(index);
        }
    }
    Overlapping {
        communities,
        memberships,
    }
}

/// One label drawn from a memory, in proportion to how often it is held.
fn speak(memory: &BTreeMap<usize, usize>, held: usize, rng: &mut SmallRng) -> usize {
    let mut pick = rng.gen_range(0..held.max(1));
    for (&label, &count) in memory {
        if pick < count {
            return label;
        }
        pick -= count;
    }
    memory.keys().next().copied().unwrap_or(0)
}

/// `small` is sorted and so is `large`.
fn is_subset(small: &[usize], large: &[usize]) -> bool {
    small.iter().all(|x| large.binary_search(x).is_ok())
}

fn share(count: usize, total: usize) -> f64 {
    let to = |n: usize| f64::from(u32::try_from(n).unwrap_or(u32::MAX));
    to(count) / to(total.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All pairs of `nodes`.
    fn clique(nodes: &[usize]) -> Vec<(usize, usize)> {
        nodes
            .iter()
            .enumerate()
            .flat_map(|(i, &a)| nodes[i + 1..].iter().map(move |&b| (a, b)))
            .collect()
    }

    #[test]
    fn a_bridge_belongs_to_both_groups() {
        // Two circles of eight; node 16 has four friends in each.
        let mut edges = clique(&[0, 1, 2, 3, 4, 5, 6, 7]);
        edges.extend(clique(&[8, 9, 10, 11, 12, 13, 14, 15]));
        edges.extend([0, 1, 2, 3, 8, 9, 10, 11].iter().map(|&v| (v, 16)));
        for seed in 0..20 {
            let found = detect_overlapping(&edges, 17, seed, 100, 0.2);
            assert_eq!(found.memberships[16].len(), 2, "seed {seed}: {found:?}");
            assert_eq!(found.memberships[7].len(), 1, "seed {seed}");
            assert_ne!(found.memberships[7], found.memberships[15], "seed {seed}");
        }
    }

    #[test]
    fn apart_groups_stay_apart_and_isolated_nodes_have_none() {
        let mut edges = clique(&[0, 1, 2, 3]);
        edges.extend(clique(&[4, 5, 6, 7]));
        let found = detect_overlapping(&edges, 9, 3, 30, 0.3);
        assert_eq!(found.communities, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
        assert_eq!(found.memberships[8], [] as [usize; 0]);
    }

    #[test]
    fn same_seed_same_communities() {
        let mut edges = clique(&[0, 1, 2, 3, 4]);
        edges.extend([(4, 5), (5, 6), (6, 7), (7, 5)]);
        assert_eq!(
            detect_overlapping(&edges, 8, 11, 30, 0.2),
            detect_overlapping(&edges, 8, 11, 30, 0.2)
        );
    }
}
