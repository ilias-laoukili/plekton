// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! The partition — the one input to a blockmodel fit that is a decision rather
//! than a measurement.
//!
//! Every number a blockmodel produces is conditional on which nodes were called
//! one block. Two partitions of the same graph give two different mixing
//! matrices, both correct, answering different questions. So the partition is
//! never defaulted here and never inferred from the absence of an argument: it
//! is a value the caller constructs, and it says where it came from.

use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

/// A detection algorithm and the parameters it actually has.
///
/// An enum rather than a bag of `Option`s so that no algorithm has to declare a
/// parameter it does not possess. Label propagation has no resolution knob;
/// recording `resolution: 1.0` for it would put a number in a passport that
/// means nothing, and a declaration that can only be satisfied by inventing a
/// value produces invented values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Detection {
    /// Asynchronous label propagation (Raghavan et al. 2007): every node adopts
    /// the label most common among its neighbours, in a seeded node order, until
    /// no node changes or the round budget is spent.
    LabelPropagation {
        /// The seed that fixed the node visiting order.
        seed: u64,
        /// Rounds actually run.
        rounds_used: usize,
        /// Whether a full round passed with no node changing label. **A partition
        /// that did not converge is still a partition**, and it is reported as
        /// such rather than retried silently or raised as an error.
        converged: bool,
    },
}

impl Detection {
    /// The algorithm's name, for a passport that has to say what ran.
    #[must_use]
    pub const fn algorithm(&self) -> &'static str {
        match *self {
            Self::LabelPropagation { .. } => "label_propagation",
        }
    }
}

/// Where the blocks came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Partition {
    /// An attribute the source declared — subnet, role, protocol group.
    ///
    /// This arm claims nothing about community structure. It is the entity kind
    /// the data already carried, and a mixing matrix over it answers "how do
    /// these declared groups talk to each other", not "which hosts form a
    /// cluster".
    Declared {
        /// The attribute the labels were read from, named so a reader of the
        /// passport can tell which question the fit answered.
        attribute: String,
        /// Block index per node, indexed by node id.
        labels: Vec<usize>,
    },
    /// Found in the graph itself.
    Detected {
        /// The algorithm and its parameters.
        detection: Detection,
        /// Block index per node, indexed by node id.
        labels: Vec<usize>,
    },
}

impl Partition {
    /// Block index per node.
    #[must_use]
    pub fn labels(&self) -> &[usize] {
        match self {
            Self::Declared { labels, .. } | Self::Detected { labels, .. } => labels,
        }
    }

    /// The number of blocks, i.e. one more than the largest label.
    #[must_use]
    pub fn n_blocks(&self) -> usize {
        self.labels().iter().copied().max().map_or(0, |max| max + 1)
    }

    /// A one-word provenance for the passport: `declared` or `detected`.
    #[must_use]
    pub const fn provenance(&self) -> &'static str {
        match *self {
            Self::Declared { .. } => "declared",
            Self::Detected { .. } => "detected",
        }
    }
}

/// Rewrites arbitrary label values to a dense `0..K` range, preserving the order
/// in which the labels are first seen so the result is deterministic.
#[must_use]
pub fn compact(raw: &[usize]) -> Vec<usize> {
    let mut seen: Vec<usize> = Vec::new();
    raw.iter()
        .map(|label| {
            seen.iter()
                .position(|known| known == label)
                .unwrap_or_else(|| {
                    seen.push(*label);
                    seen.len() - 1
                })
        })
        .collect()
}

/// Detects blocks by asynchronous label propagation.
///
/// Near-linear per round, which is why it is here first: it is the cheapest
/// thing that can be wrong in an interesting way, and a fit whose partition
/// nobody trusts is a fit nobody can read.
///
/// Determinism: the node visiting order is drawn once per round from `seed`, and
/// ties between equally-frequent neighbour labels go to the smallest label id.
/// Two runs with the same seed and the same edge list give the same partition.
///
/// Isolated nodes keep their own singleton label, which is the honest answer:
/// nothing in the graph places them.
#[must_use]
pub fn detect_label_propagation(
    edges: &[(usize, usize)],
    n_nodes: usize,
    seed: u64,
    max_rounds: usize,
) -> Partition {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(source, target) in edges {
        if source < n_nodes && target < n_nodes && source != target {
            adjacency[source].push(target);
            adjacency[target].push(source);
        }
    }

    let mut labels: Vec<usize> = (0..n_nodes).collect();
    let mut order: Vec<usize> = (0..n_nodes).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut rounds_used = 0;
    let mut converged = false;

    for _ in 0..max_rounds {
        order.shuffle(&mut rng);
        rounds_used += 1;
        let mut changed = false;
        for &node in &order {
            if adjacency[node].is_empty() {
                continue;
            }
            let mut tally: Vec<(usize, usize)> = Vec::new();
            for &neighbour in &adjacency[node] {
                let label = labels[neighbour];
                if let Some(entry) = tally.iter_mut().find(|(known, _)| *known == label) {
                    entry.1 += 1;
                } else {
                    tally.push((label, 1));
                }
            }
            // Ties to the smallest label id, so the outcome does not depend on
            // neighbour ordering.
            let best = tally
                .iter()
                .copied()
                .fold((usize::MAX, 0usize), |best, candidate| {
                    if candidate.1 > best.1 || (candidate.1 == best.1 && candidate.0 < best.0) {
                        candidate
                    } else {
                        best
                    }
                });
            if best.0 != usize::MAX && labels[node] != best.0 {
                labels[node] = best.0;
                changed = true;
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }

    Partition::Detected {
        detection: Detection::LabelPropagation {
            seed,
            rounds_used,
            converged,
        },
        labels: compact(&labels),
    }
}
