// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Blockmodel estimation: the inverse of a stochastic-block-model generator.
//!
//! Given an edge list and a partition, this crate measures the three quantities a
//! DC-SBM generator consumes — block sizes, a K×K mixing matrix, and a
//! per-node degree correction θ — so that a graph can be generated with the
//! shape of the one that was measured, without the measured graph leaving the
//! machine it was measured on.
//!
//! # What it is not
//!
//! * **Not a likelihood.** The mixing matrix is the method-of-moments estimate:
//!   observed edges over possible pairs, per block pair. There is no EM, no
//!   variational bound, and no model-selection over K.
//! * **Not a community-detection benchmark.** [`partition::detect_label_propagation`]
//!   is here so that the detected arm exists at all; it is not claimed to be the
//!   best partition of anything.
//! * **Not a fidelity claim.** What a fit reproduces is exactly the statistics it
//!   consumed. Everything else is unmatched, and it is the caller's passport that
//!   has to say so.
//!
//! # Determinism
//!
//! Nothing here draws random numbers except the two detectors, which take a seed:
//! [`partition::detect_label_propagation`] and [`overlap::detect_overlapping`]
//! draw from a `rand_chacha::ChaCha8Rng` built with `seed_from_u64(seed)`, so the
//! same seed and the same edges give the same result, and
//! `tests/determinism_golden.rs` pins both outputs. [`fit`] is a pure function of
//! its inputs.

pub mod overlap;
pub mod partition;

use std::fmt;

pub use overlap::{detect_overlapping, Overlapping};
pub use partition::{compact, detect_label_propagation, Detection, Partition};

/// Largest K² this crate will allocate for a mixing matrix.
///
/// 2 000 blocks — 4 M `f64` entries, 32 MB. The bound is on the *projection*,
/// checked before anything is allocated, because a partition with one node per
/// block is not an error a caller notices in advance: it is what a detection
/// algorithm returns on a graph with no structure, and on a million-node graph it
/// would ask for eight terabytes.
pub const MAX_MIXING_ENTRIES: usize = 4_000_000;

/// Why a fit could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// The label vector does not have one entry per node.
    LabelCount {
        /// Labels supplied.
        labels: usize,
        /// Nodes the caller declared.
        n_nodes: usize,
    },
    /// An edge names a node outside `0..n_nodes`.
    EdgeOutOfRange {
        /// The offending endpoint.
        node: usize,
        /// Nodes the caller declared.
        n_nodes: usize,
    },
    /// The partition names no blocks, so there is nothing to estimate.
    NoBlocks,
    /// The partition is fine enough that its mixing matrix would not fit.
    MixingMatrixTooLarge {
        /// Blocks the partition names.
        n_blocks: usize,
        /// Entries the matrix would need.
        projected: usize,
    },
}

impl fmt::Display for FitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::LabelCount { labels, n_nodes } => write!(
                formatter,
                "the partition has {labels} labels but the graph has {n_nodes} nodes"
            ),
            Self::EdgeOutOfRange { node, n_nodes } => write!(
                formatter,
                "an edge names node {node}, which is outside 0..{n_nodes}"
            ),
            Self::NoBlocks => write!(
                formatter,
                "the partition names no blocks; there is nothing to estimate"
            ),
            Self::MixingMatrixTooLarge {
                n_blocks,
                projected,
            } => write!(
                formatter,
                "this partition names {n_blocks} blocks, whose mixing matrix needs \
                 {projected} entries, above the {MAX_MIXING_ENTRIES} budget. Coarsen the \
                 partition rather than raising the ceiling: a matrix estimated over blocks \
                 this fine is estimated from a handful of pairs per entry, and a generator \
                 driven by it reproduces the input graph rather than its shape."
            ),
        }
    }
}

impl std::error::Error for FitError {}

/// A condition the fit met that a reader of its output would otherwise not see.
///
/// Not warnings-as-strings: each carries the block it is about, so a passport can
/// report *where* a number is uninformative rather than that one somewhere is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitNote {
    /// A block whose nodes have no edges at all, so degree ratios are undefined
    /// there and its θ is the uniform 1 rather than a measurement.
    BlockHasNoEdges {
        /// The block index.
        block: usize,
        /// How many nodes are in it.
        size: usize,
    },
    /// A block of one node. Its within-block density has no possible pairs to be
    /// a fraction of, so `B[k,k]` is 0 by construction rather than by observation.
    BlockOfOne {
        /// The block index.
        block: usize,
    },
}

/// What an estimator measured, in the form a generator consumes.
///
/// `theta` and `block_sizes` are in **generator node order**: nodes sorted by
/// block, and by original node id within a block. That is the order
/// `BlockAssignment::Sizes` realises, so a fit can be handed straight back
/// without the caller having to know the permutation.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    /// The partition the fit is conditional on.
    pub partition: Partition,
    /// Nodes per block, in block order. Sums to `n_nodes`.
    pub block_sizes: Vec<usize>,
    /// K×K mixing matrix, row-major, symmetric: observed edges over possible
    /// pairs for each block pair.
    pub b_matrix: Vec<f64>,
    /// Per-node degree correction in generator node order, unit mean per block.
    pub theta: Vec<f64>,
    /// Nodes measured.
    pub n_nodes: usize,
    /// Distinct undirected edges measured, self-loops and duplicates removed.
    pub n_edges: usize,
    /// Conditions met while estimating.
    pub notes: Vec<FitNote>,
}

impl Fit {
    /// The statistics this estimator **read**, and therefore the only ones a graph
    /// generated from the fit can be said to have matched.
    ///
    /// This list is what makes a generator passport's "matched" section computed
    /// rather than typed. Nobody writes "temporal patterns matched" into an
    /// artifact; the passport derives the matched set from here and reports every
    /// other statistic it can compute as unmatched, with the divergence it
    /// measured.
    ///
    /// Note what is *not* here. Density is a deterministic function of the mixing
    /// matrix and the block sizes, so it is *entailed* — but entailed is not read,
    /// and claiming it as matched would be a modelling claim rather than a record
    /// of what the estimator did.
    pub const CONSUMED: &'static [&'static str] = &[
        "n_nodes",
        "n_edges",
        "block_sizes",
        "mixing_matrix",
        "degree_sequence",
    ];

    /// The statistics this fit read. See [`Self::CONSUMED`].
    #[must_use]
    pub const fn consumed(&self) -> &'static [&'static str] {
        Self::CONSUMED
    }

    /// The number of blocks.
    #[must_use]
    pub const fn n_blocks(&self) -> usize {
        self.block_sizes.len()
    }

    /// Blocks per node — the degenerate-partition ratio.
    ///
    /// A partition with one block per node reproduces the input graph exactly, so
    /// a generator driven by it is a copy machine. This ratio is what a passport
    /// checks against a pre-registered bound.
    #[must_use]
    pub fn blocks_per_node(&self) -> f64 {
        if self.n_nodes == 0 {
            return 0.0;
        }
        count_to_f64(self.n_blocks()) / count_to_f64(self.n_nodes)
    }
}

/// Lossless `usize` -> `f64` for counts, avoiding a precision-losing cast.
#[inline]
fn count_to_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// Reduces an edge list to sorted, deduplicated, self-loop-free undirected pairs.
///
/// The model has no term for a self-loop or a repeated pair, and counting a
/// repeated pair twice would inflate exactly the density the fit is measuring.
///
/// # Errors
/// Returns [`FitError::EdgeOutOfRange`] if an endpoint is outside `0..n_nodes`.
fn canonicalise(edges: &[(usize, usize)], n_nodes: usize) -> Result<Vec<(usize, usize)>, FitError> {
    let mut canonical: Vec<(usize, usize)> = Vec::with_capacity(edges.len());
    for &(source, target) in edges {
        for endpoint in [source, target] {
            if endpoint >= n_nodes {
                return Err(FitError::EdgeOutOfRange {
                    node: endpoint,
                    n_nodes,
                });
            }
        }
        if source != target {
            canonical.push((source.min(target), source.max(target)));
        }
    }
    canonical.sort_unstable();
    canonical.dedup();
    Ok(canonical)
}

/// The method-of-moments mixing matrix: observed edges over possible pairs, for
/// every block pair.
///
/// `possible` is zero for a block of one node, whose within-block entry is then
/// 0 by construction rather than by observation — which is what the returned
/// [`FitNote::BlockOfOne`] says.
fn estimate_mixing(
    pair_counts: &[f64],
    block_sizes: &[usize],
    notes: &mut Vec<FitNote>,
) -> Vec<f64> {
    let n_blocks = block_sizes.len();
    let mut b_matrix = vec![0f64; n_blocks * n_blocks];
    for block_a in 0..n_blocks {
        let size_a = count_to_f64(block_sizes[block_a]);
        for block_b in 0..n_blocks {
            let possible = if block_a == block_b {
                size_a * (size_a - 1.0) / 2.0
            } else {
                size_a * count_to_f64(block_sizes[block_b])
            };
            b_matrix[block_a * n_blocks + block_b] = if possible > 0.0 {
                (pair_counts[block_a * n_blocks + block_b] / possible).min(1.0)
            } else {
                0.0
            };
        }
        if block_sizes[block_a] == 1 {
            notes.push(FitNote::BlockOfOne { block: block_a });
        }
    }
    b_matrix
}

/// Per-node degree correction, in generator node order.
///
/// `θ_i = d_i / (mean degree of i's block)`, which is unit mean per block by
/// construction — the normalisation the sampler assumes, since it uses
/// `B[a,b]·θ_i·θ_j` and only θ's shape within a block is identifiable.
///
/// A block with no edges has no degree ratios; its θ is the uniform 1, reported
/// as [`FitNote::BlockHasNoEdges`] so a reader can tell a measurement from a
/// stand-in.
fn estimate_theta(
    degree: &[usize],
    labels: &[usize],
    block_sizes: &[usize],
    notes: &mut Vec<FitNote>,
) -> Vec<f64> {
    let n_blocks = block_sizes.len();
    let mut block_degree_sum = vec![0f64; n_blocks];
    for (node, &node_degree) in degree.iter().enumerate() {
        block_degree_sum[labels[node]] += count_to_f64(node_degree);
    }

    let mut order: Vec<usize> = (0..degree.len()).collect();
    order.sort_by_key(|&node| (labels[node], node));

    let theta = order
        .iter()
        .map(|&node| {
            let block = labels[node];
            let mean = block_degree_sum[block] / count_to_f64(block_sizes[block]);
            if mean > 0.0 {
                count_to_f64(degree[node]) / mean
            } else {
                1.0
            }
        })
        .collect();

    for (block, &sum) in block_degree_sum.iter().enumerate() {
        if sum == 0.0 {
            notes.push(FitNote::BlockHasNoEdges {
                block,
                size: block_sizes[block],
            });
        }
    }
    theta
}

/// Estimates block sizes, a mixing matrix and a per-node θ from an edge list.
///
/// Edges are treated as undirected; self-loops and duplicate pairs are dropped,
/// and `n_edges` reports what remained.
///
/// # Errors
/// Returns [`FitError`] if the labels do not cover the nodes, an edge names a node
/// outside the graph, the partition names no blocks, or the mixing matrix would
/// exceed [`MAX_MIXING_ENTRIES`].
pub fn fit(
    edges: &[(usize, usize)],
    n_nodes: usize,
    partition: Partition,
) -> Result<Fit, FitError> {
    let labels = partition.labels();
    if labels.len() != n_nodes {
        return Err(FitError::LabelCount {
            labels: labels.len(),
            n_nodes,
        });
    }
    let n_blocks = partition.n_blocks();
    if n_blocks == 0 {
        return Err(FitError::NoBlocks);
    }

    // Projection before allocation. A partition of one node per block asks for
    // n×n entries, and on a large graph that is the whole machine.
    let projected = n_blocks.saturating_mul(n_blocks);
    if projected > MAX_MIXING_ENTRIES {
        return Err(FitError::MixingMatrixTooLarge {
            n_blocks,
            projected,
        });
    }

    let mut block_sizes = vec![0usize; n_blocks];
    for &label in labels {
        block_sizes[label] += 1;
    }

    let canonical = canonicalise(edges, n_nodes)?;

    let mut pair_counts = vec![0f64; projected];
    let mut degree = vec![0usize; n_nodes];
    for &(source, target) in &canonical {
        let (block_a, block_b) = (labels[source], labels[target]);
        pair_counts[block_a * n_blocks + block_b] += 1.0;
        if block_a != block_b {
            pair_counts[block_b * n_blocks + block_a] += 1.0;
        }
        degree[source] += 1;
        degree[target] += 1;
    }

    let mut notes = Vec::new();
    let b_matrix = estimate_mixing(&pair_counts, &block_sizes, &mut notes);
    let theta = estimate_theta(&degree, labels, &block_sizes, &mut notes);

    Ok(Fit {
        partition,
        block_sizes,
        b_matrix,
        theta,
        n_nodes,
        n_edges: canonical.len(),
        notes,
    })
}
