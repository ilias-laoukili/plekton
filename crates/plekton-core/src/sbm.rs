// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! SBM / DC-SBM (Stochastic Block Model / Degree-Corrected SBM) backbone generator.

use crate::{
    sampling::{compute_dc_theta, edge_homophily, sample_all_blocks, seeded_rng, DegreeCorrection},
    types::SbmCoreResult,
    SynthGraphError,
};

/// Tolerance on the symmetry of a supplied mixing matrix.
///
/// A `B` estimated from counts is symmetric by construction but arrives through
/// JSON, so it is symmetric to the last printed digit rather than bit for bit.
const B_SYMMETRY_TOLERANCE: f64 = 1e-9;

/// Which nodes belong to which block.
///
/// The generator has always produced contiguous, near-equal blocks. A fit does
/// not: a partition measured on a real network has blocks of whatever size the
/// network has, and applying a fitted `B` to equal blocks would generate at a
/// density nobody asked for. Both are here, named, rather than one being the
/// default and the other reachable only by arithmetic on `n_communities`.
#[derive(Debug, Clone, Copy)]
pub enum BlockAssignment<'sizes> {
    /// Contiguous near-equal blocks: `labels[i] = i·K/n`. The historical shape.
    Balanced {
        /// Number of blocks K; must be in `1..=n_nodes`.
        n_communities: usize,
    },
    /// Block sizes a fit measured. Blocks stay contiguous in node order, so the
    /// caller supplies θ and `B` in that same order.
    Sizes(&'sizes [usize]),
}

impl BlockAssignment<'_> {
    /// Expands to per-node block labels, validating the assignment against `n_nodes`.
    ///
    /// # Errors
    /// Returns [`SynthGraphError::InvalidParameter`] for an empty or oversized
    /// `Balanced`, or for `Sizes` that are empty, contain a zero, or do not sum
    /// to `n_nodes`.
    fn labels(&self, n_nodes: usize) -> Result<Vec<usize>, SynthGraphError> {
        match *self {
            Self::Balanced { n_communities } => {
                if n_communities == 0 {
                    return Err(SynthGraphError::InvalidParameter(
                        "n_nodes and n_communities must be > 0".into(),
                    ));
                }
                if n_communities > n_nodes {
                    return Err(SynthGraphError::InvalidParameter(format!(
                        "n_communities ({n_communities}) must be ≤ n_nodes ({n_nodes})"
                    )));
                }
                Ok((0..n_nodes)
                    .map(|node_idx| node_idx * n_communities / n_nodes)
                    .collect())
            }
            Self::Sizes(sizes) => {
                if sizes.is_empty() {
                    return Err(SynthGraphError::InvalidParameter(
                        "block sizes must name at least one block".into(),
                    ));
                }
                if let Some(empty) = sizes.iter().position(|size| *size == 0) {
                    return Err(SynthGraphError::InvalidParameter(format!(
                        "block {empty} has size 0; an empty block has no θ to normalise \
                         and no density to reproduce"
                    )));
                }
                let total: usize = sizes.iter().sum();
                if total != n_nodes {
                    return Err(SynthGraphError::InvalidParameter(format!(
                        "block sizes sum to {total}, not n_nodes ({n_nodes})"
                    )));
                }
                let mut labels = Vec::with_capacity(n_nodes);
                for (block, size) in sizes.iter().enumerate() {
                    labels.extend(std::iter::repeat_n(block, *size));
                }
                Ok(labels)
            }
        }
    }

    /// The number of blocks this assignment names.
    const fn n_communities(&self) -> usize {
        match *self {
            Self::Balanced { n_communities } => n_communities,
            Self::Sizes(sizes) => sizes.len(),
        }
    }
}

/// How block-pair connection probabilities are set.
///
/// [`Self::Uniform`] is the two-scalar surface every existing caller uses.
/// [`Self::Matrix`] is what an estimator produces, and the reason this enum
/// exists: `p_in`/`p_out` cannot express a `B` in which block 0 talks to block
/// 3 and to nothing else, which is the shape real networks turn out to have.
#[derive(Debug, Clone, Copy)]
pub enum BlockConnection<'matrix> {
    /// One within-block and one between-block probability, both in `[0, 1]`.
    Uniform {
        /// Probability of an edge between two nodes of the same block.
        p_in: f64,
        /// Probability of an edge between two nodes of different blocks.
        p_out: f64,
    },
    /// A full K×K mixing matrix in row-major order, symmetric to
    /// [`B_SYMMETRY_TOLERANCE`], every entry in `[0, 1]`.
    Matrix(&'matrix [f64]),
}

impl BlockConnection<'_> {
    /// Validates the connection against a block count.
    ///
    /// # Errors
    /// Returns [`SynthGraphError::InvalidParameter`] for an out-of-range
    /// probability, a wrong-length matrix, or an asymmetric one.
    fn validate(&self, n_communities: usize) -> Result<(), SynthGraphError> {
        match *self {
            Self::Uniform { p_in, p_out } => {
                if !(0.0..=1.0).contains(&p_in) || !(0.0..=1.0).contains(&p_out) {
                    return Err(SynthGraphError::InvalidParameter(
                        "p_in and p_out must be in [0, 1]".into(),
                    ));
                }
            }
            Self::Matrix(matrix) => {
                if matrix.len() != n_communities * n_communities {
                    return Err(SynthGraphError::InvalidParameter(format!(
                        "b_matrix must have length n_communities²={}, got {}",
                        n_communities * n_communities,
                        matrix.len()
                    )));
                }
                if matrix
                    .iter()
                    .any(|prob| !prob.is_finite() || !(0.0..=1.0).contains(prob))
                {
                    return Err(SynthGraphError::InvalidParameter(
                        "b_matrix values must be finite and in [0, 1]".into(),
                    ));
                }
                for row in 0..n_communities {
                    for col in (row + 1)..n_communities {
                        let upper = matrix[row * n_communities + col];
                        let lower = matrix[col * n_communities + row];
                        if (upper - lower).abs() > B_SYMMETRY_TOLERANCE {
                            return Err(SynthGraphError::InvalidParameter(format!(
                                "b_matrix must be symmetric: B[{row},{col}]={upper} != \
                                 B[{col},{row}]={lower}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The connection probability for the block pair `(block_a, block_b)`.
    const fn probability(&self, n_communities: usize, block_a: usize, block_b: usize) -> f64 {
        match *self {
            Self::Uniform { p_in, p_out } => {
                if block_a == block_b {
                    p_in
                } else {
                    p_out
                }
            }
            Self::Matrix(matrix) => matrix[block_a * n_communities + block_b],
        }
    }
}

/// Everything a fitted generator needs that `p_in`/`p_out`/γ cannot carry.
///
/// This is the type the estimator in `graph-fit` produces and the only way a
/// measured mixing matrix or a measured θ reaches the sampler.
#[derive(Debug, Clone, Copy)]
pub struct SbmFit<'fit> {
    /// Which nodes belong to which block.
    pub assignment: BlockAssignment<'fit>,
    /// The block-pair connection probabilities.
    pub connection: BlockConnection<'fit>,
    /// The per-node degree correction.
    pub degree_correction: DegreeCorrection<'fit>,
}

/// Generates an SBM / DC-SBM graph from a fit.
///
/// [`generate_sbm_core`] is this function with the historical arguments packed
/// into [`BlockAssignment::Balanced`] + [`BlockConnection::Uniform`], so the two
/// share one implementation and one set of invariants.
///
/// # Errors
/// Returns [`SynthGraphError`] if `n_nodes` is zero, if the assignment does not
/// cover exactly `n_nodes` nodes, if the connection is out of range or
/// asymmetric, or if the degree correction is invalid for this node count.
pub fn generate_sbm_from_fit(
    n_nodes: usize,
    fit: &SbmFit<'_>,
    seed: Option<u64>,
) -> Result<SbmCoreResult, SynthGraphError> {
    if n_nodes == 0 {
        return Err(SynthGraphError::InvalidParameter(
            "n_nodes and n_communities must be > 0".into(),
        ));
    }
    if let DegreeCorrection::Drawn { exponent } = fit.degree_correction {
        if !exponent.is_finite() {
            return Err(SynthGraphError::InvalidParameter(
                "theta_exponent must be finite".into(),
            ));
        }
        if exponent <= 2.0 {
            return Err(SynthGraphError::InvalidParameter(
                "theta_exponent (γ) must be > 2 (finite-mean Pareto requirement)".into(),
            ));
        }
    }
    let n_communities = fit.assignment.n_communities();
    fit.connection.validate(n_communities)?;
    let labels = fit.assignment.labels(n_nodes)?;

    let mut rng = seeded_rng(seed);

    let mut block_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_communities];
    for (node_idx, &community) in labels.iter().enumerate() {
        block_nodes[community].push(node_idx);
    }

    let use_dc = fit.degree_correction.is_corrected();
    let (theta, block_max_theta) = compute_dc_theta(
        fit.degree_correction,
        n_nodes,
        &labels,
        &block_nodes,
        &mut rng,
    )?;

    // Pre-allocate the edge buffer to avoid `O(log E)` reallocations during
    // sampling. Expected canonical-edge count:
    //   Σ_a |B_a|·(|B_a|-1)/2 · B[a,a]  +  Σ_{a<b} |B_a|·|B_b| · B[a,b]
    // with 15 % headroom to absorb DC-SBM variance (DC-SBM accept rates vary
    // between blocks, so the actual count can exceed the Erdős–Rényi mean).
    let expected_edges = {
        let mut expected_accum: f64 = 0.0;
        for community_a in 0..n_communities {
            let block_size_a =
                num_traits::ToPrimitive::to_f64(&(block_nodes[community_a].len())).unwrap_or(0.0);
            expected_accum = (block_size_a * (block_size_a - 1.0) / 2.0).mul_add(
                fit.connection
                    .probability(n_communities, community_a, community_a),
                expected_accum,
            );
            for (offset, block_b) in block_nodes[(community_a + 1)..].iter().enumerate() {
                let block_size_b = num_traits::ToPrimitive::to_f64(&(block_b.len())).unwrap_or(0.0);
                expected_accum = (block_size_a * block_size_b).mul_add(
                    fit.connection.probability(
                        n_communities,
                        community_a,
                        community_a + 1 + offset,
                    ),
                    expected_accum,
                );
            }
        }
        num_traits::ToPrimitive::to_usize(&(expected_accum * 1.15)).unwrap_or(0)
    };
    let mut edges: Vec<(i64, i64)> = Vec::with_capacity(expected_edges);
    sample_all_blocks(
        &block_nodes,
        &theta,
        &block_max_theta,
        use_dc,
        |block_a, block_b| fit.connection.probability(n_communities, block_a, block_b),
        &mut rng,
        &mut edges,
    );

    let effective_h = edge_homophily(&edges, &labels)?;

    Ok(SbmCoreResult {
        canonical_edges: edges,
        labels,
        effective_h,
        n_nodes,
    })
}

// SBM core

/// Generates an SBM or DC-SBM graph; returns a pure-Rust result with no Python types.
///
/// The balanced-blocks, two-scalar surface. A fitted mixing matrix, fitted block
/// sizes or a fitted θ go through [`generate_sbm_from_fit`] instead; this
/// function is that one with [`BlockAssignment::Balanced`] and
/// [`BlockConnection::Uniform`], and its output is unchanged.
///
/// # Errors
/// Returns [`SynthGraphError`] if the parameters are invalid (e.g. mismatched
/// community/probability dimensions or out-of-range probabilities).
pub fn generate_sbm_core(
    n_nodes: usize,
    n_communities: usize,
    p_in: f64,
    p_out: f64,
    theta_exponent: Option<f64>,
    seed: Option<u64>,
) -> Result<SbmCoreResult, SynthGraphError> {
    if n_communities == 0 || n_nodes == 0 {
        return Err(SynthGraphError::InvalidParameter(
            "n_nodes and n_communities must be > 0".into(),
        ));
    }
    let fit = SbmFit {
        assignment: BlockAssignment::Balanced { n_communities },
        connection: BlockConnection::Uniform { p_in, p_out },
        degree_correction: theta_exponent.map_or(DegreeCorrection::Off, |exponent| {
            DegreeCorrection::Drawn { exponent }
        }),
    };
    generate_sbm_from_fit(n_nodes, &fit, seed)
}
