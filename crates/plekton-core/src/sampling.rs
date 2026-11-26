// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Edge samplers and the scaffolding shared by every backbone generator: RNG
//! construction, DC-SBM degree-correction weights, the block-pair sampling
//! loop, and effective-homophily — kept in one place so SBM / cSBM / BA stay
//! focused on their distinct algorithm.

use std::collections::BTreeMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Geometric, Pareto, Uniform};

use crate::SynthGraphError;

// Upper-triangle index decode (used by geometric skip sampling)
//
// Maps a linear index flat_idx (0-based) in the upper triangle of an n×n
// matrix to the pair (row, col) with 0 ≤ row < col < matrix_size.
//
// Row `row` contains (matrix_size-1-row) entries. Cumulative count before row:
//   cumul(row) = row*(2*matrix_size-1-row)/2
//
// We binary-search for the largest row s.t. cumul(row) ≤ flat_idx.

/// Maps a flat upper-triangle index to `(row, col)` with `row < col < matrix_size`.
fn decode_upper_tri(flat_idx: u64, matrix_size: usize) -> (usize, usize) {
    debug_assert!(matrix_size >= 2);
    let size64 = matrix_size as u64;
    let cumul = |row: u64| row * (2 * size64 - 1 - row) / 2;

    // Binary search: largest row in [0, matrix_size-2] s.t. cumul(row) <= flat_idx
    let mut lo = 0u64;
    let mut hi = size64 - 2; // last valid row (has exactly 1 entry)
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if cumul(mid) <= flat_idx {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let row = usize::try_from(lo).unwrap_or(usize::MAX);
    let col = row + 1 + usize::try_from(flat_idx - cumul(lo)).unwrap_or(usize::MAX);
    (row, col)
}

// Geometric skip sampling helpers
//
// For a Bernoulli(p) process over `total` positions the gap between consecutive
// successes is Geometric(p) — number of failures before the next success.
// This turns O(N²) trial-by-trial work into O(E) where E is the output size.
//
// For DC-SBM we use p_max = p_base * max_θ_a * max_θ_b as the proposal and
// accept each candidate with probability p_actual/p_max (rejection sampling).

/// Degree-correction acceptance parameters shared by the intra/inter-block edge
/// samplers. Grouping them keeps both kernel signatures within the argument-count
/// limit and avoids threading the same four values through every call.
#[derive(Clone, Copy)]
struct DcSbmSampling<'a> {
    /// Per-node degree-correction weights θ, indexed by node id.
    theta: &'a [f64],
    /// Base connection probability for the block (pair).
    p_base: f64,
    /// Proposal ceiling, clipped to [0, 1].
    p_max: f64,
    /// Whether degree-correction rejection sampling is active.
    use_dc: bool,
}

/// Samples intra-block edges using geometric skip sampling.
fn sample_intra_block(
    nodes: &[usize],
    sampling: &DcSbmSampling,
    rng: &mut ChaCha8Rng,
    edges: &mut Vec<(i64, i64)>,
) {
    let DcSbmSampling {
        theta,
        p_base,
        p_max,
        use_dc,
    } = *sampling;
    let block_size = nodes.len();
    if block_size < 2 || p_max <= 0.0 {
        return;
    }
    let total = (block_size as u64) * (block_size as u64 - 1) / 2;
    let p_proposal = p_max.min(1.0);
    // The guard above ensures p_max > 0 ⇒ p_proposal ∈ (0, 1], so `Geometric::new`
    // cannot return `Err`.
    // Guard above ensures p_proposal ∈ (0, 1]; the else branch is unreachable.
    let Ok(geom) = Geometric::new(p_proposal) else {
        return;
    };
    let uniform = Uniform::new(0.0f64, 1.0);

    let mut pair_idx: u64 = geom.sample(rng);
    while pair_idx < total {
        let (row, col) = decode_upper_tri(pair_idx, block_size);
        let node_u = nodes[row];
        let node_v = nodes[col];

        let accept = if use_dc {
            let p_actual = (p_base * theta[node_u] * theta[node_v]).min(1.0);
            p_actual >= p_proposal || uniform.sample(rng) < p_actual / p_proposal
        } else {
            true // p_proposal == p_base, always accept
        };

        if accept {
            let (lo, hi) = if node_u < node_v {
                (node_u, node_v)
            } else {
                (node_v, node_u)
            };
            edges.push((
                i64::try_from(lo).unwrap_or(i64::MAX),
                i64::try_from(hi).unwrap_or(i64::MAX),
            ));
        }

        let gap: u64 = geom.sample(rng);
        pair_idx = match pair_idx.checked_add(gap + 1) {
            Some(next) => next,
            None => break,
        };
    }
}

/// Samples inter-block edges between two distinct blocks using geometric skip sampling.
fn sample_inter_block(
    nodes_a: &[usize],
    nodes_b: &[usize],
    sampling: &DcSbmSampling,
    rng: &mut ChaCha8Rng,
    edges: &mut Vec<(i64, i64)>,
) {
    let DcSbmSampling {
        theta,
        p_base,
        p_max,
        use_dc,
    } = *sampling;
    let count_a = nodes_a.len() as u64;
    let count_b = nodes_b.len() as u64;
    let total = count_a * count_b;
    if total == 0 || p_max <= 0.0 {
        return;
    }
    let p_proposal = p_max.min(1.0);
    // The guard above ensures p_max > 0 ⇒ p_proposal ∈ (0, 1], so `Geometric::new`
    // cannot return `Err`.
    // Guard above ensures p_proposal ∈ (0, 1]; the else branch is unreachable.
    let Ok(geom) = Geometric::new(p_proposal) else {
        return;
    };
    let uniform = Uniform::new(0.0f64, 1.0);

    let mut pair_idx: u64 = geom.sample(rng);
    while pair_idx < total {
        let row = usize::try_from(pair_idx / count_b).unwrap_or(usize::MAX);
        let col = usize::try_from(pair_idx % count_b).unwrap_or(usize::MAX);
        let node_u = nodes_a[row];
        let node_v = nodes_b[col];

        let accept = if use_dc {
            let p_actual = (p_base * theta[node_u] * theta[node_v]).min(1.0);
            p_actual >= p_proposal || uniform.sample(rng) < p_actual / p_proposal
        } else {
            true
        };

        if accept {
            let (lo, hi) = if node_u < node_v {
                (node_u, node_v)
            } else {
                (node_v, node_u)
            };
            edges.push((
                i64::try_from(lo).unwrap_or(i64::MAX),
                i64::try_from(hi).unwrap_or(i64::MAX),
            ));
        }

        let gap: u64 = geom.sample(rng);
        pair_idx = match pair_idx.checked_add(gap + 1) {
            Some(next) => next,
            None => break,
        };
    }
}

/// Expands canonical undirected edges to a bidirectional flat `[2, 2E]` array for `PyG`.
///
/// Each canonical edge `(src, dst)` produces `src→dst` and `dst→src`.
/// Row 0 holds all sources, row 1 all targets (column-major within the row).
#[must_use]
pub fn build_bidirectional_edge_flat(canonical: &[(i64, i64)]) -> (Vec<i64>, usize) {
    let n_canonical = canonical.len();
    let total_directed = 2 * n_canonical;
    let mut flat = vec![0i64; 2 * total_directed];
    for (edge_idx, &(src, dst)) in canonical.iter().enumerate() {
        flat[edge_idx] = src;
        flat[n_canonical + edge_idx] = dst;
        flat[total_directed + edge_idx] = dst;
        flat[total_directed + n_canonical + edge_idx] = src;
    }
    (flat, total_directed)
}

// Shared generator helpers
//
// Small pieces common to the SBM / cSBM / BA backbones. Kept here next to the
// edge samplers so the three generators stay focused on their distinct algorithm
// and share their incidental scaffolding (RNG construction, DC-SBM θ, the
// block-pair sampling loop, effective-homophily) rather than re-implementing it.

/// Builds the generator PRNG: a `ChaCha8Rng` from a fixed `seed`
/// (`seed_from_u64`) for reproducible output, or from OS entropy when `None`.
/// Centralised so every backbone seeds identically — the load-bearing "same seed
/// ⇒ identical graph" guarantee. `ChaCha8Rng` rather than rand's `SmallRng`,
/// which rand does not promise to be reproducible across platforms or versions.
pub fn seeded_rng(seed: Option<u64>) -> ChaCha8Rng {
    seed.map_or_else(SeedableRng::from_entropy, SeedableRng::seed_from_u64)
}

/// Effective homophily: the fraction of canonical edges whose endpoints share a
/// community label. Defined as `0.0` for an edgeless graph. No RNG.
pub fn effective_homophily(edges: &[(i64, i64)], labels: &[usize]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }
    let intra = edges
        .iter()
        .filter(|&&(src, dst)| {
            labels[usize::try_from(src).unwrap_or(usize::MAX)]
                == labels[usize::try_from(dst).unwrap_or(usize::MAX)]
        })
        .count();
    num_traits::ToPrimitive::to_f64(&(intra)).unwrap_or(0.0)
        / num_traits::ToPrimitive::to_f64(&(edges.len())).unwrap_or(0.0)
}

/// How per-node degree correction θ is obtained.
///
/// Three arms rather than an `Option<f64>` beside an `Option<&[f64]>`, because
/// "drawn from a Pareto" and "supplied by a fit" are mutually exclusive, and a
/// pair of options would let a caller ask for both and leave the tie-break to
/// whichever branch happens to be written first.
#[derive(Debug, Clone, Copy)]
pub enum DegreeCorrection<'theta> {
    /// Plain SBM: every θ = 1. Consumes no RNG values.
    Off,
    /// DC-SBM with θ drawn from Pareto(γ−1). Consumes exactly `n_nodes`
    /// RNG values, in node order.
    Drawn {
        /// The Pareto exponent γ; must be finite and > 2 for a finite mean.
        exponent: f64,
    },
    /// DC-SBM with θ measured per node — a fit, not a draw. Consumes **no**
    /// RNG values, so replacing a drawn θ with a fitted one does not shift every
    /// subsequent draw in the pipeline.
    Supplied(&'theta [f64]),
}

impl DegreeCorrection<'_> {
    /// Whether the sampler applies the θ rejection step at all.
    #[must_use]
    pub const fn is_corrected(&self) -> bool {
        !matches!(*self, Self::Off)
    }
}

/// Computes the DC-SBM degree-correction weights θ and the per-block maximum θ
/// used as the rejection-sampling ceiling, shared by SBM and cSBM.
///
/// [`DegreeCorrection::Off`] is the plain-SBM identity (all θ = 1).
/// [`DegreeCorrection::Drawn`] draws `n_nodes` Pareto(γ−1) samples from `rng`.
/// [`DegreeCorrection::Supplied`] takes the caller's vector. The `Drawn` and
/// `Supplied` arms both normalise each block to unit mean, so block connection
/// probabilities stay on target.
///
/// **Normalising a supplied θ is a gauge fix, not an edit of a measurement.**
/// The sampler uses `B[a,b]·θ_i·θ_j`, in which only θ's *shape within a
/// block* is identifiable: any per-block rescaling of θ is absorbed by `B`.
/// Passing θ and 3·θ therefore yields the same graph, and a test says so.
///
/// Draws exactly `n_nodes` RNG values in node order in the `Drawn` case and none
/// in the `Off` or `Supplied` cases — the calling generators rely on that exact
/// consumption.
///
/// # Errors
/// Returns [`SynthGraphError`] if the Pareto distribution cannot be built, if any
/// block is empty (defensive — callers enforce `n_communities ≤ n_nodes`), or
/// if a supplied θ has the wrong length, holds a non-finite or negative entry,
/// or has a block summing to zero, which no rescaling brings to unit mean.
pub fn compute_dc_theta(
    correction: DegreeCorrection<'_>,
    n_nodes: usize,
    labels: &[usize],
    block_nodes: &[Vec<usize>],
    rng: &mut ChaCha8Rng,
) -> Result<(Vec<f64>, Vec<f64>), SynthGraphError> {
    let n_communities = block_nodes.len();
    let raw: Vec<f64> = match correction {
        DegreeCorrection::Off => {
            return Ok((vec![1.0f64; n_nodes], vec![1.0f64; n_communities]));
        }
        DegreeCorrection::Drawn { exponent } => {
            let pareto = Pareto::new(1.0, exponent - 1.0).map_err(|pareto_err| {
                SynthGraphError::GenerationFailed(format!("Pareto dist error: {pareto_err}"))
            })?;
            (0..n_nodes).map(|_| pareto.sample(rng)).collect()
        }
        DegreeCorrection::Supplied(supplied) => {
            if supplied.len() != n_nodes {
                return Err(SynthGraphError::InvalidParameter(format!(
                    "supplied theta must have one entry per node (n_nodes={n_nodes}), got {}",
                    supplied.len()
                )));
            }
            if let Some(bad) = supplied
                .iter()
                .position(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(SynthGraphError::InvalidParameter(format!(
                    "supplied theta[{bad}]={} is not a finite non-negative weight",
                    supplied[bad]
                )));
            }
            supplied.to_vec()
        }
    };

    for (community_idx, community_nodes) in block_nodes.iter().enumerate() {
        if community_nodes.is_empty() {
            return Err(SynthGraphError::InvalidParameter(format!(
                "block {community_idx} has no nodes; cannot normalise DC-SBM theta \
                 (ensure n_communities ≤ n_nodes)"
            )));
        }
    }
    let mut block_sum = vec![0.0f64; n_communities];
    for (node_idx, &community) in labels.iter().enumerate() {
        block_sum[community] += raw[node_idx];
    }
    if let Some(flat) = block_sum.iter().position(|sum| *sum <= 0.0) {
        return Err(SynthGraphError::InvalidParameter(format!(
            "theta sums to {} over block {flat}; no rescaling gives it unit mean",
            block_sum[flat]
        )));
    }
    let block_mean: Vec<f64> = (0..n_communities)
        .map(|community_idx| {
            block_sum[community_idx]
                / num_traits::ToPrimitive::to_f64(&(block_nodes[community_idx].len()))
                    .unwrap_or(0.0)
        })
        .collect();
    let theta: Vec<f64> = raw
        .iter()
        .enumerate()
        .map(|(node_idx, &raw_theta)| raw_theta / block_mean[labels[node_idx]])
        .collect();
    let block_max_theta: Vec<f64> = (0..n_communities)
        .map(|community_idx| {
            block_nodes[community_idx]
                .iter()
                .map(|&node_idx| theta[node_idx])
                .fold(0.0f64, f64::max)
        })
        .collect();
    Ok((theta, block_max_theta))
}

/// Exact IEEE-754 binary exponent `⌊log2(weight)⌋` for a positive finite weight.
///
/// Used to group nodes into dyadic buckets `[2^r, 2^(r+1))` so every pair of
/// buckets has acceptance probability strictly `> 1/4` under DC-SBM rejection
/// sampling.
#[inline]
fn dyadic_bucket_key(weight: f64) -> i32 {
    let raw_exp = u16::try_from((weight.to_bits() >> 52) & 0x7ff).unwrap_or(0);
    i32::from(raw_exp) - 1023
}

/// Partitions a block's positive-weight nodes into dyadic θ buckets
/// `(nodes_in_bucket, bucket_max_theta)`, ordered deterministically by binary
/// exponent ascending (and original node order within each bucket).
///
/// Nodes with `theta == 0` are omitted because their edge probability is zero.
fn partition_dyadic_buckets(nodes: &[usize], theta: &[f64]) -> Vec<(Vec<usize>, f64)> {
    let mut by_exp: BTreeMap<i32, (Vec<usize>, f64)> = BTreeMap::new();
    for &node_idx in nodes {
        let weight = theta[node_idx];
        if weight <= 0.0 || !weight.is_finite() {
            continue;
        }
        let exp = dyadic_bucket_key(weight);
        let entry = by_exp.entry(exp).or_insert_with(|| (Vec::new(), 0.0_f64));
        entry.0.push(node_idx);
        if weight > entry.1 {
            entry.1 = weight;
        }
    }
    by_exp.into_values().collect()
}

/// Samples every block pair `(a, b)` with `a ≤ b`, appending canonical edges
/// (`src < dst`) to `edges`.
///
/// `p_base_for(a, b)` supplies the block-pair connection probability
/// (`p_in`/`p_out` for SBM, the `B`-matrix entry for cSBM); zero-probability and
/// DC-ceiling-zero pairs are skipped.
///
/// When `use_dc` is `false`, runs plain geometric skip sampling in `O(E)`.
/// When `use_dc` is `true`, partitions each block into dyadic weight buckets
/// `[2^r, 2^(r+1))` so that every bucket-pair proposal has exact acceptance
/// probability `> 1/4`, yielding exact `Bernoulli(min(1, p_base · θ_u · θ_v))`
/// sampling in `O(N + E)` expected time (instead of `O(N²)` under heavy tails).
///
/// Iterates in a fixed deterministic order so RNG consumption — and the emitted
/// edge order — is reproducible across platforms.
pub fn sample_all_blocks(
    block_nodes: &[Vec<usize>],
    theta: &[f64],
    block_max_theta: &[f64],
    use_dc: bool,
    p_base_for: impl Fn(usize, usize) -> f64,
    rng: &mut ChaCha8Rng,
    edges: &mut Vec<(i64, i64)>,
) {
    let n_communities = block_nodes.len();
    if !use_dc {
        for block_a in 0..n_communities {
            for block_b in block_a..n_communities {
                let p_base = p_base_for(block_a, block_b);
                if p_base <= 0.0 {
                    continue;
                }
                let sampling = DcSbmSampling {
                    theta,
                    p_base,
                    p_max: p_base,
                    use_dc: false,
                };
                if block_a == block_b {
                    sample_intra_block(&block_nodes[block_a], &sampling, rng, edges);
                } else {
                    sample_inter_block(
                        &block_nodes[block_a],
                        &block_nodes[block_b],
                        &sampling,
                        rng,
                        edges,
                    );
                }
            }
        }
        return;
    }

    let block_buckets: Vec<Vec<(Vec<usize>, f64)>> = block_nodes
        .iter()
        .map(|nodes| partition_dyadic_buckets(nodes, theta))
        .collect();

    for block_a in 0..n_communities {
        for block_b in block_a..n_communities {
            let p_base = p_base_for(block_a, block_b);
            if p_base <= 0.0 {
                continue;
            }
            if block_max_theta[block_a] <= 0.0 || block_max_theta[block_b] <= 0.0 {
                continue;
            }
            if block_a == block_b {
                let buckets = &block_buckets[block_a];
                for idx_r in 0..buckets.len() {
                    let (ref nodes_r, max_r) = buckets[idx_r];
                    let p_max = (p_base * max_r * max_r).min(1.0);
                    if p_max > 0.0 {
                        let sampling = DcSbmSampling {
                            theta,
                            p_base,
                            p_max,
                            use_dc: true,
                        };
                        sample_intra_block(nodes_r, &sampling, rng, edges);
                    }
                    for (nodes_s, max_s) in &buckets[(idx_r + 1)..] {
                        let p_max = (p_base * max_r * *max_s).min(1.0);
                        if p_max > 0.0 {
                            let sampling = DcSbmSampling {
                                theta,
                                p_base,
                                p_max,
                                use_dc: true,
                            };
                            sample_inter_block(nodes_r, nodes_s, &sampling, rng, edges);
                        }
                    }
                }
            } else {
                for (nodes_r, max_r) in &block_buckets[block_a] {
                    for (nodes_s, max_s) in &block_buckets[block_b] {
                        let p_max = (p_base * *max_r * *max_s).min(1.0);
                        if p_max > 0.0 {
                            let sampling = DcSbmSampling {
                                theta,
                                p_base,
                                p_max,
                                use_dc: true,
                            };
                            sample_inter_block(nodes_r, nodes_s, &sampling, rng, edges);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_upper_tri_n2_k0() {
        assert_eq!(decode_upper_tri(0, 2), (0, 1));
    }

    #[test]
    fn decode_upper_tri_n4_all_entries() {
        let expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (flat_idx, &(expected_row, expected_col)) in expected.iter().enumerate() {
            assert_eq!(
                decode_upper_tri(flat_idx as u64, 4),
                (expected_row, expected_col),
                "flat_idx={flat_idx} should map to ({expected_row},{expected_col})"
            );
        }
    }

    #[test]
    fn decode_upper_tri_last_entry_n3() {
        assert_eq!(decode_upper_tri(2, 3), (1, 2));
    }

    #[test]
    fn bidirectional_flat_empty() {
        let (flat, total_directed) = build_bidirectional_edge_flat(&[]);
        assert_eq!(total_directed, 0);
        assert_eq!(flat.len(), 0);
    }

    #[test]
    fn bidirectional_flat_single_edge() {
        let (flat, total_directed) = build_bidirectional_edge_flat(&[(2, 5)]);
        assert_eq!(total_directed, 2);
        // Row-major 2×2: [src_fwd, src_bwd | dst_fwd, dst_bwd] = [2, 5, 5, 2]
        assert_eq!(flat, vec![2i64, 5, 5, 2]);
    }

    #[test]
    fn bidirectional_flat_two_edges() {
        let canonical = vec![(0i64, 1i64), (2i64, 3i64)];
        let (flat, total_directed) = build_bidirectional_edge_flat(&canonical);
        assert_eq!(total_directed, 4);
        assert_eq!(flat.len(), 2 * total_directed);
        // Row 0 (sources): fwd then rev = [0, 2, 1, 3]
        // Row 1 (targets): fwd then rev = [1, 3, 0, 2]
        assert_eq!(&flat[..4], &[0, 2, 1, 3]);
        assert_eq!(&flat[4..], &[1, 3, 0, 2]);
    }
}
