// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Approximate effective-resistance edge scores (Spielman-Srivastava).
//!
//! Effective resistance `R_eff(u, v)` measures the "electrical distance" between
//! two nodes: bridge / bottleneck edges have **high** resistance, redundant
//! edges (many alternative paths) have **low** resistance.
//!
//! ## Method
//!
//! Rather than the `O(n³)` exact Laplacian pseudoinverse, this uses the
//! Johnson-Lindenstrauss projection of Spielman & Srivastava (2011):
//!
//! 1. Draw a random Gaussian matrix `R` (m × k), scaled by `1/√k`.
//! 2. Project through the signed incidence matrix `B`: `Y = B · R` (n × k).
//! 3. Solve `L · Z = Y` for each of the `k` columns (`L` = graph Laplacian),
//!    giving `Z = L⁺ · B · R`.
//! 4. `R_eff(u, v) ≈ ‖Z[u] − Z[v]‖²`.
//!
//! The `k` linear systems are solved with a **dependency-free conjugate-gradient
//! solver** operating directly on the Laplacian via adjacency-list matrix-vector
//! products (`(L x)[i] = deg(i)·x[i] − Σ_{j∈N(i)} x[j]`), so this crate needs no
//! sparse-linear-algebra dependency. A small regularisation (`+1e-6·I`) makes the
//! Laplacian positive-definite so CG converges on connected and disconnected
//! graphs alike.
//!
//! ## Cost
//!
//! `k = round(24·ln n / ε²)` projected columns (at least 1, capped by
//! `max_dim`), each solved by at most `max_cg_iters` iterations of
//! unpreconditioned conjugate gradient costing `O(n + m)` apiece:
//! `O(k · max_cg_iters · (n + m))` time in the worst case, and `2·n·k` `f64` of
//! memory for the projection and the solutions. The near-linear running time of
//! Spielman and Srivastava relies on a near-linear-time Laplacian solver, which
//! this crate does not have.
//!
//! ## Determinism
//!
//! The Gaussian draws use a seeded [`ChaCha8Rng`] in edge order, so results are
//! reproducible across platforms for a given seed. They come from stream 0, the
//! default, of `ChaCha8Rng::seed_from_u64(seed)`; weighted sampling
//! ([`crate::Strategy::Sampled`]) reads stream 1 of the same construction, so
//! one seed given to both never makes the sampling re-read these draws. As with
//! any sampling method, another seed (or another implementation of the method)
//! gives different scores, each approximating the same `R_eff` within the JL
//! error bound.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::SparsifyError;

/// Regularisation added to the Laplacian diagonal so CG converges even when the
/// graph is disconnected (the Laplacian is otherwise singular on each component).
const LAPLACIAN_REGULARISATION: f64 = 1e-6;

/// Lower bound applied to returned resistances (avoids exact zeros from noise).
const RESISTANCE_FLOOR: f64 = 1e-10;

/// Configuration for [`approx_er_scores`].
#[derive(Debug, Clone, Copy)]
pub struct ApproxErConfig {
    /// JL approximation error `ε`, a finite number `> 0`. Smaller is more
    /// accurate but uses more projections (`k = 24·ln(n)/ε²`). Default `0.3`.
    pub epsilon: f64,
    /// RNG seed for the Gaussian projection.
    pub seed: u64,
    /// Maximum conjugate-gradient iterations per column solve. Default `500`.
    pub max_cg_iters: usize,
    /// CG convergence tolerance on the residual L2 norm. Default `1e-6`.
    pub cg_tol: f64,
    /// Optional cap on the JL dimension `k`. `None` uses the formula's `k`
    /// unchanged; `Some(cap)` bounds `k ≤ cap` so the method stays
    /// tractable on large subgraphs (identical behaviour when the formula's `k`
    /// is already `≤ cap`, e.g. on small graphs and tests).
    pub max_dim: Option<usize>,
}

impl Default for ApproxErConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.3,
            seed: 42,
            max_cg_iters: 500,
            cg_tol: 1e-6,
            max_dim: None,
        }
    }
}

/// The JL projection dimension `k = max(round(24·ln(max(n,2))/ε²), 1)`, capped
/// by `max_dim` when set.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "k is a small positive count; the formula is inherently floating-point"
)]
fn jl_dimension(n_nodes: usize, config: &ApproxErConfig) -> usize {
    let ln_n = (n_nodes.max(2) as f64).ln();
    let raw = (24.0 * ln_n / (config.epsilon * config.epsilon)).round() as usize;
    let k = raw.max(1);
    config.max_dim.map_or(k, |cap| k.min(cap.max(1)))
}

/// Compute approximate effective-resistance scores for every edge.
///
/// Returns a `Vec<f64>` aligned with `edges`; higher values indicate more
/// critical (bridge-like) edges. Self-loops score `0.0`. An empty graph returns
/// a zero-filled vector of the right length.
///
/// See the [module docs](self) for the algorithm and complexity.
///
/// # Errors
/// - [`SparsifyError::InvalidEpsilon`] if `config.epsilon` is not a finite
///   number `> 0` (checked before anything else, whatever `max_dim` is).
/// - [`SparsifyError::ProjectionTooLarge`] if the `n_nodes × k` projection
///   overflows or cannot be allocated — `k` grows as `1/ε²`, so a vanishing
///   `epsilon` without a `max_dim` cap gets here instead of aborting.
pub fn approx_er_scores(
    n_nodes: usize,
    edges: &[(usize, usize)],
    config: &ApproxErConfig,
) -> Result<Vec<f64>, SparsifyError> {
    // `ε = 0` would drive `k` to `usize::MAX` and overflow the projection
    // buffer; NaN and ±∞ would silently collapse `k` to 1. None is an error bound.
    if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
        return Err(SparsifyError::InvalidEpsilon(config.epsilon));
    }

    let n_edges = edges.len();
    if n_edges == 0 || n_nodes == 0 {
        return Ok(vec![0.0; n_edges]);
    }

    // Adjacency (for Laplacian mat-vec) and degrees, ignoring self-loops.
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    let mut degree: Vec<f64> = vec![0.0; n_nodes];
    for &(u, v) in edges {
        if u != v {
            neighbours[u].push(v);
            neighbours[v].push(u);
            degree[u] += 1.0;
            degree[v] += 1.0;
        }
    }

    let k = jl_dimension(n_nodes, config);
    let projection = build_projection(n_nodes, edges, k, config.seed)?;

    // Solve L_reg · z_j = y_j for each column j (independent → parallel).
    let z_columns: Vec<Vec<f64>> = (0..k)
        .into_par_iter()
        .map(|column| {
            let rhs: Vec<f64> = (0..n_nodes)
                .map(|node| projection[node * k + column])
                .collect();
            conjugate_gradient(
                &neighbours,
                &degree,
                &rhs,
                config.max_cg_iters,
                config.cg_tol,
            )
        })
        .collect();

    // R_eff(u, v) ≈ ‖Z[u] − Z[v]‖² over the k projected dimensions.
    Ok(edges
        .par_iter()
        .map(|&(u, v)| {
            if u == v {
                return 0.0;
            }
            let squared: f64 = z_columns
                .iter()
                .map(|column| {
                    let delta = column[u] - column[v];
                    delta * delta
                })
                .sum();
            squared.max(RESISTANCE_FLOOR)
        })
        .collect())
}

/// Build the row-major `n × k` projection `Y = B · R`, where `R` is a Gaussian
/// matrix scaled by `1/√k` and `B` is the signed incidence matrix.
///
/// `R` is never materialised: each edge draws its own `k`-vector and adds it to
/// the source row and subtracts it from the target row. Draws happen in edge
/// order from a seeded RNG, so the projection is deterministic.
///
/// The `n_nodes × k` buffer is sized with checked arithmetic and reserved
/// fallibly, so an overflowing or unallocatable size is a
/// [`SparsifyError::ProjectionTooLarge`] rather than a panic or an abort.
#[allow(
    clippy::cast_precision_loss,
    reason = "k is small; 1/√k scale is inherently floating-point"
)]
fn build_projection(
    n_nodes: usize,
    edges: &[(usize, usize)],
    k: usize,
    seed: u64,
) -> Result<Vec<f64>, SparsifyError> {
    let too_large = || SparsifyError::ProjectionTooLarge { n_nodes, k };
    let len = n_nodes.checked_mul(k).ok_or_else(too_large)?;
    let mut projection: Vec<f64> = Vec::new();
    projection.try_reserve_exact(len).map_err(|_| too_large())?;
    projection.resize(len, 0.0);

    let scale = 1.0 / (k as f64).sqrt();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for &(u, v) in edges {
        if u == v {
            continue;
        }
        let (u_base, v_base) = (u * k, v * k);
        for offset in 0..k {
            let gaussian: f64 = StandardNormal.sample(&mut rng);
            let value = gaussian * scale;
            projection[u_base + offset] += value;
            projection[v_base + offset] -= value;
        }
    }
    Ok(projection)
}

/// Solve `(L + reg·I) x = b` by conjugate gradient, where `L` is the graph
/// Laplacian described by `neighbours` and `degree`.
///
/// The Laplacian is applied matrix-free: `(L x)[i] = deg(i)·x[i] − Σ_{j∈N(i)} x[j]`.
/// Returns the solution vector; iterates until the residual norm drops below
/// `tol` or `max_iters` is reached.
#[allow(
    clippy::suboptimal_flops,
    reason = "plain mul+add (no FMA) keeps CG rounding identical across platforms"
)]
fn conjugate_gradient(
    neighbours: &[Vec<usize>],
    degree: &[f64],
    b: &[f64],
    max_iters: usize,
    tol: f64,
) -> Vec<f64> {
    let n = b.len();
    let matvec = |x: &[f64], out: &mut [f64]| {
        for i in 0..n {
            let neighbour_sum: f64 = neighbours[i].iter().map(|&j| x[j]).sum();
            out[i] = (degree[i] + LAPLACIAN_REGULARISATION) * x[i] - neighbour_sum;
        }
    };
    let dot = |a: &[f64], c: &[f64]| -> f64 { a.iter().zip(c).map(|(x, y)| x * y).sum() };

    let mut x = vec![0.0; n];
    let mut residual = b.to_vec(); // r = b − A·0
    let mut direction = residual.clone();
    let mut rs_old = dot(&residual, &residual);
    let mut ap = vec![0.0; n];

    for _ in 0..max_iters {
        if rs_old.sqrt() < tol {
            break;
        }
        matvec(&direction, &mut ap);
        let denom = dot(&direction, &ap);
        if denom.abs() < f64::MIN_POSITIVE {
            break;
        }
        let alpha = rs_old / denom;
        for i in 0..n {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * ap[i];
        }
        let rs_new = dot(&residual, &residual);
        let beta = rs_new / rs_old;
        for i in 0..n {
            direction[i] = residual[i] + beta * direction[i];
        }
        rs_old = rs_new;
    }

    x
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
    fn empty_graph_returns_zeros() {
        let scores = approx_er_scores(0, &[], &ApproxErConfig::default()).unwrap();
        assert_eq!(scores, [] as [f64; 0]);
        let scores = approx_er_scores(3, &[], &ApproxErConfig::default()).unwrap();
        assert_eq!(scores, [] as [f64; 0]);
    }

    #[test]
    fn invalid_epsilon_is_rejected() {
        // ε = 0 used to overflow the projection and abort the process; NaN and
        // ±∞ silently collapsed k to 1. All are refused, even with a max_dim cap
        // and on an edgeless graph.
        let edges = [(0, 1), (1, 2)];
        for bad in [0.0, -0.3, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (n_nodes, graph) in [(3, &edges[..]), (3, &[][..])] {
                for max_dim in [None, Some(8)] {
                    let config = ApproxErConfig {
                        epsilon: bad,
                        max_dim,
                        ..ApproxErConfig::default()
                    };
                    let err = approx_er_scores(n_nodes, graph, &config).unwrap_err();
                    assert!(
                        matches!(err, SparsifyError::InvalidEpsilon(e) if e.to_bits() == bad.to_bits()),
                        "epsilon {bad} gave {err:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn vanishing_epsilon_errors_instead_of_overflowing() {
        // 24·ln(3)/ε² saturates k at usize::MAX, so n × k overflows: a typed
        // error, not a panic. (A merely small ε that still fits would request a
        // real, huge allocation — never exercise that in a test.)
        let config = ApproxErConfig {
            epsilon: 1e-10,
            ..ApproxErConfig::default()
        };
        assert_eq!(jl_dimension(3, &config), usize::MAX);
        let err = approx_er_scores(3, &[(0, 1), (1, 2)], &config).unwrap_err();
        assert_eq!(
            err,
            SparsifyError::ProjectionTooLarge {
                n_nodes: 3,
                k: usize::MAX
            }
        );
        // A one-node graph does not overflow the multiplication, but k entries of
        // f64 still exceed what a Vec can hold.
        let err = approx_er_scores(1, &[(0, 0)], &config).unwrap_err();
        assert!(matches!(err, SparsifyError::ProjectionTooLarge { .. }));
    }

    #[test]
    fn bridge_has_higher_resistance_than_triangle_edge() {
        // Two triangles {0,1,2} and {3,4,5} joined by a single bridge (2,3).
        // The bridge is the only path between the halves → highest resistance.
        let edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3), // bridge
            (3, 4),
            (3, 5),
            (4, 5),
        ];
        let config = ApproxErConfig {
            epsilon: 0.1,
            seed: 1,
            ..ApproxErConfig::default()
        };
        let scores = approx_er_scores(6, &edges, &config).unwrap();
        let bridge = scores[3];
        let triangle_max = scores
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 3)
            .map(|(_, &s)| s)
            .fold(0.0_f64, f64::max);
        assert!(
            bridge > triangle_max,
            "bridge {bridge} should exceed triangle max {triangle_max}",
        );
    }

    #[test]
    fn deterministic_per_seed() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
        let config = ApproxErConfig {
            seed: 99,
            ..ApproxErConfig::default()
        };
        let a = approx_er_scores(4, &edges, &config).unwrap();
        let b = approx_er_scores(4, &edges, &config).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn scores_are_unchanged_by_the_sampling_stream() {
        // Frozen bit for bit on main (bc3b398) before sampling moved to its own
        // stream: the projection still reads stream 0, so the scores did not move.
        let edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)];
        let cases: [(u64, usize, [u64; 7]); 2] = [
            (
                1,
                8,
                [
                    0x3fd6_3827_4340_8f80,
                    0x3ff2_f229_daf0_d433,
                    0x3ff6_1392_a26b_c9cc,
                    0x3ff9_0f54_40a9_6b24,
                    0x3fed_5b1f_fa64_ede3,
                    0x3ff4_3f4f_f6fb_6a6f,
                    0x3fe4_15e0_2d96_6a8b,
                ],
            ),
            (
                7,
                4,
                [
                    0x3fed_2871_d891_9076,
                    0x3fd5_e05b_1481_5df6,
                    0x3fec_3773_684f_a955,
                    0x3fa4_4840_55ce_7711,
                    0x3fd3_4756_a151_71e5,
                    0x3fe0_3bab_1c22_bd4c,
                    0x3fd4_2bbb_74a3_5914,
                ],
            ),
        ];
        for (seed, max_dim, want) in cases {
            let config = ApproxErConfig {
                epsilon: 0.3,
                seed,
                max_dim: Some(max_dim),
                ..ApproxErConfig::default()
            };
            let got: Vec<u64> = approx_er_scores(6, &edges, &config)
                .unwrap()
                .iter()
                .map(|score| score.to_bits())
                .collect();
            assert_eq!(got, want, "seed {seed}, max_dim {max_dim}");
        }
    }

    #[test]
    fn max_dim_caps_projection() {
        // With max_dim=4, k is capped regardless of the epsilon-driven formula.
        let config = ApproxErConfig {
            epsilon: 0.05, // formula would give a large k
            max_dim: Some(4),
            ..ApproxErConfig::default()
        };
        assert_eq!(jl_dimension(1000, &config), 4);
    }
}
