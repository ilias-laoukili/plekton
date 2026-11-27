// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! cSBM (contextual Stochastic Block Model) backbone generator — graph topology plus correlated node features.

use std::collections::HashSet;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp, Normal, Uniform};

use crate::{
    sampling::{compute_dc_theta, edge_homophily, sample_all_blocks, seeded_rng, DegreeCorrection},
    types::CsbmCoreResult,
    SynthGraphError,
};

/// Configuration for [`generate_csbm`].
///
/// Use struct-update syntax with [`CsbmConfig::default`] to override only the
/// parameters of interest:
///
/// ```
/// # fn main() -> Result<(), plekton::SynthGraphError> {
/// use plekton::{generate_csbm, CsbmConfig};
///
/// let graph = generate_csbm(&CsbmConfig {
///     n_nodes: 50,
///     n_communities: 2,
///     homophily: 0.8,
///     avg_degree: 6.0,
///     feat_dim: 8,
///     mu: 1.5,
///     seed: Some(42),
///     ..CsbmConfig::default()
/// })?;
/// assert_eq!(graph.n_nodes, 50);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CsbmConfig<'a> {
    /// Total number of nodes (`> 0`).
    pub n_nodes: usize,
    /// Number of communities / blocks (`1..=n_nodes`).
    pub n_communities: usize,
    /// Target edge homophily in `[0, 1]`.
    pub homophily: f64,
    /// Target average degree (`>= 0`, finite).
    pub avg_degree: f64,
    /// Node feature dimension (`> 0`).
    pub feat_dim: usize,
    /// Class centroid separation scale `μ` (`>= 0`, finite).
    pub mu: f64,
    /// Feature noise distribution: `"gaussian"`, `"uniform"`, or `"laplacian"`.
    pub feature_dist: &'a str,
    /// Optional class proportions summing to `1.0` (even split when `None`).
    pub class_weights: Option<&'a [f64]>,
    /// Optional DC-SBM Pareto exponent `γ > 2` for heavy-tailed degrees.
    pub theta_exponent: Option<f64>,
    /// Triangle-closing probability per sampled edge in `[0, 1]`.
    pub p_triangle: f64,
    /// Per-node feature label-noise probability in `[0, 1]`.
    pub feat_noise_ratio: f64,
    /// When `true`, stitch disconnected components into a single component.
    pub ensure_connected: bool,
    /// Explicit intra-community edge probability (must be paired with `p_out_override`).
    pub p_in_override: Option<f64>,
    /// Explicit inter-community edge probability (must be paired with `p_in_override`).
    pub p_out_override: Option<f64>,
    /// Explicit symmetric `K × K` row-major block probability matrix overriding `p_in`/`p_out`.
    pub b_matrix: Option<&'a [f64]>,
    /// Optional explicit `K × feat_dim` row-major class centroids.
    pub centroids_override: Option<&'a [f64]>,
    /// Optional positive-definite `feat_dim × feat_dim` row-major noise covariance matrix.
    pub covariance: Option<&'a [f64]>,
    /// Optional deterministic PRNG seed (`None` draws OS entropy).
    pub seed: Option<u64>,
}

impl Default for CsbmConfig<'_> {
    fn default() -> Self {
        Self {
            n_nodes: 100,
            n_communities: 2,
            homophily: 0.7,
            avg_degree: 5.0,
            feat_dim: 16,
            mu: 1.0,
            feature_dist: "gaussian",
            class_weights: None,
            theta_exponent: None,
            p_triangle: 0.0,
            feat_noise_ratio: 0.0,
            ensure_connected: false,
            p_in_override: None,
            p_out_override: None,
            b_matrix: None,
            centroids_override: None,
            covariance: None,
            seed: None,
        }
    }
}

/// Node feature noise distribution for cSBM generation.
enum FeatDist {
    /// Zero-mean unit-variance Gaussian.
    Gaussian,
    /// Uniform on `[-√3, √3]` (unit variance).
    Uniform,
    /// Laplacian with unit variance (scale = 1/√2).
    Laplacian,
}

// cSBM core

/// Cholesky decomposition of a positive-definite matrix (flat row-major, `dim`×`dim`).
///
/// Returns the lower-triangular factor `L` with `Σ = L · Lᵀ`, also flat
/// row-major (upper triangle stays zero). Used to generate correlated feature
/// vectors `ε = L · z` where `z ~ N(0, I)` — this lets cSBM users pass an
/// arbitrary covariance instead of being forced into iid noise.
///
/// # Errors
/// Returns [`SynthGraphError::InvalidParameter`] if the matrix is not positive
/// definite (a non-positive pivot appears during reduction).
#[allow(
    clippy::suboptimal_flops,
    reason = "preserves exact IEEE rounding during Cholesky factorisation"
)]
fn cholesky_decompose(matrix: &[f64], dim: usize) -> Result<Vec<f64>, SynthGraphError> {
    let mut lower = vec![0.0f64; dim * dim];
    for col_idx in 0..dim {
        let mut pivot_accum = matrix[col_idx * dim + col_idx];
        for reduction_idx in 0..col_idx {
            pivot_accum -=
                lower[col_idx * dim + reduction_idx] * lower[col_idx * dim + reduction_idx];
        }
        if pivot_accum <= 0.0 {
            return Err(SynthGraphError::InvalidParameter(format!(
                "covariance matrix is not positive definite \
                 (pivot [{col_idx},{col_idx}] = {pivot_accum:.6e} after reduction)",
            )));
        }
        lower[col_idx * dim + col_idx] = pivot_accum.sqrt();
        for row_idx in (col_idx + 1)..dim {
            let mut entry_accum = matrix[row_idx * dim + col_idx];
            for reduction_idx in 0..col_idx {
                entry_accum -=
                    lower[row_idx * dim + reduction_idx] * lower[col_idx * dim + reduction_idx];
            }
            lower[row_idx * dim + col_idx] = entry_accum / lower[col_idx * dim + col_idx];
        }
    }
    Ok(lower)
}

/// Applies lower-triangular Cholesky factor `L` to `standard_normal_sample`,
/// writing `L · standard_normal_sample` into `output`. When the factor is
/// absent (identity covariance) the input is copied through unchanged — this
/// is the iid-noise fast path.
#[inline]
fn apply_chol_or_copy(
    cholesky_factor: Option<&[f64]>,
    standard_normal_sample: &[f64],
    output: &mut [f64],
    dim: usize,
) {
    match cholesky_factor {
        Some(lower) => {
            for row_idx in 0..dim {
                output[row_idx] = (0..=row_idx)
                    .map(|reduction_idx| {
                        lower[row_idx * dim + reduction_idx] * standard_normal_sample[reduction_idx]
                    })
                    .sum();
            }
        }
        None => output.copy_from_slice(standard_normal_sample),
    }
}

/// Pushes one node's worth of features into `x_flat` given the per-feature
/// correlated noise already in `correlated_noise_buffer`.
fn push_node_features(
    x_flat: &mut Vec<f32>,
    centroids: &[f64],
    correlated_noise_buffer: &[f64],
    feat_dim: usize,
    class_idx: usize,
    mu: f64,
) {
    let centroid_offset = class_idx * feat_dim;
    x_flat.extend(
        correlated_noise_buffer
            .iter()
            .enumerate()
            .map(|(feat_idx, &noise_value)| {
                // Two rounded ops (not `mul_add`) to keep the blessed IEEE result.
                let scaled = mu * centroids[centroid_offset + feat_idx];
                num_traits::ToPrimitive::to_f32(&(scaled + noise_value)).unwrap_or(0.0)
            }),
    );
}

/// Generates the per-node correlated feature block (`x_flat`) for the cSBM.
///
/// # Errors
/// Propagates a Cholesky failure or an (unreachable) invalid noise scale.
/// Reassigns a node to a wrong class with probability `feat_noise_ratio`
/// (per-node label noise); returns the true class otherwise.
fn perturbed_class(
    true_class: usize,
    n_communities: usize,
    feat_noise_ratio: f64,
    rng: &mut ChaCha8Rng,
) -> usize {
    if n_communities > 1 && feat_noise_ratio > 0.0 && rng.gen::<f64>() < feat_noise_ratio {
        let class_shift = rng.gen_range(1..n_communities);
        (true_class + class_shift) % n_communities
    } else {
        true_class
    }
}

/// Runs the shared per-node feature loop: draw each raw noise value from
/// `sample_noise`, correlate it through the Cholesky factor, and append the row.
fn fill_feature_rows(
    config: &CsbmConfig<'_>,
    labels: &[usize],
    centroids: &[f64],
    chol: Option<&[f64]>,
    rng: &mut ChaCha8Rng,
    x_flat: &mut Vec<f32>,
    mut sample_noise: impl FnMut(&mut ChaCha8Rng) -> f64,
) {
    let mut standard_normal_buffer = vec![0.0f64; config.feat_dim];
    let mut correlated_noise_buffer = vec![0.0f64; config.feat_dim];
    for &label in labels {
        let class_idx = perturbed_class(label, config.n_communities, config.feat_noise_ratio, rng);
        for sample_slot in &mut standard_normal_buffer {
            *sample_slot = sample_noise(rng);
        }
        apply_chol_or_copy(
            chol,
            &standard_normal_buffer,
            &mut correlated_noise_buffer,
            config.feat_dim,
        );
        push_node_features(
            x_flat,
            centroids,
            &correlated_noise_buffer,
            config.feat_dim,
            class_idx,
            config.mu,
        );
    }
}

fn generate_node_features(
    config: &CsbmConfig<'_>,
    dist_enum: &FeatDist,
    labels: &[usize],
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f32>, SynthGraphError> {
    let feat_dim = config.feat_dim;
    // Use caller-supplied centroids or generate `N(0, 1/√feat_dim)` defaults.
    let centroids: Vec<f64> = if let Some(supplied) = config.centroids_override {
        supplied.to_vec()
    } else {
        {
            let centroid_std =
                (1.0_f64 / num_traits::ToPrimitive::to_f64(&feat_dim).unwrap_or(0.0)).sqrt();
            // feat_dim ≥ 1 ⇒ centroid_std > 0, so `Normal::new` cannot return `Err`.
            let Ok(centroid_dist) = Normal::new(0.0_f64, centroid_std) else {
                return Err(SynthGraphError::InvalidParameter(
                    "centroid scale must be positive".into(),
                ));
            };
            (0..config.n_communities * feat_dim)
                .map(|_| centroid_dist.sample(rng))
                .collect()
        }
    };

    // Pre-compute Cholesky factor L once (None → Z_i ~ N(0,I) / U / Laplacian iid).
    let chol: Option<Vec<f64>> = match config.covariance {
        Some(cov) => Some(cholesky_decompose(cov, feat_dim)?),
        None => None,
    };

    let mut x_flat: Vec<f32> = Vec::with_capacity(config.n_nodes * feat_dim);
    let ch = chol.as_deref();
    match *dist_enum {
        FeatDist::Gaussian => {
            // σ = 1.0 > 0 (literal), so `Normal::new` cannot return `Err`.
            let Ok(gaussian_noise) = Normal::new(0.0_f64, 1.0_f64) else {
                return Err(SynthGraphError::InvalidParameter(
                    "gaussian noise scale must be positive".into(),
                ));
            };
            fill_feature_rows(config, labels, &centroids, ch, rng, &mut x_flat, |r| {
                gaussian_noise.sample(r)
            });
        }
        FeatDist::Uniform => {
            let sqrt_three = 3.0_f64.sqrt();
            let uniform_noise = Uniform::new(-sqrt_three, sqrt_three);
            fill_feature_rows(config, labels, &centroids, ch, rng, &mut x_flat, |r| {
                uniform_noise.sample(r)
            });
        }
        FeatDist::Laplacian => {
            // Laplacian (unit variance) = scale · sign · Exp(1), scale = 1/√2.
            let laplacian_scale = 1.0_f64 / 2.0_f64.sqrt();
            let Ok(exponential) = Exp::new(1.0_f64) else {
                return Err(SynthGraphError::InvalidParameter(
                    "exponential rate must be positive".into(),
                ));
            };
            let sign_uniform = Uniform::new(0.0_f64, 1.0_f64);
            fill_feature_rows(config, labels, &centroids, ch, rng, &mut x_flat, |r| {
                let exponential_magnitude = exponential.sample(r);
                let sign_value = if sign_uniform.sample(r) < 0.5 {
                    1.0_f64
                } else {
                    -1.0_f64
                };
                laplacian_scale * exponential_magnitude * sign_value
            });
        }
    }
    Ok(x_flat)
}

/// Validates the cSBM scalar/array parameters and resolves the feature
/// distribution.
///
/// # Errors
/// Returns [`SynthGraphError::InvalidParameter`] for any out-of-range,
/// mismatched, or non-finite input.
fn validate_csbm_params(config: &CsbmConfig<'_>) -> Result<FeatDist, SynthGraphError> {
    if config.n_nodes == 0 || config.n_communities == 0 {
        return Err(SynthGraphError::InvalidParameter(
            "n_nodes and n_communities must be > 0".into(),
        ));
    }
    if config.n_communities > config.n_nodes {
        return Err(SynthGraphError::InvalidParameter(
            "n_communities must be ≤ n_nodes".into(),
        ));
    }
    if !(0.0..=1.0).contains(&config.homophily) {
        return Err(SynthGraphError::InvalidParameter(
            "homophily must be in [0, 1]".into(),
        ));
    }
    if !config.avg_degree.is_finite() || config.avg_degree < 0.0 {
        return Err(SynthGraphError::InvalidParameter(
            "avg_degree must be finite and ≥ 0".into(),
        ));
    }
    if config.feat_dim == 0 {
        return Err(SynthGraphError::InvalidParameter(
            "feat_dim must be > 0".into(),
        ));
    }
    if !config.mu.is_finite() || config.mu < 0.0 {
        return Err(SynthGraphError::InvalidParameter(
            "mu must be finite and ≥ 0".into(),
        ));
    }
    let dist_enum = match config.feature_dist {
        "gaussian" => FeatDist::Gaussian,
        "uniform" => FeatDist::Uniform,
        "laplacian" => FeatDist::Laplacian,
        other => {
            return Err(SynthGraphError::InvalidParameter(format!(
                "feature_dist must be 'gaussian', 'uniform', or 'laplacian'; got '{other}'"
            )));
        }
    };
    if let Some(gamma) = config.theta_exponent {
        if !gamma.is_finite() {
            return Err(SynthGraphError::InvalidParameter(
                "theta_exponent must be finite".into(),
            ));
        }
        if gamma <= 2.0 {
            return Err(SynthGraphError::InvalidParameter(
                "theta_exponent (γ) must be > 2 (finite-mean Pareto requirement)".into(),
            ));
        }
    }
    if !(0.0..=1.0).contains(&config.p_triangle) {
        return Err(SynthGraphError::InvalidParameter(
            "p_triangle must be in [0, 1]".into(),
        ));
    }
    if !(0.0..=1.0).contains(&config.feat_noise_ratio) {
        return Err(SynthGraphError::InvalidParameter(
            "feat_noise_ratio must be in [0, 1]".into(),
        ));
    }

    match (config.p_in_override, config.p_out_override) {
        (Some(p_in), Some(p_out)) => {
            if !(0.0..=1.0).contains(&p_in) || !(0.0..=1.0).contains(&p_out) {
                return Err(SynthGraphError::InvalidParameter(
                    "p_in_override and p_out_override must be in [0, 1]".into(),
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(SynthGraphError::InvalidParameter(
                "p_in_override and p_out_override must both be set or both be None".into(),
            ));
        }
    }

    validate_csbm_arrays(config)?;
    Ok(dist_enum)
}

/// Validates the optional matrix inputs against the community/feature dims.
///
/// # Errors
/// Returns [`SynthGraphError::InvalidParameter`] on a shape, symmetry, range,
/// or finiteness violation.
fn validate_csbm_arrays(config: &CsbmConfig<'_>) -> Result<(), SynthGraphError> {
    let n_communities = config.n_communities;
    let feat_dim = config.feat_dim;
    if let Some(bm) = config.b_matrix {
        if bm.len() != n_communities * n_communities {
            return Err(SynthGraphError::InvalidParameter(format!(
                "b_matrix must have length n_communities²={}, got {}",
                n_communities * n_communities,
                bm.len()
            )));
        }
        for &prob in bm {
            if !(0.0..=1.0).contains(&prob) {
                return Err(SynthGraphError::InvalidParameter(
                    "b_matrix values must be in [0, 1]".into(),
                ));
            }
        }
        for row in 0..n_communities {
            for col in 0..n_communities {
                if (bm[row * n_communities + col] - bm[col * n_communities + row]).abs() > 1e-9 {
                    return Err(SynthGraphError::InvalidParameter(format!(
                        "b_matrix must be symmetric: B[{row},{col}]={} != B[{col},{row}]={}",
                        bm[row * n_communities + col],
                        bm[col * n_communities + row]
                    )));
                }
            }
        }
    }

    if let Some(centroids) = config.centroids_override {
        if centroids.len() != n_communities * feat_dim {
            return Err(SynthGraphError::InvalidParameter(format!(
                "centroids must have length n_communities * feat_dim = {}, got {}",
                n_communities * feat_dim,
                centroids.len()
            )));
        }
        if centroids.iter().any(|&value| !value.is_finite()) {
            return Err(SynthGraphError::InvalidParameter(
                "centroids must all be finite".into(),
            ));
        }
    }
    if let Some(cov) = config.covariance {
        if cov.len() != feat_dim * feat_dim {
            return Err(SynthGraphError::InvalidParameter(format!(
                "covariance must have length feat_dim² = {}, got {}",
                feat_dim * feat_dim,
                cov.len()
            )));
        }
        if cov.iter().any(|&v| !v.is_finite()) {
            return Err(SynthGraphError::InvalidParameter(
                "covariance entries must all be finite".into(),
            ));
        }
    }
    Ok(())
}

/// Computes the per-community block sizes (from `class_weights` or an even
/// split), preserving the largest-remainder rounding.
///
/// # Errors
/// Returns [`SynthGraphError::InvalidParameter`] on a bad `class_weights` vector.
fn compute_block_sizes(
    n_nodes: usize,
    n_communities: usize,
    class_weights: Option<&[f64]>,
) -> Result<Vec<usize>, SynthGraphError> {
    // block sizes
    let block_sizes: Vec<usize> = if let Some(weights) = class_weights {
        if weights.len() != n_communities {
            return Err(SynthGraphError::InvalidParameter(format!(
                "class_weights length ({}) must equal n_communities ({})",
                weights.len(),
                n_communities
            )));
        }
        if weights
            .iter()
            .any(|&weight| !weight.is_finite() || weight < 0.0)
        {
            return Err(SynthGraphError::InvalidParameter(
                "class_weights values must all be finite and ≥ 0".into(),
            ));
        }
        let total_weight: f64 = weights.iter().sum();
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(SynthGraphError::InvalidParameter(format!(
                "class_weights must sum to 1.0, got {total_weight:.6}"
            )));
        }
        let exact: Vec<f64> = weights
            .iter()
            .map(|&weight| weight * num_traits::ToPrimitive::to_f64(&(n_nodes)).unwrap_or(0.0))
            .collect();
        let mut sizes: Vec<usize> = exact
            .iter()
            .map(|&exact_size| {
                num_traits::ToPrimitive::to_usize(&(exact_size.floor())).unwrap_or(0)
            })
            .collect();
        let remainder: usize = n_nodes - sizes.iter().sum::<usize>();
        let mut fracs: Vec<(usize, f64)> = exact
            .iter()
            .enumerate()
            .map(|(frac_idx, &exact_size)| (frac_idx, exact_size - exact_size.floor()))
            .collect();
        fracs.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
        for remainder_idx in 0..remainder {
            sizes[fracs[remainder_idx].0] += 1;
        }
        if sizes.contains(&0) {
            return Err(SynthGraphError::InvalidParameter(
                "some blocks have zero nodes; increase n_nodes or adjust class_weights".into(),
            ));
        }
        sizes
    } else {
        let base_size = n_nodes / n_communities;
        let remainder = n_nodes % n_communities;
        (0..n_communities)
            .map(|community_idx| base_size + usize::from(community_idx < remainder))
            .collect()
    };
    Ok(block_sizes)
}

/// Samples the cSBM edge list (block structure + optional triangle closure)
/// and returns it alongside the per-node community labels.
///
/// # Errors
/// Propagates a degree-correction (`compute_dc_theta`) failure.
/// Sampled cSBM graph: the canonical edge list plus per-node community labels.
type SampledGraph = (Vec<(i64, i64)>, Vec<usize>);

/// Closes triangles: for each existing edge (w.p. `p_triangle`) links a random
/// neighbour of one endpoint to the other, adding a third edge.
fn close_triangles(
    edges: &mut Vec<(i64, i64)>,
    adj: &mut [Vec<usize>],
    edge_set: &mut HashSet<(usize, usize)>,
    p_triangle: f64,
    rng: &mut ChaCha8Rng,
) {
    // Snapshot the pre-triangle edges in their canonical emission order, NOT
    // by iterating `edge_set` — a `HashSet` with a per-process-randomized
    // hasher whose iteration order varies run-to-run. Because each visited
    // edge draws from `rng`, iterating the set would make RNG consumption
    // (hence the whole graph) non-deterministic for the same seed. `edges`
    // already holds exactly these pairs (`src < dst`, equal to the `(min,max)`
    // keys in `edge_set`) in a deterministic order, so it is the correct
    // snapshot source. (Audit F-DET1.)
    let snapshot: Vec<(usize, usize)> = edges
        .iter()
        .map(|&(src, dst)| {
            (
                usize::try_from(src).unwrap_or(usize::MAX),
                usize::try_from(dst).unwrap_or(usize::MAX),
            )
        })
        .collect();
    for (endpoint_a, endpoint_b) in snapshot {
        if rng.gen::<f64>() < p_triangle {
            let (hub, other) = if rng.gen::<bool>() {
                (endpoint_a, endpoint_b)
            } else {
                (endpoint_b, endpoint_a)
            };
            let hub_neighbors: Vec<usize> =
                adj[hub].iter().copied().filter(|&w| w != other).collect();
            if hub_neighbors.is_empty() {
                continue;
            }
            let witness = hub_neighbors[rng.gen_range(0..hub_neighbors.len())];
            let key = (other.min(witness), other.max(witness));
            if edge_set.insert(key) {
                edges.push((
                    i64::try_from(key.0).unwrap_or(i64::MAX),
                    i64::try_from(key.1).unwrap_or(i64::MAX),
                ));
                adj[other].push(witness);
                adj[witness].push(other);
            }
        }
    }
}

/// Stitches disconnected components into one graph by bridging each extra
/// component to a random node of the main component.
fn stitch_connectivity(
    edges: &mut Vec<(i64, i64)>,
    adj: &mut [Vec<usize>],
    edge_set: &mut HashSet<(usize, usize)>,
    n_nodes: usize,
    rng: &mut ChaCha8Rng,
) {
    let mut visited = vec![false; n_nodes];
    let mut main_component: Vec<usize> = Vec::new();
    visited[0] = true;
    main_component.push(0);
    let mut head = 0;
    while head < main_component.len() {
        let current = main_component[head];
        head += 1;
        for &neighbor in &adj[current] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                main_component.push(neighbor);
            }
        }
    }
    while let Some(unvisited_start) = visited.iter().position(|&was_visited| !was_visited) {
        let mut component: Vec<usize> = vec![unvisited_start];
        visited[unvisited_start] = true;
        let mut head2 = 0;
        while head2 < component.len() {
            let current = component[head2];
            head2 += 1;
            for &neighbor in &adj[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    component.push(neighbor);
                }
            }
        }
        let bridge_src = component[0];
        let bridge_dst = main_component[rng.gen_range(0..main_component.len())];
        let key = (bridge_src.min(bridge_dst), bridge_src.max(bridge_dst));
        if edge_set.insert(key) {
            edges.push((
                i64::try_from(key.0).unwrap_or(i64::MAX),
                i64::try_from(key.1).unwrap_or(i64::MAX),
            ));
            adj[bridge_src].push(bridge_dst);
            adj[bridge_dst].push(bridge_src);
        }
        main_component.extend_from_slice(&component);
    }
}

fn sample_csbm_edges(
    config: &CsbmConfig<'_>,
    block_sizes: &[usize],
    rng: &mut ChaCha8Rng,
) -> Result<SampledGraph, SynthGraphError> {
    let n_nodes = config.n_nodes;
    let n_communities = config.n_communities;
    let mut labels = vec![0usize; n_nodes];
    {
        let mut offset = 0usize;
        for (community_idx, &community_size) in block_sizes.iter().enumerate() {
            for label in &mut labels[offset..offset + community_size] {
                *label = community_idx;
            }
            offset += community_size;
        }
    }

    let mut block_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_communities];
    for (node_idx, &community) in labels.iter().enumerate() {
        block_nodes[community].push(node_idx);
    }

    let (p_in, p_out) =
        if let (Some(p_in), Some(p_out)) = (config.p_in_override, config.p_out_override) {
            (p_in, p_out)
        } else {
            // Generalised formula supporting unequal block sizes.
            // p_in  = homophily·avg_degree·n / Σ_k s_k(s_k−1)
            // p_out = (1−homophily)·avg_degree·n / (n² − Σ_k s_k²)
            let sum_sk_sk1: f64 = block_sizes
                .iter()
                .map(|&block_size| {
                    num_traits::ToPrimitive::to_f64(&(block_size)).unwrap_or(0.0)
                        * (num_traits::ToPrimitive::to_f64(&(block_size)).unwrap_or(0.0) - 1.0)
                })
                .sum();
            let sum_sk2: f64 = block_sizes
                .iter()
                .map(|&block_size| {
                    num_traits::ToPrimitive::to_f64(&(block_size)).unwrap_or(0.0)
                        * num_traits::ToPrimitive::to_f64(&(block_size)).unwrap_or(0.0)
                })
                .sum();
            let n_float = num_traits::ToPrimitive::to_f64(&(n_nodes)).unwrap_or(0.0);
            // Split into two rounded ops (not `mul_add`) to preserve the exact IEEE
            // result the determinism goldens were blessed against.
            let n_squared = n_float * n_float;
            let cross_pairs = n_squared - sum_sk2; // = 2 · Σ_{a<b} s_a·s_b
            let p_in = if sum_sk_sk1 > 0.0 {
                (config.homophily * config.avg_degree * n_float / sum_sk_sk1).min(1.0)
            } else {
                0.0
            };
            let p_out = if cross_pairs > 0.0 {
                ((1.0 - config.homophily) * config.avg_degree * n_float / cross_pairs).min(1.0)
            } else {
                0.0
            };
            (p_in, p_out)
        };

    let edge_probs: Vec<f64> = config.b_matrix.map_or_else(
        || {
            let mut ep = vec![0.0f64; n_communities * n_communities];
            for row in 0..n_communities {
                for col in 0..n_communities {
                    ep[row * n_communities + col] = if row == col { p_in } else { p_out };
                }
            }
            ep
        },
        <[f64]>::to_vec,
    );

    let correction = config
        .theta_exponent
        .map_or(DegreeCorrection::Off, |exponent| DegreeCorrection::Drawn {
            exponent,
        });
    let use_dc = correction.is_corrected();
    let (theta, block_max_theta) =
        compute_dc_theta(correction, n_nodes, &labels, &block_nodes, rng)?;

    let need_adj = config.p_triangle > 0.0 || config.ensure_connected;

    let mut edges: Vec<(i64, i64)> = Vec::new();
    sample_all_blocks(
        &block_nodes,
        &theta,
        &block_max_theta,
        use_dc,
        |block_a, block_b| edge_probs[block_a * n_communities + block_b],
        rng,
        &mut edges,
    );

    // adjacency list (needed for triangle closing / connectivity stitching)
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); if need_adj { n_nodes } else { 0 }];
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    if need_adj {
        for &(src, dst) in &edges {
            let src_idx = usize::try_from(src).unwrap_or(usize::MAX);
            let dst_idx = usize::try_from(dst).unwrap_or(usize::MAX);
            let key = (src_idx.min(dst_idx), src_idx.max(dst_idx));
            if edge_set.insert(key) {
                adj[src_idx].push(dst_idx);
                adj[dst_idx].push(src_idx);
            }
        }
    }

    // triangle closing (randomly pick which endpoint is the hub)
    if config.p_triangle > 0.0 {
        close_triangles(&mut edges, &mut adj, &mut edge_set, config.p_triangle, rng);
    }

    // connectivity stitching
    if config.ensure_connected && n_nodes > 0 {
        stitch_connectivity(&mut edges, &mut adj, &mut edge_set, n_nodes, rng);
    }
    Ok((edges, labels))
}

/// Generates a cSBM graph with node features; returns a pure-Rust result with no Python types.
///
/// # Errors
/// Returns [`SynthGraphError`] if the parameters are invalid (e.g. mismatched
/// dimensions) or the supplied covariance is not positive-definite (Cholesky fails).
pub fn generate_csbm(config: &CsbmConfig<'_>) -> Result<CsbmCoreResult, SynthGraphError> {
    let dist_enum = validate_csbm_params(config)?;
    let block_sizes =
        compute_block_sizes(config.n_nodes, config.n_communities, config.class_weights)?;
    let mut rng = seeded_rng(config.seed);
    let (edges, labels) = sample_csbm_edges(config, &block_sizes, &mut rng)?;
    let effective_h = edge_homophily(&edges, &labels)?;
    let x_flat = generate_node_features(config, &dist_enum, &labels, &mut rng)?;

    Ok(CsbmCoreResult {
        canonical_edges: edges,
        labels,
        x_flat,
        effective_h,
        n_nodes: config.n_nodes,
        feat_dim: config.feat_dim,
    })
}
