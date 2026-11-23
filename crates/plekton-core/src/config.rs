// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Config-driven JSON entry points (`generate_from_config_native`,
//! `validate_config_warn`).

use serde::Deserialize;

use crate::{
    ba::generate_ba_core,
    csbm::{generate_csbm, CsbmConfig},
    output::{csbm_result_to_graph_output, sbm_result_to_graph_output},
    sampling::DegreeCorrection,
    sbm::{generate_sbm_from_fit, BlockAssignment, BlockConnection, SbmFit},
    SynthGraphError,
};

// Config-driven API
//
// Single Rust entry point: `generate_from_config_native`. Parses the
// canonical graph-JSON contract and returns a pretty-printed JSON string
// on success or a `SynthGraphError` on failure. The Python-facing wrapper
// lives in the `plekton-python` crate.

/// Config-driven generation request parsed from the TUI JSON input.
#[derive(Deserialize)]
pub struct GenerationConfig {
    /// Backbone generator to invoke: `"sbm_classique"`, `"sbm"`, `"csbm"`, or `"ba"`.
    pub model_type: String,
    /// Optional fixed PRNG seed; omit for entropy seeding.
    pub seed: Option<u64>,
    /// Generator-specific parameters; see [`ConfigParameters`].
    pub parameters: ConfigParameters,
}

/// Neutral homophily when a model does not use it (see [`ConfigParameters::homophily`]).
/// 0.5 is the no-community-preference point, so an accidental omission on a
/// block model degrades to "no structure" rather than to a silent strong prior.
const fn default_homophily() -> f64 {
    0.5
}

/// Neutral average degree when a model does not use it. Chosen above the
/// validator's `< 1.0` "likely disconnected" warning so an omission on a block
/// model is not additionally flagged for a value the user never supplied.
const fn default_avg_degree() -> f64 {
    1.0
}

/// Parameters block within a [`GenerationConfig`].
#[derive(Deserialize)]
pub struct ConfigParameters {
    /// Target number of nodes in the generated graph.
    pub n_nodes: usize,
    /// Number of communities / blocks.
    pub n_communities: usize,
    /// Requested homophily (fraction of edges within blocks); in `[0, 1]`.
    ///
    /// Optional because Barabási–Albert has no block structure to be homophilous
    /// about — its edges follow preferential attachment. The validator already
    /// skips this field for `ba` (`if !is_ba`), so requiring it at parse time only
    /// forced BA users to invent a value their model ignores.
    #[serde(default = "default_homophily")]
    pub homophily: f64,
    /// Target average degree across the whole graph.
    ///
    /// Optional for the same reason: BA's degree distribution is set by `m`, not
    /// by a target average. Skipped by the validator for `ba`.
    #[serde(default = "default_avg_degree")]
    pub avg_degree: f64,
    /// cSBM feature dimension; falls back to 16 if omitted.
    #[serde(default)]
    pub features_dim: Option<usize>,
    /// cSBM centroid scale `μ`; controls how easy class separation is.
    #[serde(default)]
    pub mu: Option<f64>,
    /// DC-SBM Pareto exponent `γ` for degree heterogeneity; must be > 2.
    #[serde(default)]
    pub theta_exponent: Option<f64>,
    /// cSBM per-node label-noise probability; in `[0, 1]`.
    #[serde(default)]
    pub feat_noise_ratio: Option<f64>,
    /// cSBM feature noise distribution: `"gaussian"`, `"uniform"`, or `"laplacian"`.
    #[serde(default)]
    pub feature_dist: Option<String>,
    /// When `true`, the cSBM generator stitches components until the graph
    /// becomes connected.
    #[serde(default)]
    pub ensure_connected: bool,
    /// Barabási–Albert — edges added per new node (≥ 1). Standard literature name.
    #[serde(default)]
    pub m: Option<usize>,
    /// Barabási–Albert — initial clique size (≥ `m`, < `n_nodes`). Standard literature name.
    #[serde(default)]
    pub m0: Option<usize>,
    /// A fitted K×K mixing matrix, row-major, symmetric, entries in `[0, 1]`.
    ///
    /// Supplying it overrides `homophily`/`avg_degree` on the SBM path entirely.
    /// Those two scalars cannot express a `B` in which one block talks to exactly
    /// one other, which is what an estimator run on a real network produces.
    #[serde(default)]
    pub b_matrix: Option<Vec<f64>>,
    /// Fitted block sizes, in block order, summing to `n_nodes`.
    ///
    /// Without this a fit is applied to near-equal blocks, so a measured `B` and
    /// a measured θ are realised on a partition that is not the one they were
    /// measured on — which reproduces neither the density nor the degrees.
    #[serde(default)]
    pub block_sizes: Option<Vec<usize>>,
    /// A fitted per-node θ, one entry per node in node order.
    ///
    /// Mutually exclusive with `theta_exponent`: that one asks the generator to
    /// *draw* a degree correction, this one supplies a measured one, and a config
    /// carrying both is refused rather than silently resolved.
    #[serde(default)]
    pub theta: Option<Vec<f64>>,
}

/// Pure-Rust config-driven entry point.
///
/// Accepts a JSON configuration (schema documented below) and returns the
/// graph JSON string on success, or a [`SynthGraphError`] on failure.
///
/// No `PyO3` types are involved — this function is safe to call from any Rust
/// binary without a Python runtime.
///
/// # JSON input format
/// ```json
/// {
///   "model_type": "sbm_classique" | "sbm" | "csbm",
///   "seed": 42,
///   "parameters": {
///     "n_nodes": 500, "n_communities": 5,
///     "homophily": 0.8, "avg_degree": 10.0,
///     "features_dim": 16, "mu": 1.0,
///     "theta_exponent": null, "feat_noise_ratio": 0.0
///   }
/// }
/// ```
/// Warns about the shape of a supplied fit — a mixing matrix, block sizes, a
/// per-node θ — without deciding anything.
///
/// Split out of [`validate_config_warn`] to keep that function under the
/// complexity gate, and kept in warning form deliberately: the generator refuses
/// these for real. A validator that only warns and a generator that only refuses
/// would disagree, so both check and only one of them stops.
fn warn_about_supplied_fit(params: &ConfigParameters, warnings: &mut Vec<String>) {
    if params.theta.is_some() && params.theta_exponent.is_some() {
        warnings.push(
            "both theta and theta_exponent are set — one supplies a measured degree \
             correction and the other asks for a drawn one; generation will refuse"
                .to_owned(),
        );
    }
    if let Some(theta) = params.theta.as_ref() {
        if theta.len() != params.n_nodes {
            warnings.push(format!(
                "theta has {} entries but n_nodes is {} — one weight per node is required",
                theta.len(),
                params.n_nodes
            ));
        }
        if theta
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            warnings.push(
                "theta holds a non-finite or negative weight; a degree correction is a \
                 non-negative multiplier"
                    .to_owned(),
            );
        }
    }
    if let Some(sizes) = params.block_sizes.as_ref() {
        let total: usize = sizes.iter().sum();
        if total != params.n_nodes {
            warnings.push(format!(
                "block_sizes sum to {total} but n_nodes is {}",
                params.n_nodes
            ));
        }
        if sizes.len() != params.n_communities {
            warnings.push(format!(
                "block_sizes names {} blocks but n_communities is {}",
                sizes.len(),
                params.n_communities
            ));
        }
        if sizes.contains(&0) {
            warnings.push(
                "block_sizes contains an empty block; it has no theta to normalise and no \
                 density to reproduce"
                    .to_owned(),
            );
        }
    }
    if let Some(matrix) = params.b_matrix.as_ref() {
        let side = params.n_communities;
        if matrix.len() == side * side {
            for row in 0..side {
                for col in (row + 1)..side {
                    let (upper, lower) = (matrix[row * side + col], matrix[col * side + row]);
                    if (upper - lower).abs() > 1e-9 {
                        warnings.push(format!(
                            "b_matrix is not symmetric at [{row},{col}]: \
                             {upper} != {lower}"
                        ));
                    }
                }
            }
        } else {
            warnings.push(format!(
                "b_matrix has {} entries but n_communities² is {}",
                matrix.len(),
                side * side
            ));
        }
        if matrix
            .iter()
            .any(|prob| !prob.is_finite() || !(0.0..=1.0).contains(prob))
        {
            warnings.push("b_matrix holds a value outside [0, 1] or a non-finite one".to_owned());
        }
    }
}

/// Validates a generation config JSON and returns a list of human-readable warnings.
///
/// Never fails or panics. If the JSON cannot be parsed, returns a single warning.
/// Call this before `generate_from_config_native` to surface potential issues early.
#[must_use]
pub fn validate_config_warn(config_json: &str) -> Vec<String> {
    let config: GenerationConfig = match serde_json::from_str(config_json) {
        Ok(parsed) => parsed,
        Err(err) => return vec![format!("Could not pre-validate config: {err}")],
    };
    let params = &config.parameters;
    let model = config.model_type.as_str();
    let is_ba = model == "ba";
    let mut warnings = Vec::new();

    // ── Graph-size checks (always) ────────────────────────────────────────────
    if params.n_nodes < 10 {
        warnings.push(format!(
            "very small graph ({} nodes) — results unreliable; use n_nodes ≥ 10",
            params.n_nodes
        ));
    }

    if params.n_communities > params.n_nodes / 4 {
        warnings.push(format!(
            "blocks too small ({} communities, {} nodes) — community statistics unreliable",
            params.n_communities, params.n_nodes
        ));
    }

    if !is_ba {
        if params.avg_degree < 1.0 {
            warnings.push(format!(
                "avg_degree ({}) too low — graph likely disconnected",
                params.avg_degree
            ));
        }

        if (params.homophily - 0.5).abs() < 0.05 {
            warnings.push(format!(
                "homophily ({:.2}) near 0.5 — community structure very weak",
                params.homophily
            ));
        }

        let block_size = num_traits::ToPrimitive::to_f64(&(params.n_nodes)).unwrap_or(0.0)
            / num_traits::ToPrimitive::to_f64(&(params.n_communities.max(1))).unwrap_or(0.0);
        let p_in = if block_size > 1.0 {
            (params.homophily * params.avg_degree / (block_size - 1.0)).min(1.0)
        } else {
            0.0
        };
        if p_in > 0.5 {
            warnings.push(format!("dense intra-block connections (p_in ≈ {p_in:.2})"));
        }

        if model == "csbm" {
            let feat_dim = params.features_dim.unwrap_or(16);
            let mu = params.mu.unwrap_or(1.0);

            if mu < 0.5 && feat_dim < 8 {
                warnings.push(format!(
                    "cSBM features essentially noise (mu={mu:.2}, feat_dim={feat_dim})"
                ));
            }

            // High mu: classification becomes trivially easy — not a useful benchmark.
            if mu > 10.0 {
                warnings.push(format!(
                    "cSBM mu ({mu:.1}) very large — class separation trivial; \
                     not a challenging benchmark"
                ));
            }

            // Feature dimension exceeds node count: GNN will overfit immediately.
            if feat_dim > params.n_nodes {
                warnings.push(format!(
                    "feat_dim ({feat_dim}) exceeds n_nodes ({}) — \
                     features overdetermined; GNN will overfit trivially",
                    params.n_nodes
                ));
            }
        }

        warn_about_supplied_fit(params, &mut warnings);

        // theta_exponent near the phase transition [2.0, 2.1]: degree distribution unstable.
        if let Some(theta) = params.theta_exponent {
            if (2.0..=2.1).contains(&theta) {
                warnings.push(format!(
                    "theta_exponent ({theta:.3}) is near the DC-SBM phase transition \
                     [2.0, 2.1] — degree distribution may be unstable"
                ));
            }
        }
    }

    warnings
}

/// Parses the canonical config JSON contract,
/// dispatches to the right backbone generator, computes graph metrics, and
/// returns the pretty-printed JSON graph as a `String`.
///
/// The output JSON conforms to [`crate::GraphOutput`]; the input JSON must match
/// [`GenerationConfig`].
///
/// # Errors
/// Propagates [`SynthGraphError::InvalidParameter`] for malformed JSON or an
/// unknown `model_type`, and any error raised by the underlying generator.
pub fn generate_from_config_native(config_json: &str) -> Result<String, SynthGraphError> {
    let config: GenerationConfig = serde_json::from_str(config_json).map_err(|parse_err| {
        SynthGraphError::InvalidParameter(format!("Config parse error: {parse_err}"))
    })?;

    let params = &config.parameters;
    match config.model_type.as_str() {
        "sbm_classique" | "sbm" => {
            let block_size = num_traits::ToPrimitive::to_f64(&(params.n_nodes)).unwrap_or(0.0)
                / num_traits::ToPrimitive::to_f64(&(params.n_communities)).unwrap_or(0.0);
            let p_in = if block_size > 1.0 {
                (params.homophily * params.avg_degree / (block_size - 1.0)).min(1.0)
            } else {
                0.0
            };
            let p_out = if params.n_communities > 1 {
                ((1.0 - params.homophily) * params.avg_degree
                    / (num_traits::ToPrimitive::to_f64(&(params.n_communities - 1)).unwrap_or(0.0)
                        * block_size))
                    .min(1.0)
            } else {
                0.0
            };
            // A supplied theta and a drawn one are different requests, so a config
            // carrying both is refused here rather than resolved by branch order.
            if params.theta.is_some() && params.theta_exponent.is_some() {
                return Err(SynthGraphError::InvalidParameter(
                    "theta and theta_exponent are mutually exclusive: one supplies a measured \
                     degree correction, the other asks the generator to draw one"
                        .into(),
                ));
            }
            let degree_correction = match (params.theta.as_ref(), params.theta_exponent) {
                (Some(theta), _) => DegreeCorrection::Supplied(theta),
                (None, Some(exponent)) => DegreeCorrection::Drawn { exponent },
                (None, None) => DegreeCorrection::Off,
            };
            let fit = SbmFit {
                assignment: params.block_sizes.as_ref().map_or(
                    BlockAssignment::Balanced {
                        n_communities: params.n_communities,
                    },
                    |sizes| BlockAssignment::Sizes(sizes),
                ),
                connection: params
                    .b_matrix
                    .as_ref()
                    .map_or(BlockConnection::Uniform { p_in, p_out }, |matrix| {
                        BlockConnection::Matrix(matrix)
                    }),
                degree_correction,
            };
            let result = generate_sbm_from_fit(params.n_nodes, &fit, config.seed)?;
            let graph = sbm_result_to_graph_output(&result);
            Ok(serde_json::to_string_pretty(&graph)?)
        }
        "csbm" => {
            let result = generate_csbm(&CsbmConfig {
                n_nodes: params.n_nodes,
                n_communities: params.n_communities,
                homophily: params.homophily,
                avg_degree: params.avg_degree,
                feat_dim: params.features_dim.unwrap_or(16),
                mu: params.mu.unwrap_or(1.0),
                feature_dist: params.feature_dist.as_deref().unwrap_or("gaussian"),
                theta_exponent: params.theta_exponent,
                feat_noise_ratio: params.feat_noise_ratio.unwrap_or(0.0),
                ensure_connected: params.ensure_connected,
                seed: config.seed,
                ..CsbmConfig::default()
            })?;
            let graph = csbm_result_to_graph_output(&result);
            Ok(serde_json::to_string_pretty(&graph)?)
        }
        "ba" => {
            let m = params.m.unwrap_or(2);
            let m0 = params.m0.unwrap_or(m);
            let result =
                generate_ba_core(params.n_nodes, m, m0, params.n_communities, config.seed)?;
            let graph = sbm_result_to_graph_output(&result);
            Ok(serde_json::to_string_pretty(&graph)?)
        }
        other => Err(SynthGraphError::InvalidParameter(format!(
            "Unknown model_type: '{other}'. Use 'sbm_classique', 'sbm', 'csbm', or 'ba'"
        ))),
    }
}
