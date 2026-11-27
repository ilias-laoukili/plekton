// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
// Integration-test scaffolding: production-grade lints (panic-on-setup, numeric
// casts, exact float compares, literal grouping) are noise in tests, not signal.
//! Public-API behavior tests for the generation engine (relocated from the
//! in-`lib.rs` test module during the module split). Everything here exercises
//! only the crate's public surface; the unit tests that touch private items
//! (`decode_upper_tri`, `compute_metrics`) stay in `sampling.rs` / `output.rs`.

use plekton::*;

#[test]
fn sbm_rejects_more_communities_than_nodes() -> Result<(), Box<dyn std::error::Error>> {
    let err = generate_sbm_core(5, 10, 0.5, 0.1, None, Some(42))
        .err()
        .ok_or("expected an error")?;
    assert!(
        err.to_string().contains("n_communities"),
        "expected 'n_communities' in error: {err}"
    );
    Ok(())
}

#[test]
fn sbm_allows_equal_communities_and_nodes() -> Result<(), Box<dyn std::error::Error>> {
    generate_sbm_core(5, 5, 0.5, 0.1, None, Some(42))?;
    Ok(())
}

#[test]
fn sbm_dc_rejects_more_communities_than_nodes() {
    assert!(generate_sbm_core(5, 10, 0.5, 0.1, Some(3.0), Some(42)).is_err());
}

#[test]
fn sbm_dc_empty_block_guard_fires() {
    assert!(generate_sbm_core(2, 3, 0.5, 0.1, Some(3.0), Some(1)).is_err());
}

#[test]
fn sbm_dc_equal_communities_and_nodes() -> Result<(), Box<dyn std::error::Error>> {
    generate_sbm_core(4, 4, 0.6, 0.1, Some(3.0), Some(7))?;
    Ok(())
}

/// Test helper — builds a [`NodeJson`] inline so the feature-validation
/// tests can keep their fixtures terse.
const fn make_node(id: usize, community: usize, features: Option<Vec<f32>>) -> NodeJson {
    NodeJson {
        id,
        community,
        features,
    }
}

#[test]
fn features_first_present_second_absent_errors() {
    let nodes = vec![make_node(0, 0, Some(vec![1.0, 2.0])), make_node(1, 1, None)];
    assert!(collect_node_features(&nodes).is_err());
}

#[test]
fn features_first_absent_second_present_errors() {
    let nodes = vec![make_node(0, 0, None), make_node(1, 1, Some(vec![1.0, 2.0]))];
    assert!(collect_node_features(&nodes).is_err());
}

#[test]
fn features_inconsistent_dim_errors() {
    let nodes = vec![
        make_node(0, 0, Some(vec![1.0, 2.0])),
        make_node(1, 1, Some(vec![3.0])),
    ];
    assert!(collect_node_features(&nodes).is_err());
}

#[test]
fn features_all_absent_returns_none() -> Result<(), Box<dyn std::error::Error>> {
    let nodes = vec![make_node(0, 0, None), make_node(1, 1, None)];
    assert_eq!(collect_node_features(&nodes)?, None);
    Ok(())
}

#[test]
fn features_empty_graph_returns_none() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(collect_node_features(&[])?, None);
    Ok(())
}

#[test]
fn features_consistent_returns_flat_buffer() -> Result<(), Box<dyn std::error::Error>> {
    let nodes = vec![
        make_node(0, 0, Some(vec![1.0, 2.0])),
        make_node(1, 1, Some(vec![3.0, 4.0])),
    ];
    assert_eq!(
        collect_node_features(&nodes)?,
        Some((2, vec![1.0, 2.0, 3.0, 4.0]))
    );
    Ok(())
}

#[test]
fn effective_h_in_unit_interval() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_csbm(&CsbmConfig {
        n_nodes: 100,
        n_communities: 4,
        homophily: 0.7,
        avg_degree: 5.0,
        feat_dim: 8,
        mu: 1.0,
        seed: Some(42),
        ..CsbmConfig::default()
    })?;
    let effective_h = result
        .effective_h
        .ok_or("a graph with edges has a homophily")?;
    assert!(
        (0.0..=1.0).contains(&effective_h),
        "effective_h={effective_h} not in [0,1]"
    );
    Ok(())
}

#[test]
fn fully_intra_graph_has_effective_h_one() -> Result<(), Box<dyn std::error::Error>> {
    // p_out=0 → all edges are intra-block → effective_h must be 1.0
    let result = generate_sbm_core(20, 4, 0.5, 0.0, None, Some(5))?;
    if !result.canonical_edges.is_empty() {
        assert_eq!(
            result.effective_h.map(f64::to_bits),
            Some(1.0_f64.to_bits())
        );
    }
    Ok(())
}

#[test]
fn an_edgeless_graph_has_no_homophily() -> Result<(), Box<dyn std::error::Error>> {
    // p_in=p_out=0 → no edges → no fraction to measure: None, never a made-up 0.0,
    // and the JSON says why.
    let result = generate_sbm_core(10, 2, 0.0, 0.0, None, Some(0))?;
    assert_eq!(result.canonical_edges, [] as [(i64, i64); 0]);
    assert_eq!(result.effective_h, None);

    let graph = sbm_result_to_graph_output(&result);
    assert_eq!(graph.metadata.homophily, None);
    let reasons: Vec<(&str, &str)> = graph
        .metadata
        .metrics_skipped
        .iter()
        .map(|skipped| (skipped.metric.as_str(), skipped.reason.as_str()))
        .collect();
    assert!(
        reasons.contains(&("homophily", "undefined here: the graph has no edges")),
        "{reasons:?}"
    );
    Ok(())
}

#[test]
fn a_single_community_has_homophily_one_everywhere() -> Result<(), Box<dyn std::error::Error>> {
    // One block: every edge is intra-community. The generator, its JSON and the
    // metric API agree on 1.0 — the metric no longer reads "all labels 0" as
    // "unlabelled".
    let result = generate_sbm_core(20, 1, 0.3, 0.0, None, Some(3))?;
    assert_ne!(result.canonical_edges, [] as [(i64, i64); 0]);
    assert_eq!(
        result.effective_h.map(f64::to_bits),
        Some(1.0_f64.to_bits())
    );

    let graph = sbm_result_to_graph_output(&result);
    assert_eq!(
        graph.metadata.homophily.map(f64::to_bits),
        Some(1.0_f64.to_bits())
    );

    let edges: Vec<(usize, usize)> = result
        .canonical_edges
        .iter()
        .map(|&(u, v)| Ok((usize::try_from(u)?, usize::try_from(v)?)))
        .collect::<Result<_, std::num::TryFromIntError>>()?;
    assert_eq!(
        plekton_metrics::homophily::compute_homophily(&edges, &result.labels).map(f64::to_bits),
        Some(1.0_f64.to_bits())
    );
    Ok(())
}

#[test]
fn p_triangle_zero_is_deterministic() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = CsbmConfig {
        n_nodes: 20,
        n_communities: 2,
        homophily: 0.8,
        avg_degree: 5.0,
        feat_dim: 4,
        mu: 1.0,
        seed: Some(99),
        ..CsbmConfig::default()
    };
    let r1 = generate_csbm(&cfg)?;
    let r2 = generate_csbm(&cfg)?;
    assert_eq!(r1.canonical_edges.len(), r2.canonical_edges.len());
    Ok(())
}

#[test]
fn p_triangle_one_never_removes_edges() -> Result<(), Box<dyn std::error::Error>> {
    let base_cfg = CsbmConfig {
        n_nodes: 30,
        n_communities: 2,
        homophily: 0.5,
        avg_degree: 4.0,
        feat_dim: 4,
        mu: 1.0,
        seed: Some(7),
        ..CsbmConfig::default()
    };
    let base = generate_csbm(&base_cfg)?;
    let with_triangles = generate_csbm(&CsbmConfig {
        p_triangle: 1.0,
        ..base_cfg
    })?;
    assert!(with_triangles.canonical_edges.len() >= base.canonical_edges.len());
    Ok(())
}

#[test]
fn ensure_connected_produces_one_component() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_csbm(&CsbmConfig {
        n_nodes: 50,
        n_communities: 5,
        homophily: 0.5,
        avg_degree: 1.0,
        feat_dim: 4,
        mu: 1.0,
        ensure_connected: true,
        seed: Some(11),
        ..CsbmConfig::default()
    })?;
    let n_nodes = result.n_nodes;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(src, dst) in &result.canonical_edges {
        adj[usize::try_from(src).unwrap_or(usize::MAX)]
            .push(usize::try_from(dst).unwrap_or(usize::MAX));
        adj[usize::try_from(dst).unwrap_or(usize::MAX)]
            .push(usize::try_from(src).unwrap_or(usize::MAX));
    }
    let mut visited = vec![false; n_nodes];
    visited[0] = true;
    let mut queue = vec![0usize];
    let mut head = 0;
    while head < queue.len() {
        for &neighbor in &adj[queue[head]] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
        head += 1;
    }
    assert!(
        visited.iter().all(|&was_visited| was_visited),
        "graph must be connected"
    );
    Ok(())
}

#[test]
fn ensure_connected_false_does_not_panic() -> Result<(), Box<dyn std::error::Error>> {
    generate_csbm(&CsbmConfig {
        n_nodes: 100,
        n_communities: 10,
        homophily: 0.5,
        avg_degree: 0.001,
        feat_dim: 4,
        mu: 1.0,
        seed: Some(1),
        ..CsbmConfig::default()
    })?;
    Ok(())
}

#[test]
fn gaussian_features_shape_and_finite() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_csbm(&CsbmConfig {
        n_nodes: 10,
        n_communities: 2,
        homophily: 0.7,
        avg_degree: 3.0,
        feat_dim: 8,
        mu: 1.0,
        feature_dist: "gaussian",
        seed: Some(1),
        ..CsbmConfig::default()
    })?;
    assert_eq!(result.x_flat.len(), result.n_nodes * result.feat_dim);
    assert!(result.x_flat.iter().all(|feat| feat.is_finite()));
    Ok(())
}

#[test]
fn uniform_features_shape_and_finite() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_csbm(&CsbmConfig {
        n_nodes: 10,
        n_communities: 2,
        homophily: 0.7,
        avg_degree: 3.0,
        feat_dim: 4,
        mu: 1.0,
        feature_dist: "uniform",
        seed: Some(2),
        ..CsbmConfig::default()
    })?;
    assert_eq!(result.x_flat.len(), result.n_nodes * result.feat_dim);
    assert!(result.x_flat.iter().all(|feat| feat.is_finite()));
    Ok(())
}

#[test]
fn laplacian_features_shape_and_finite() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_csbm(&CsbmConfig {
        n_nodes: 10,
        n_communities: 2,
        homophily: 0.7,
        avg_degree: 3.0,
        feat_dim: 4,
        mu: 1.0,
        feature_dist: "laplacian",
        seed: Some(3),
        ..CsbmConfig::default()
    })?;
    assert_eq!(result.x_flat.len(), result.n_nodes * result.feat_dim);
    assert!(result.x_flat.iter().all(|feat| feat.is_finite()));
    Ok(())
}

#[test]
fn sbm_json_round_trip_preserves_structure() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_sbm_core(20, 3, 0.5, 0.1, None, Some(77))?;
    let graph_out = sbm_result_to_graph_output(&result);
    let json = serde_json::to_string(&graph_out)?;
    let parsed: GraphOutput = serde_json::from_str(&json)?;
    assert_eq!(parsed.nodes.len(), result.n_nodes);
    assert_eq!(parsed.edges.len(), result.canonical_edges.len());
    let orig: std::collections::HashSet<(usize, usize)> = result
        .canonical_edges
        .iter()
        .map(|&(src, dst)| {
            (
                usize::try_from(src).unwrap_or(usize::MAX),
                usize::try_from(dst).unwrap_or(usize::MAX),
            )
        })
        .collect();
    let parsed_set: std::collections::HashSet<(usize, usize)> = parsed
        .edges
        .iter()
        .map(|edge| (edge.source, edge.target))
        .collect();
    assert_eq!(orig, parsed_set);
    Ok(())
}

#[test]
fn csbm_json_round_trip_preserves_features() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_csbm(&CsbmConfig {
        n_nodes: 8,
        n_communities: 2,
        homophily: 0.7,
        avg_degree: 3.0,
        feat_dim: 4,
        mu: 1.0,
        seed: Some(5),
        ..CsbmConfig::default()
    })?;
    let graph_out = csbm_result_to_graph_output(&result);
    let json = serde_json::to_string(&graph_out)?;
    let parsed: GraphOutput = serde_json::from_str(&json)?;
    for (node_idx, node) in parsed.nodes.iter().enumerate() {
        let feats = node.features.as_ref().ok_or("expected a value")?;
        assert_eq!(feats.len(), result.feat_dim);
        for (feat_idx, &orig) in result.x_flat
            [node_idx * result.feat_dim..(node_idx + 1) * result.feat_dim]
            .iter()
            .enumerate()
        {
            assert!(
                (feats[feat_idx] - orig).abs() < 1e-6,
                "feature [{node_idx},{feat_idx}]: got {}, expected {orig}",
                feats[feat_idx]
            );
        }
    }
    Ok(())
}

#[test]
fn csbm_rejects_nan_d() {
    assert!(generate_csbm(&CsbmConfig {
        n_nodes: 10,
        avg_degree: f64::NAN,
        feat_dim: 4,
        seed: Some(1),
        ..CsbmConfig::default()
    })
    .is_err());
}

#[test]
fn csbm_rejects_inf_d() {
    assert!(generate_csbm(&CsbmConfig {
        n_nodes: 10,
        avg_degree: f64::INFINITY,
        feat_dim: 4,
        seed: Some(1),
        ..CsbmConfig::default()
    })
    .is_err());
}

#[test]
fn csbm_rejects_nan_mu() {
    assert!(generate_csbm(&CsbmConfig {
        n_nodes: 10,
        avg_degree: 3.0,
        feat_dim: 4,
        mu: f64::NAN,
        seed: Some(1),
        ..CsbmConfig::default()
    })
    .is_err());
}

#[test]
fn csbm_rejects_nan_class_weight() {
    assert!(generate_csbm(&CsbmConfig {
        n_nodes: 10,
        avg_degree: 3.0,
        feat_dim: 4,
        class_weights: Some(&[f64::NAN, 1.0]),
        seed: Some(1),
        ..CsbmConfig::default()
    })
    .is_err());
}

#[test]
fn csbm_rejects_inf_class_weight() {
    assert!(generate_csbm(&CsbmConfig {
        n_nodes: 10,
        avg_degree: 3.0,
        feat_dim: 4,
        class_weights: Some(&[f64::INFINITY, 0.0]),
        seed: Some(1),
        ..CsbmConfig::default()
    })
    .is_err());
}

#[test]
fn sbm_rejects_nan_theta_exponent() {
    assert!(generate_sbm_core(10, 2, 0.5, 0.1, Some(f64::NAN), Some(1)).is_err());
}

#[test]
fn csbm_rejects_nan_theta_exponent() {
    assert!(generate_csbm(&CsbmConfig {
        n_nodes: 10,
        avg_degree: 3.0,
        feat_dim: 4,
        theta_exponent: Some(f64::NAN),
        seed: Some(1),
        ..CsbmConfig::default()
    })
    .is_err());
}

// generate_ba_core tests

#[test]
fn ba_basic_growth() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_ba_core(100, 2, 2, 5, Some(42))?;
    assert_eq!(result.n_nodes, 100);
    // Initial clique (m0=2): 1 edge. Then 98 new nodes × 2 edges each = 197 total.
    assert_eq!(result.canonical_edges.len(), 197);
    for &(src, dst) in &result.canonical_edges {
        assert!(src >= 0 && dst >= 0);
        assert!(
            usize::try_from(src).unwrap_or(usize::MAX) < 100
                && usize::try_from(dst).unwrap_or(usize::MAX) < 100
        );
        assert!(src < dst, "edges must be canonical (src < dst)");
    }
    Ok(())
}

#[test]
fn ba_reproducible() -> Result<(), Box<dyn std::error::Error>> {
    let r1 = generate_ba_core(50, 2, 2, 3, Some(7))?;
    let r2 = generate_ba_core(50, 2, 2, 3, Some(7))?;
    assert_eq!(r1.canonical_edges, r2.canonical_edges);
    assert_eq!(r1.labels, r2.labels);
    Ok(())
}

#[test]
fn ba_rejects_m_zero() {
    assert!(generate_ba_core(10, 0, 0, 2, None).is_err());
}

#[test]
fn ba_rejects_m0_less_than_m() {
    assert!(generate_ba_core(10, 3, 2, 2, None).is_err());
}

#[test]
fn ba_rejects_m0_ge_n_nodes() {
    assert!(generate_ba_core(5, 2, 5, 2, None).is_err());
}

#[test]
fn ba_sequential_labels() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_ba_core(10, 1, 1, 2, Some(0))?;
    // K=2, N=10: first 5 nodes → community 0, last 5 → community 1
    assert_eq!(result.labels[0], 0);
    assert_eq!(result.labels[9], 1);
    Ok(())
}

#[test]
fn ba_all_new_nodes_get_edges() -> Result<(), Box<dyn std::error::Error>> {
    let result = generate_ba_core(20, 2, 2, 1, Some(1))?;
    // Every node v >= m0=2 must appear in at least one edge.
    let mut appears = [false; 20];
    for &(src, dst) in &result.canonical_edges {
        appears[usize::try_from(src).unwrap_or(usize::MAX)] = true;
        appears[usize::try_from(dst).unwrap_or(usize::MAX)] = true;
    }
    for (v, &appeared) in appears.iter().enumerate() {
        assert!(appeared, "node {v} has no edges");
    }
    Ok(())
}

#[test]
fn ba_m1_grows_a_spanning_tree() -> Result<(), Box<dyn std::error::Error>> {
    // m = m0 = 1: the one-node seed has no edge, so node 1 must attach to node 0
    // or node 0 is never drawn and stays isolated. Each later node adds one edge,
    // so the n − 1 edges must form a single tree covering every node.
    for seed in 0..5 {
        let result = generate_ba_core(50, 1, 1, 1, Some(seed))?;
        assert_eq!(result.canonical_edges.len(), 49, "seed {seed}");
        let mut degree = [0usize; 50];
        for &(src, dst) in &result.canonical_edges {
            degree[usize::try_from(src)?] += 1;
            degree[usize::try_from(dst)?] += 1;
        }
        assert!(
            degree.iter().all(|&d| d >= 1),
            "seed {seed}: isolated node, degrees {degree:?}"
        );
        let graph = plekton::sbm_result_to_graph_output(&result);
        assert_eq!(graph.metadata.n_components, Some(1), "seed {seed}");
    }
    Ok(())
}

#[test]
fn validate_survives_invalid_json() {
    let warnings = validate_config_warn("not json at all");
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("Could not pre-validate"));
}

#[test]
fn validate_warns_low_avg_degree() {
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":100,"n_communities":3,"homophily":0.8,"avg_degree":0.5}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings.iter().any(|w| w.contains("avg_degree")),
        "{warnings:?}"
    );
}

#[test]
fn validate_warns_near_random_homophily() {
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":100,"n_communities":3,"homophily":0.51,"avg_degree":5.0}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings.iter().any(|w| w.contains("homophily")),
        "{warnings:?}"
    );
}

#[test]
fn validate_warns_blocks_too_small() {
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":8,"n_communities":4,"homophily":0.8,"avg_degree":3.0}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings.iter().any(|w| w.contains("blocks too small")),
        "{warnings:?}"
    );
}

#[test]
fn validate_warns_csbm_noise_features() {
    let json = r#"{"model_type":"csbm","parameters":{"n_nodes":200,"n_communities":3,"homophily":0.8,"avg_degree":5.0,"features_dim":4,"mu":0.3}}"#;
    let warnings = validate_config_warn(json);
    assert!(warnings.iter().any(|w| w.contains("noise")), "{warnings:?}");
}

#[test]
fn validate_ba_skips_homophily_and_avgdegree_checks() {
    let json = r#"{"model_type":"ba","parameters":{"n_nodes":100,"n_communities":3,"homophily":0.5,"avg_degree":0.0,"m":2}}"#;
    let warnings = validate_config_warn(json);
    // BA ignores avg_degree and homophily — no false positives
    assert!(
        !warnings.iter().any(|w| w.contains("avg_degree")),
        "{warnings:?}"
    );
    assert!(
        !warnings.iter().any(|w| w.contains("homophily")),
        "{warnings:?}"
    );
}

#[test]
fn validate_warns_very_small_graph() {
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":5,"n_communities":2,"homophily":0.8,"avg_degree":2.0}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings.iter().any(|w| w.contains("very small")),
        "{warnings:?}"
    );
}

#[test]
fn validate_no_small_graph_warning_at_ten() {
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":10,"n_communities":2,"homophily":0.8,"avg_degree":3.0}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        !warnings.iter().any(|w| w.contains("very small")),
        "{warnings:?}"
    );
}

#[test]
fn validate_warns_csbm_large_mu() {
    let json = r#"{"model_type":"csbm","parameters":{"n_nodes":500,"n_communities":3,"homophily":0.8,"avg_degree":5.0,"features_dim":16,"mu":15.0}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings.iter().any(|w| w.contains("trivial")),
        "{warnings:?}"
    );
}

#[test]
fn validate_warns_feat_dim_exceeds_n_nodes() {
    let json = r#"{"model_type":"csbm","parameters":{"n_nodes":50,"n_communities":3,"homophily":0.8,"avg_degree":5.0,"features_dim":100,"mu":1.0}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings.iter().any(|w| w.contains("feat_dim")),
        "{warnings:?}"
    );
}

#[test]
fn validate_warns_theta_near_phase_transition() {
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":200,"n_communities":3,"homophily":0.8,"avg_degree":5.0,"theta_exponent":2.05}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings.iter().any(|w| w.contains("phase transition")),
        "{warnings:?}"
    );
}

#[test]
fn validate_no_theta_warning_outside_transition() {
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":200,"n_communities":3,"homophily":0.8,"avg_degree":5.0,"theta_exponent":2.5}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        !warnings.iter().any(|w| w.contains("phase transition")),
        "{warnings:?}"
    );
}

// ─── BA does not carry block-model parameters ────────────────────────────────
//
// `homophily` and `avg_degree` describe a block model: BA's edges come from
// preferential attachment and its degree distribution is set by `m`. The
// validator already skipped both for `ba`, but the deserializer still demanded
// them, so a BA user had to invent values their model ignores. These pin the
// asymmetry so it cannot silently return.

#[test]
fn ba_config_omits_block_model_parameters() {
    let json = r#"{"model_type":"ba","seed":7,"parameters":{"n_nodes":120,"n_communities":3,"m":3,"m0":3}}"#;
    let Ok(graph) = generate_from_config_native(json) else {
        panic!("BA must generate without homophily/avg_degree");
    };
    assert!(
        graph.contains("\"nodes\""),
        "expected a graph document, got: {}",
        &graph[..graph.len().min(120)]
    );
}

#[test]
fn ba_config_without_block_params_pre_validates() {
    let json = r#"{"model_type":"ba","parameters":{"n_nodes":120,"n_communities":3,"m":3,"m0":3}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        !warnings
            .iter()
            .any(|w| w.contains("Could not pre-validate")),
        "BA config without homophily/avg_degree must parse: {warnings:?}"
    );
}

#[test]
fn block_models_still_require_their_own_parameters() {
    // The relaxation must not weaken the block models: n_communities is genuinely
    // used by every model (BA included, for community labels), so it stays required.
    let json = r#"{"model_type":"sbm_classique","parameters":{"n_nodes":100,"homophily":0.8,"avg_degree":5.0}}"#;
    let warnings = validate_config_warn(json);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("Could not pre-validate")),
        "a config missing n_communities must still fail to parse: {warnings:?}"
    );
}

#[test]
fn dcsbm_dyadic_large_scale_is_canonical_unique_and_on_target(
) -> Result<(), Box<dyn std::error::Error>> {
    // N = 20_000, K = 5, heavy-tailed gamma = 2.5, target avg_degree ≈ 10, homophily ≈ 0.8.
    // Under the old O(N²) single-ceiling rejection sampler this took >1.5 s; dyadic bucketing
    // completes in milliseconds while preserving canonical uniqueness and expected degree.
    let n = 20_000;
    let k = 5;
    let p_in = 0.8 * 10.0 / 3_999.0;
    let p_out = 0.2 * 10.0 / 16_000.0;
    let r = generate_sbm_core(n, k, p_in, p_out, Some(2.5), Some(42))?;

    let mut unique = std::collections::HashSet::with_capacity(r.canonical_edges.len());
    for &(u, v) in &r.canonical_edges {
        assert!(
            u >= 0 && v > u && usize::try_from(v)? < n,
            "non-canonical edge ({u}, {v})"
        );
        assert!(unique.insert((u, v)), "duplicate edge ({u}, {v})");
    }
    let avg_deg =
        2.0 * f64::from(u32::try_from(r.canonical_edges.len())?) / f64::from(u32::try_from(n)?);
    assert!(
        (avg_deg - 10.0).abs() < 1.5,
        "DC-SBM avg_degree={avg_deg:.2} drifted far from 10.0"
    );
    let effective_h = r.effective_h.ok_or("a graph with edges has a homophily")?;
    assert!(
        (effective_h - 0.8).abs() < 0.05,
        "DC-SBM homophily={effective_h:.3} drifted far from 0.8"
    );
    Ok(())
}
