// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Graph metrics and core-result → [`GraphOutput`] conversion.

use plekton_metrics::{
    assortativity::degree_assortativity,
    components::count_components,
    density::density as graph_density,
    skipped::SkippedMetric,
    transitivity::{transitivity, TRANSITIVITY_THRESHOLD},
};

use crate::{
    sampling::edge_homophily,
    types::{CsbmCoreResult, EdgeJson, GraphOutput, Metadata, NodeJson, SbmCoreResult},
    SynthGraphError,
};

/// Why `homophily` is `null` for a graph without edges.
const NO_EDGES: &str = "the graph has no edges";

/// Why `homophily` is `null` for a graph written without labels.
const NO_LABELS: &str = "the graph carries no labels; every node is written in community 0";

// Helpers: core result → GraphOutput

/// Computes graph metrics from the canonical edge list by delegating to
/// `plekton-metrics`.
///
/// Returns `(avg_degree_actual, n_components, transitivity_global, density, assortativity)`.
/// Metrics that are mathematically undefined or exceed the node threshold return
/// `None` (paired with a [`SkippedMetric`] entry in [`assemble_graph_output`]).
fn compute_metrics(
    n_nodes: usize,
    edges: &[(i64, i64)],
) -> (Option<f64>, usize, Option<f64>, Option<f64>, Option<f64>) {
    if n_nodes == 0 {
        // No node: no mean degree and no density to speak of, and no component.
        return (None, 0, None, None, None);
    }

    let n_edges = edges.len();
    let avg_degree = 2.0 * num_traits::ToPrimitive::to_f64(&n_edges).unwrap_or(0.0)
        / num_traits::ToPrimitive::to_f64(&n_nodes).unwrap_or(0.0);

    let edges_usize: Vec<(usize, usize)> = edges
        .iter()
        .map(|&(src, dst)| {
            (
                usize::try_from(src).unwrap_or(usize::MAX),
                usize::try_from(dst).unwrap_or(usize::MAX),
            )
        })
        .collect();

    let n_components = count_components(n_nodes, &edges_usize);
    let transitivity_global = transitivity(n_nodes, &edges_usize);
    let density = graph_density(n_nodes, n_edges);
    let assortativity = degree_assortativity(n_nodes, &edges_usize);

    (
        Some(avg_degree),
        n_components,
        transitivity_global,
        density,
        assortativity,
    )
}

/// Assemble a [`GraphOutput`] from the parts all result converters share: the
/// metadata block (incl. the `compute_metrics` pass) and the edge list.
///
/// `homophily` is the measured value, or the reason there is none.
fn assemble_graph_output(
    n_nodes: usize,
    canonical_edges: &[(i64, i64)],
    homophily: Result<f64, &'static str>,
    nodes: Vec<NodeJson>,
) -> GraphOutput {
    let (avg_degree_actual, n_components, transitivity_global, density, assortativity) =
        compute_metrics(n_nodes, canonical_edges);

    // In field order, so the reasons come in the order of the nulls they explain.
    let mut metrics_skipped = Vec::new();
    let homophily = match homophily {
        Ok(value) => Some(value),
        Err(reason) => {
            metrics_skipped.push(SkippedMetric::undefined("homophily", reason));
            None
        }
    };
    if avg_degree_actual.is_none() {
        metrics_skipped.push(SkippedMetric::undefined(
            "avg_degree_actual",
            "the graph has no nodes, so there is no mean to take",
        ));
    }
    if transitivity_global.is_none() {
        metrics_skipped.push(if n_nodes > TRANSITIVITY_THRESHOLD {
            SkippedMetric::too_large("transitivity_global", n_nodes, TRANSITIVITY_THRESHOLD)
        } else {
            SkippedMetric::undefined(
                "transitivity_global",
                "no node has two neighbours, so there is no open triplet to be a fraction of",
            )
        });
    }
    // Named rather than left as a bare null: this generator computes global
    // transitivity and nothing else. Before the two statistics had separate
    // fields, a reader of `clustering_coeff` could not tell which one it held.
    metrics_skipped.push(SkippedMetric::new(
        "clustering_mean_local",
        "this producer computes global transitivity only; mean local clustering is a \
         different statistic and was never measured here",
    ));
    if density.is_none() {
        metrics_skipped.push(SkippedMetric::undefined(
            "density",
            "the graph has fewer than two nodes, so no pair of them can be an edge",
        ));
    }
    if assortativity.is_none() {
        metrics_skipped.push(SkippedMetric::undefined(
            "assortativity",
            "the graph has no edges, or every edge joins the same degree pair",
        ));
    }

    GraphOutput {
        metadata: Metadata {
            schema: crate::types::GRAPH_SCHEMA_ID.to_string(),
            n_nodes,
            n_edges: canonical_edges.len(),
            homophily,
            avg_degree_actual,
            n_components: Some(n_components),
            // Deprecated, and written with exactly the value it has always held
            // here so no reader breaks in this cycle.
            clustering_coeff: transitivity_global,
            transitivity_global,
            clustering_mean_local: None,
            metrics_skipped,
            density,
            assortativity,
        },
        nodes,
        edges: canonical_edges
            .iter()
            .map(|&(src, dst)| EdgeJson {
                source: usize::try_from(src).unwrap_or(usize::MAX),
                target: usize::try_from(dst).unwrap_or(usize::MAX),
            })
            .collect(),
    }
}

/// Converts raw canonical undirected edges, optional per-node community labels,
/// and optional row-major node features into a canonical [`GraphOutput`].
///
/// Works uniformly for SBM, DC-SBM, fitted SBM, cSBM, and BA graphs. Without
/// labels (`None`), every node is written in community `0`, since the field is
/// required, and `homophily` is `null` with the reason in `metrics_skipped`:
/// there is no labelling to measure.
///
/// # Errors
/// Returns [`SynthGraphError::InvalidParameter`] if `labels` does not hold
/// `n_nodes` entries or an edge endpoint is negative or out of bounds
/// (`>= n_nodes`), or [`SynthGraphError::Feature`] if `x_flat` does not have
/// length `n_nodes * feat_dim`.
pub fn arrays_to_graph_output(
    n_nodes: usize,
    canonical_edges: &[(i64, i64)],
    labels: Option<&[usize]>,
    x_flat: Option<(&[f32], usize)>,
) -> Result<GraphOutput, SynthGraphError> {
    if let Some(labels) = labels.filter(|labels| labels.len() != n_nodes) {
        return Err(SynthGraphError::InvalidParameter(format!(
            "labels has {} entries for n_nodes={n_nodes}",
            labels.len()
        )));
    }
    for &(src, dst) in canonical_edges {
        let Ok(u) = usize::try_from(src) else {
            return Err(SynthGraphError::InvalidParameter(format!(
                "edge endpoint {src} must be non-negative"
            )));
        };
        let Ok(v) = usize::try_from(dst) else {
            return Err(SynthGraphError::InvalidParameter(format!(
                "edge endpoint {dst} must be non-negative"
            )));
        };
        if u >= n_nodes || v >= n_nodes {
            return Err(SynthGraphError::InvalidParameter(format!(
                "edge ({u}, {v}) out of bounds for n_nodes={n_nodes}"
            )));
        }
    }

    let community = |node_idx: usize| labels.map_or(0, |labels| labels[node_idx]);
    let nodes = match x_flat {
        Some((flat, feat_dim)) => {
            if flat.len() != n_nodes * feat_dim {
                return Err(SynthGraphError::Feature(format!(
                    "feature buffer length {} does not match n_nodes ({n_nodes}) * feat_dim ({feat_dim})",
                    flat.len()
                )));
            }
            (0..n_nodes)
                .map(|node_idx| NodeJson {
                    id: node_idx,
                    community: community(node_idx),
                    features: Some(flat[node_idx * feat_dim..(node_idx + 1) * feat_dim].to_vec()),
                })
                .collect()
        }
        None => (0..n_nodes)
            .map(|node_idx| NodeJson {
                id: node_idx,
                community: community(node_idx),
                features: None,
            })
            .collect(),
    };

    let homophily = match labels {
        Some(labels) => edge_homophily(canonical_edges, labels)?.ok_or(NO_EDGES),
        None => Err(NO_LABELS),
    };
    Ok(assemble_graph_output(
        n_nodes,
        canonical_edges,
        homophily,
        nodes,
    ))
}

/// Converts an SBM result into the JSON output structure.
#[must_use]
pub fn sbm_result_to_graph_output(result: &SbmCoreResult) -> GraphOutput {
    let nodes = (0..result.n_nodes)
        .map(|node_idx| NodeJson {
            id: node_idx,
            community: result.labels[node_idx],
            features: None,
        })
        .collect();
    assemble_graph_output(
        result.n_nodes,
        &result.canonical_edges,
        result.effective_h.ok_or(NO_EDGES),
        nodes,
    )
}

/// Converts a cSBM result into the JSON output structure.
#[must_use]
pub fn csbm_result_to_graph_output(result: &CsbmCoreResult) -> GraphOutput {
    let nodes = (0..result.n_nodes)
        .map(|node_idx| {
            let feats: Vec<f32> = result.x_flat
                [node_idx * result.feat_dim..(node_idx + 1) * result.feat_dim]
                .to_vec();
            NodeJson {
                id: node_idx,
                community: result.labels[node_idx],
                features: Some(feats),
            }
        })
        .collect();
    assemble_graph_output(
        result.n_nodes,
        &result.canonical_edges,
        result.effective_h.ok_or(NO_EDGES),
        nodes,
    )
}

/// Collects node features into a flat buffer.
/// Returns `Some((feat_dim, flat))` if all nodes have features, `None` if none do,
/// or an error if the feature vectors are inconsistent.
///
/// # Errors
/// Returns [`SynthGraphError`] if some (but not all) nodes carry features, or if
/// the per-node feature vectors differ in length.
pub fn collect_node_features(
    nodes: &[NodeJson],
) -> Result<Option<(usize, Vec<f32>)>, SynthGraphError> {
    let has_any = nodes.iter().any(|node| node.features.is_some());
    if !has_any {
        return Ok(None);
    }
    // At least one node has features — all must.
    let feat_dim = match nodes.iter().find(|node| node.features.is_some()) {
        Some(node) => match node.features.as_ref() {
            Some(feats) => feats.len(),
            None => {
                return Err(SynthGraphError::Feature(
                    "internal: feature predicate inconsistency".into(),
                ))
            }
        },
        None => {
            return Err(SynthGraphError::Feature(
                "internal: no node with features found despite has_any=true".into(),
            ));
        }
    };
    let mut flat: Vec<f32> = Vec::with_capacity(nodes.len() * feat_dim);
    for node in nodes {
        match &node.features {
            Some(feats) => {
                if feats.len() != feat_dim {
                    return Err(SynthGraphError::Feature(format!(
                        "node {} has {} features, expected {feat_dim}",
                        node.id,
                        feats.len()
                    )));
                }
                flat.extend_from_slice(feats);
            }
            None => {
                return Err(SynthGraphError::Feature(format!(
                    "node {} is missing features — all nodes must have features or none",
                    node.id
                )));
            }
        }
    }
    Ok(Some((feat_dim, flat)))
}

#[cfg(test)]
// exact-value assertions on deterministic metric outputs
mod tests {
    use super::*;

    #[test]
    fn compute_metrics_empty_graph() {
        let (avg, comps, clust, density, assort) = compute_metrics(5, &[]);
        // 0 edges over 5 nodes: a real mean degree of 0.
        assert_eq!(avg.map(f64::to_bits), Some(0.0_f64.to_bits()));
        assert_eq!(comps, 5); // all isolated
                              // no edges → no open triplets → transitivity undefined
        assert_eq!(clust, None);
        // density of a graph with 5 nodes and 0 edges = 0.0
        assert_eq!(density, Some(0.0));
        // no edges → assortativity undefined
        assert_eq!(assort, None);
    }

    #[test]
    fn compute_metrics_zero_nodes() {
        // No node: no mean degree and no density, rather than a made-up 0.0.
        let (avg, comps, clust, density, assort) = compute_metrics(0, &[]);
        assert_eq!(avg, None);
        assert_eq!(comps, 0);
        assert_eq!(clust, None);
        assert_eq!(density, None);
        assert_eq!(assort, None);
    }

    #[test]
    fn compute_metrics_single_node_has_no_density() {
        let (avg, comps, _, density, _) = compute_metrics(1, &[]);
        assert_eq!(avg.map(f64::to_bits), Some(0.0_f64.to_bits()));
        assert_eq!(comps, 1);
        assert_eq!(density, None);
    }

    /// The `metrics_skipped` entry for `metric`, if any.
    fn skipped_reason<'a>(graph: &'a GraphOutput, metric: &str) -> Option<&'a str> {
        graph
            .metadata
            .metrics_skipped
            .iter()
            .find(|skipped| skipped.metric == metric)
            .map(|skipped| skipped.reason.as_str())
    }

    #[test]
    fn every_undefined_metric_is_null_with_a_reason() -> Result<(), Box<dyn std::error::Error>> {
        // No node: no homophily, mean degree or density, each with its reason.
        let empty = arrays_to_graph_output(0, &[], Some(&[]), None)?;
        assert_eq!(empty.metadata.homophily, None);
        assert_eq!(empty.metadata.avg_degree_actual, None);
        assert_eq!(empty.metadata.n_components, Some(0));
        assert_eq!(empty.metadata.density, None);
        assert_eq!(
            skipped_reason(&empty, "homophily"),
            Some("undefined here: the graph has no edges")
        );
        assert!(skipped_reason(&empty, "avg_degree_actual").is_some());
        assert!(skipped_reason(&empty, "density").is_some());

        // One node: a mean degree of 0, but no density.
        let single = arrays_to_graph_output(1, &[], Some(&[0]), None)?;
        assert_eq!(
            single.metadata.avg_degree_actual.map(f64::to_bits),
            Some(0.0_f64.to_bits())
        );
        assert_eq!(single.metadata.density, None);
        assert!(skipped_reason(&single, "avg_degree_actual").is_none());
        assert!(skipped_reason(&single, "density").is_some());

        // No labels: every node in community 0, and no homophily to report.
        let unlabelled = arrays_to_graph_output(3, &[(0, 1), (1, 2)], None, None)?;
        assert!(unlabelled.nodes.iter().all(|node| node.community == 0));
        assert_eq!(unlabelled.metadata.homophily, None);
        assert_eq!(
            skipped_reason(&unlabelled, "homophily"),
            Some(
                "undefined here: the graph carries no labels; every node is written in \
                 community 0"
            )
        );

        // One community is a labelling: every edge stays inside it.
        let one_community = arrays_to_graph_output(3, &[(0, 1), (1, 2)], Some(&[0, 0, 0]), None)?;
        assert_eq!(
            one_community.metadata.homophily.map(f64::to_bits),
            Some(1.0_f64.to_bits())
        );
        assert!(skipped_reason(&one_community, "homophily").is_none());
        Ok(())
    }

    #[test]
    fn labels_must_cover_every_node() {
        let err = arrays_to_graph_output(3, &[(0, 1)], Some(&[0, 0]), None);
        assert!(matches!(err, Err(SynthGraphError::InvalidParameter(_))));
    }

    #[test]
    fn compute_metrics_triangle() -> Result<(), Box<dyn std::error::Error>> {
        // 3 nodes forming a triangle: (0,1), (1,2), (0,2)
        let edges = vec![(0i64, 1i64), (1i64, 2i64), (0i64, 2i64)];
        let (avg, comps, clust, density, assort) = compute_metrics(3, &edges);
        let avg = avg.ok_or("expected a mean degree")?;
        assert!((avg - 2.0).abs() < 1e-9, "avg_degree={avg}");
        assert_eq!(comps, 1);
        assert!(
            (clust.ok_or("expected a value")? - 1.0).abs() < 1e-9,
            "clustering={clust:?}"
        );
        // K3 is a complete graph: density = 1.0
        assert!(
            (density.ok_or("expected a value")? - 1.0).abs() < 1e-9,
            "density={density:?}"
        );
        // K3: all nodes degree 2 → zero degree variance → assortativity undefined
        assert_eq!(assort, None);
        Ok(())
    }

    #[test]
    fn compute_metrics_two_components() {
        // 4 nodes: (0,1) and (2,3) — two separate edges
        let edges = vec![(0i64, 1i64), (2i64, 3i64)];
        let (_, comps, _, _, _) = compute_metrics(4, &edges);
        assert_eq!(comps, 2);
    }

    #[test]
    fn compute_metrics_large_no_clustering() {
        // n_nodes > 20_000 → clustering_coeff must be None
        let (_, _, clust, _, _) = compute_metrics(20_001, &[]);
        assert!(clust.is_none());
    }

    #[test]
    fn compute_metrics_path_graph() -> Result<(), Box<dyn std::error::Error>> {
        // Path 0-1-2: no triangles → clustering = 0
        let edges = vec![(0i64, 1i64), (1i64, 2i64)];
        let (avg, comps, clust, density, _assort) = compute_metrics(3, &edges);
        let avg = avg.ok_or("expected a mean degree")?;
        assert!((avg - 4.0 / 3.0).abs() < 1e-9, "avg_degree={avg}");
        assert_eq!(comps, 1);
        assert_eq!(clust, Some(0.0));
        // density: 2 edges / (3*2/2=3 max) = 2/3
        assert!(
            (density.ok_or("expected a value")? - 2.0 / 3.0).abs() < 1e-9,
            "density={density:?}"
        );
        Ok(())
    }

    #[test]
    fn compute_metrics_star_is_disassortative() -> Result<(), Box<dyn std::error::Error>> {
        // Star K_{1,3}: hub (node 0) degree=3, leaves degree=1 → r < 0
        let edges = vec![(0i64, 1i64), (0i64, 2i64), (0i64, 3i64)];
        let (_, _, _, _, assort) = compute_metrics(4, &edges);
        assert!(
            assort.ok_or("expected a value")? < 0.0,
            "star should be disassortative, got {assort:?}"
        );
        Ok(())
    }

    #[test]
    fn compute_metrics_density_complete_k4() -> Result<(), Box<dyn std::error::Error>> {
        // K4: 6 edges, N=4 → density = 6 / (4*3/2) = 1.0
        let edges = vec![
            (0i64, 1i64),
            (0i64, 2i64),
            (0i64, 3i64),
            (1i64, 2i64),
            (1i64, 3i64),
            (2i64, 3i64),
        ];
        let (_, _, _, density, _) = compute_metrics(4, &edges);
        assert!(
            (density.ok_or("expected a value")? - 1.0).abs() < 1e-9,
            "density={density:?}"
        );
        Ok(())
    }
}
