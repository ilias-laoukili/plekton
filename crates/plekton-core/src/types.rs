// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Serializable graph DTOs (the JSON contract) and the pure-Rust generator
//! result types shared by the JSON and `PyG` output paths.

use plekton_metrics::skipped::SkippedMetric;
use serde::{Deserialize, Serialize};

// JSON output structs — canonical graph-JSON contract: metadata / nodes / edges

/// Format id stamped into every graph JSON's `metadata.schema`.
///
/// The external contract is `schemas/graph.schema.json`.
pub const GRAPH_SCHEMA_ID: &str = "plekton.graph/v1";

/// Serde default: files written before the tag existed read as the current id.
fn graph_schema_id() -> String {
    GRAPH_SCHEMA_ID.to_string()
}

/// Graph-level statistics serialised in the JSON metadata block.
#[derive(Serialize, Deserialize, Clone)]
pub struct Metadata {
    /// Graph-JSON format id ([`GRAPH_SCHEMA_ID`]).
    #[serde(default = "graph_schema_id")]
    pub schema: String,
    /// Number of nodes in the generated graph.
    pub n_nodes: usize,
    /// Number of canonical (undirected) edges in the graph.
    pub n_edges: usize,
    /// Effective homophily — the fraction of edges whose endpoints share a
    /// community label. Differs from the *requested* homophily because of
    /// rounding when block sizes are unequal.
    ///
    /// Always serialised; `None` (JSON `null`) when it is undefined — the graph
    /// has no edges, or carries no labels at all (every node then written in
    /// community `0`) — with a `metrics_skipped` entry saying which. A single
    /// community is a labelling, and gives `1.0`.
    pub homophily: Option<f64>,
    /// Mean degree across all nodes (`2 · n_edges / n_nodes`).
    ///
    /// `None`, and not serialised, for a graph with no node, with a
    /// `metrics_skipped` entry. A document without the field reads as `None`
    /// rather than a made-up `0.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avg_degree_actual: Option<f64>,
    /// Number of connected components reachable through the canonical edges
    /// (`0` for a graph with no node).
    ///
    /// Always written by this crate. A document without the field reads as
    /// `None` rather than a made-up `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_components: Option<usize>,
    /// **Deprecated — read [`Self::transitivity_global`] instead.**
    ///
    /// Legacy alias kept for compatibility with earlier graph-JSON readers;
    /// holds the same value as [`Self::transitivity_global`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clustering_coeff: Option<f64>,
    /// Global transitivity: `3·triangles / open_triplets`.
    ///
    /// Named for the statistic it is. `None` above the 20 000-node threshold,
    /// where triangle counting would dominate the generation time, and `None`
    /// for a graph with no open triplet to close — `metrics_skipped` says which.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transitivity_global: Option<f64>,
    /// Mean local clustering: the average over nodes of each node's own
    /// coefficient.
    ///
    /// A different statistic from [`Self::transitivity_global`], which weights
    /// by degree. Always `None` here — this producer does not compute it, and
    /// `metrics_skipped` says so rather than leaving a bare null to be read as
    /// zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clustering_mean_local: Option<f64>,
    /// Why any metric above is `None`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics_skipped: Vec<SkippedMetric>,
    /// Graph density: `2 · n_edges / (n_nodes · (n_nodes - 1))`.
    ///
    /// `None`, and not serialised, below two nodes, where no pair of nodes can
    /// be an edge, with a `metrics_skipped` entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density: Option<f64>,
    /// Degree assortativity (Newman's r). `None` for edgeless graphs where the
    /// quantity is mathematically undefined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assortativity: Option<f64>,
}

/// Node entry in the JSON output.
#[derive(Serialize, Deserialize, Clone)]
pub struct NodeJson {
    /// Node index — globally unique within the graph; `0..n_nodes`.
    pub id: usize,
    /// Community / block assignment used as the node's classification label.
    pub community: usize,
    /// Optional feature vector. Present for cSBM, absent for plain SBM / BA.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<f32>>,
}

/// Canonical undirected edge in the JSON output (`source < target`).
#[derive(Serialize, Deserialize, Clone)]
pub struct EdgeJson {
    /// Lower-indexed endpoint of the edge.
    pub source: usize,
    /// Higher-indexed endpoint of the edge.
    pub target: usize,
}

/// Root structure of the JSON output format.
///
/// Matches the canonical graph-JSON contract:
/// `{ metadata, nodes: [{ id, community, features? }], edges: [{ source, target }] }`
#[derive(Serialize, Deserialize, Clone)]
pub struct GraphOutput {
    /// Aggregate metrics for the whole graph.
    pub metadata: Metadata,
    /// Per-node entries; index in this slice equals `id` for SBM outputs.
    pub nodes: Vec<NodeJson>,
    /// Canonical undirected edge list (`source < target`).
    pub edges: Vec<EdgeJson>,
}

// Core result types (shared by PyG and JSON output paths)

/// Output of SBM / DC-SBM generation. Pure-Rust analogue of [`GraphOutput`] used
/// by callers that want to skip JSON (e.g. `PyG` conversion).
#[derive(Debug)]
pub struct SbmCoreResult {
    /// Canonical undirected edges — invariant `src < dst`.
    pub canonical_edges: Vec<(i64, i64)>,
    /// Community labels indexed by node id.
    pub labels: Vec<usize>,
    /// Effective homophily (see [`Metadata::homophily`]); `None` when the graph
    /// has no edge.
    pub effective_h: Option<f64>,
    /// Number of nodes in the result; mirrors `labels.len()`.
    pub n_nodes: usize,
}

/// Output of cSBM generation — SBM result plus per-node feature vectors.
#[derive(Debug)]
pub struct CsbmCoreResult {
    /// Canonical undirected edges — invariant `src < dst`.
    pub canonical_edges: Vec<(i64, i64)>,
    /// Community labels indexed by node id.
    pub labels: Vec<usize>,
    /// Row-major node features: `x_flat[node_idx · feat_dim .. (node_idx + 1) · feat_dim]`
    /// is the feature vector for `node_idx`. Flat layout avoids one `Vec<Vec<f32>>`
    /// allocation per node and matches `PyG`'s `x` tensor layout.
    pub x_flat: Vec<f32>,
    /// Effective homophily (see [`Metadata::homophily`]); `None` when the graph
    /// has no edge.
    pub effective_h: Option<f64>,
    /// Number of nodes; mirrors `labels.len()`.
    pub n_nodes: usize,
    /// Feature dimension; `x_flat.len() == n_nodes · feat_dim`.
    pub feat_dim: usize,
}
