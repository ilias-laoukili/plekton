// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! High-performance synthetic graph generation in pure Rust.
//!
//! Implements SBM, $O(N + E)$ exact DC-SBM, cSBM, Barabási–Albert, and
//! multi-attribute mixing graphs with controllable homophily, degree
//! distributions, node features, and community structure.
//!
//! Pure-Rust library — no Python or `NumPy` in the dependency tree.
//! Python bindings live in the sibling crate `plekton-python` (built by maturin).
//!
//! Block-pair sampling is intentionally sequential so the same seed yields
//! identical graphs across platforms.

// Every public item in this crate forms part of the generation API consumed by
// `plekton-python` and downstream Rust users, so all of it must be documented.
#![warn(missing_docs)]

mod error;
pub use error::SynthGraphError;

mod ba;
mod config;
mod csbm;
mod mixing;
mod output;
mod sampling;
mod sbm;
mod types;

pub use ba::generate_ba_core;
pub use config::{
    generate_from_config_native, validate_config_warn, ConfigParameters, GenerationConfig,
};
pub use csbm::{generate_csbm, CsbmConfig};
pub use mixing::{
    generate_mixing_graph, generate_mixing_graph_json, AttributeReport, ClusteringReport,
    MixingGraph, MixingRequest, MixingTarget, StrengthSpec,
};
pub use output::{
    arrays_to_graph_output, collect_node_features, csbm_result_to_graph_output,
    sbm_result_to_graph_output,
};
pub use sampling::{build_bidirectional_edge_flat, DegreeCorrection};
pub use sbm::{generate_sbm_core, generate_sbm_from_fit, BlockAssignment, BlockConnection, SbmFit};
pub use types::{
    CsbmCoreResult, EdgeJson, GraphOutput, Metadata, NodeJson, SbmCoreResult, GRAPH_SCHEMA_ID,
};
