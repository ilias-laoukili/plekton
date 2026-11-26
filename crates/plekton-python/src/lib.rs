// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Python bindings for `plekton`. Thin, zero-copy wrappers over pure-Rust crates.
//!
//! Zero-copy contract:
//!   * Every `Array1<T>` / `Array2<T>` produced here is a fresh owned, C-contiguous
//!     allocation handed to `NumPy` via `IntoPyArray::into_pyarray(py)`, transferring
//!     ownership without copying.
//!   * Never use `to_pyarray_bound` here — it clones.
//!   * Generators hand over their canonical edges as an `(E, 2)` array, built off
//!     the GIL by reusing the core's own `canonical_edges` buffer in place. The
//!     bidirectional `PyG` `edge_index` is derived in Python only when asked for.
//!
//! The compiled module is exposed as `plekton._plekton` and wrapped by
//! `python/plekton/`, where every generator returns a `plekton.Graph`. Seeds are
//! always explicit here: the Python side draws one when the caller passes none.

#![allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 hands #[pyfunction] owned PyReadonly array guards by value"
)]

use std::fs;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use plekton::{
    arrays_to_graph_output, collect_node_features, generate_ba_core,
    generate_csbm as generate_csbm_core, generate_from_config_native, generate_sbm_core,
    generate_sbm_from_fit as generate_sbm_from_fit_core, BlockAssignment, BlockConnection,
    CsbmConfig, CsbmCoreResult, DegreeCorrection, GraphOutput, SbmCoreResult, SbmFit,
    SynthGraphError,
};
use pyo3::{exceptions::PyValueError, prelude::*};

mod edges_py;
mod fit_py;
mod graph_sparsify_py;
mod metrics_py;
mod mixing_py;

fn err(e: &SynthGraphError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// `(edges [E, 2], labels [N], effective_h)`.
type SbmArrays<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>, f64);

/// `(edges [E, 2], x [N, F], labels [N], effective_h)`.
type CsbmArrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i64>>,
    f64,
);

/// `(edges [E, 2], labels [N], optional x [N, F], metadata as JSON)`.
type GraphArrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Option<Bound<'py, PyArray2<f32>>>,
    String,
);

/// The owned arrays [`json_to_graph`] builds without the GIL.
type ParsedGraph = (Array2<i64>, Array1<i64>, Option<Array2<f32>>, String);

/// Canonical `(u, v)` pairs as a C-contiguous `(E, 2)` array.
///
/// Each `(i64, i64)` becomes an `[i64; 2]` in the same allocation (identical size
/// and alignment, so the collection happens in place), then the spare capacity left
/// by the sampler's growth is released. The array holds exactly 16 bytes per edge,
/// and no second edge buffer ever coexists with the first.
fn canonical_edges_array(edges: Vec<(i64, i64)>) -> Result<Array2<i64>, SynthGraphError> {
    let n_edges = edges.len();
    let mut flat = edges
        .into_iter()
        .map(<[i64; 2]>::from)
        .collect::<Vec<[i64; 2]>>()
        .into_flattened();
    flat.shrink_to_fit();
    Array2::from_shape_vec((n_edges, 2), flat)
        .map_err(|error| SynthGraphError::GenerationFailed(format!("edges shape error: {error}")))
}

/// Community labels as `int64`, converted in place (`usize` and `i64` share a layout).
fn labels_array(labels: Vec<usize>) -> Array1<i64> {
    Array1::from(labels.into_iter().map(to_i64).collect::<Vec<i64>>())
}

/// Output of an SBM-family generator, converted without the GIL.
struct SbmOwned {
    edges: Array2<i64>,
    labels: Array1<i64>,
    effective_h: f64,
}

impl SbmOwned {
    fn from_result(result: SbmCoreResult) -> Result<Self, SynthGraphError> {
        let SbmCoreResult {
            canonical_edges,
            labels,
            effective_h,
            ..
        } = result;
        Ok(Self {
            edges: canonical_edges_array(canonical_edges)?,
            labels: labels_array(labels),
            effective_h,
        })
    }

    fn into_py(self, py: Python<'_>) -> SbmArrays<'_> {
        (
            self.edges.into_pyarray(py),
            self.labels.into_pyarray(py),
            self.effective_h,
        )
    }
}

// ─── Synthetic graph generators ──────────────────────────────────────────────

/// SBM / DC-SBM: returns `(edges [E, 2], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, n_communities, p_in, p_out, theta_exponent, seed))]
fn generate_sbm(
    py: Python<'_>,
    n_nodes: usize,
    n_communities: usize,
    p_in: f64,
    p_out: f64,
    theta_exponent: Option<f64>,
    seed: u64,
) -> PyResult<SbmArrays<'_>> {
    let owned = py
        .allow_threads(|| {
            generate_sbm_core(
                n_nodes,
                n_communities,
                p_in,
                p_out,
                theta_exponent,
                Some(seed),
            )
            .and_then(SbmOwned::from_result)
        })
        .map_err(|e| err(&e))?;
    Ok(owned.into_py(py))
}

/// Fitted SBM / DC-SBM: returns `(edges [E, 2], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, block_sizes, b_matrix, theta, seed))]
fn generate_sbm_from_fit<'py>(
    py: Python<'py>,
    n_nodes: usize,
    block_sizes: Vec<usize>,
    b_matrix: PyReadonlyArray2<'py, f64>,
    theta: Option<PyReadonlyArray1<'py, f64>>,
    seed: u64,
) -> PyResult<SbmArrays<'py>> {
    let b_slice = b_matrix
        .as_slice()
        .map_err(|_| PyValueError::new_err("b_matrix must be C-contiguous"))?;
    let theta_vec: Option<Vec<f64>> = match theta {
        Some(ref arr) => Some(
            arr.as_slice()
                .map_err(|_| PyValueError::new_err("theta must be C-contiguous"))?
                .to_vec(),
        ),
        None => None,
    };
    let b_vec = b_slice.to_vec();

    let owned = py
        .allow_threads(move || {
            let dc = theta_vec
                .as_deref()
                .map_or(DegreeCorrection::Off, DegreeCorrection::Supplied);
            let fit = SbmFit {
                assignment: BlockAssignment::Sizes(&block_sizes),
                connection: BlockConnection::Matrix(&b_vec),
                degree_correction: dc,
            };
            generate_sbm_from_fit_core(n_nodes, &fit, Some(seed)).and_then(SbmOwned::from_result)
        })
        .map_err(|e| err(&e))?;
    Ok(owned.into_py(py))
}

/// Barabási–Albert: returns `(edges [E, 2], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, m, m0, n_communities, seed))]
fn generate_ba(
    py: Python<'_>,
    n_nodes: usize,
    m: usize,
    m0: Option<usize>,
    n_communities: usize,
    seed: u64,
) -> PyResult<SbmArrays<'_>> {
    let initial = m0.unwrap_or(m);
    let owned = py
        .allow_threads(|| {
            generate_ba_core(n_nodes, m, initial, n_communities, Some(seed))
                .and_then(SbmOwned::from_result)
        })
        .map_err(|e| err(&e))?;
    Ok(owned.into_py(py))
}

/// cSBM: returns `(edges [E, 2], x [N, F], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (
    n_nodes, n_communities, homophily, avg_degree, feat_dim, mu,
    feature_dist, class_weights, theta_exponent,
    p_triangle, feat_noise_ratio, ensure_connected,
    p_in_override, p_out_override, b_matrix,
    centroids, covariance, seed
))]
#[allow(
    clippy::too_many_arguments,
    reason = "matches the cSBM keyword surface directly across the PyO3 boundary"
)]
fn generate_csbm<'py>(
    py: Python<'py>,
    n_nodes: usize,
    n_communities: usize,
    homophily: f64,
    avg_degree: f64,
    feat_dim: usize,
    mu: f64,
    feature_dist: &str,
    class_weights: Option<Vec<f64>>,
    theta_exponent: Option<f64>,
    p_triangle: f64,
    feat_noise_ratio: f64,
    ensure_connected: bool,
    p_in_override: Option<f64>,
    p_out_override: Option<f64>,
    b_matrix: Option<Vec<f64>>,
    centroids: Option<Vec<f64>>,
    covariance: Option<Vec<f64>>,
    seed: u64,
) -> PyResult<CsbmArrays<'py>> {
    let fd = feature_dist.to_owned();
    let (edges, x, labels, effective_h) = py
        .allow_threads(move || {
            let cfg = CsbmConfig {
                n_nodes,
                n_communities,
                homophily,
                avg_degree,
                feat_dim,
                mu,
                feature_dist: &fd,
                class_weights: class_weights.as_deref(),
                theta_exponent,
                p_triangle,
                feat_noise_ratio,
                ensure_connected,
                p_in_override,
                p_out_override,
                b_matrix: b_matrix.as_deref(),
                centroids_override: centroids.as_deref(),
                covariance: covariance.as_deref(),
                seed: Some(seed),
            };
            let CsbmCoreResult {
                canonical_edges,
                labels,
                x_flat,
                effective_h,
                n_nodes,
                feat_dim,
            } = generate_csbm_core(&cfg)?;
            let edges = canonical_edges_array(canonical_edges)?;
            let x = Array2::from_shape_vec((n_nodes, feat_dim), x_flat)
                .map_err(|error| SynthGraphError::Feature(format!("x shape error: {error}")))?;
            Ok::<_, SynthGraphError>((edges, x, labels_array(labels), effective_h))
        })
        .map_err(|e| err(&e))?;

    Ok((
        edges.into_pyarray(py),
        x.into_pyarray(py),
        labels.into_pyarray(py),
        effective_h,
    ))
}

// ─── JSON output ─────────────────────────────────────────────────────────────

/// Serialises `(edge_index [2, M], labels [N], optional x [N, F])` to canonical
/// `plekton.graph/v1` JSON, optionally writing the document to `path`.
#[pyfunction]
#[pyo3(signature = (edge_index, labels, x=None, path=None))]
fn graph_to_json<'py>(
    py: Python<'py>,
    edge_index: PyReadonlyArray2<'py, i64>,
    labels: PyReadonlyArray1<'py, i64>,
    x: Option<PyReadonlyArray2<'py, f32>>,
    path: Option<&str>,
) -> PyResult<String> {
    let ei_view = edge_index.as_array();
    if ei_view.nrows() != 2 {
        return Err(PyValueError::new_err(format!(
            "edge_index must have shape (2, E), got ({}, {})",
            ei_view.nrows(),
            ei_view.ncols()
        )));
    }
    let m = ei_view.ncols();
    let mut raw_edges: Vec<(i64, i64)> = Vec::with_capacity(m);
    for col in 0..m {
        raw_edges.push((ei_view[[0, col]], ei_view[[1, col]]));
    }

    let labels_view = labels.as_array();
    let mut labels_usize: Vec<usize> = Vec::with_capacity(labels_view.len());
    for &lbl in labels_view {
        let Ok(community) = usize::try_from(lbl) else {
            return Err(PyValueError::new_err(format!(
                "community label {lbl} must be non-negative"
            )));
        };
        labels_usize.push(community);
    }

    let x_owned: Option<(Vec<f32>, usize)> = match x {
        Some(ref arr) => {
            let x_view = arr.as_array();
            let feat_dim = x_view.ncols();
            if x_view.nrows() != labels_usize.len() {
                return Err(PyValueError::new_err(format!(
                    "x has {} rows but labels has length {}",
                    x_view.nrows(),
                    labels_usize.len()
                )));
            }
            let flat: Vec<f32> = x_view.iter().copied().collect();
            Some((flat, feat_dim))
        }
        None => None,
    };

    let path_owned = path.map(str::to_owned);
    py.allow_threads(move || {
        let mut canonical_edges: Vec<(i64, i64)> = raw_edges
            .into_iter()
            .filter_map(|(u, v)| {
                if u == v {
                    None
                } else if u < v {
                    Some((u, v))
                } else {
                    Some((v, u))
                }
            })
            .collect();
        canonical_edges.sort_unstable();
        canonical_edges.dedup();

        let x_ref = x_owned
            .as_ref()
            .map(|(flat, feat_dim)| (flat.as_slice(), *feat_dim));
        let graph = arrays_to_graph_output(&canonical_edges, &labels_usize, x_ref)?;
        let json = serde_json::to_string_pretty(&graph).map_err(SynthGraphError::Serialization)?;
        if let Some(out_path) = path_owned {
            fs::write(out_path, &json).map_err(SynthGraphError::Io)?;
        }
        Ok(json)
    })
    .map_err(|e| err(&e))
}

// ─── JSON input ──────────────────────────────────────────────────────────────

/// Parses canonical `plekton.graph/v1` JSON into `(edges [E, 2], labels [N],
/// optional x [N, F], metadata as JSON)`. The edges come back exactly as the
/// document lists them; `plekton.Graph.from_json` validates them.
#[pyfunction]
#[pyo3(signature = (json_str))]
fn json_to_graph<'py>(py: Python<'py>, json_str: &str) -> PyResult<GraphArrays<'py>> {
    let (edges, labels, x, metadata) = py
        .allow_threads(|| parse_graph_json(json_str))
        .map_err(PyValueError::new_err)?;
    Ok((
        edges.into_pyarray(py),
        labels.into_pyarray(py),
        x.map(|arr| arr.into_pyarray(py)),
        metadata,
    ))
}

fn parse_graph_json(json_str: &str) -> Result<ParsedGraph, String> {
    let graph: GraphOutput =
        serde_json::from_str(json_str).map_err(|error| format!("JSON parse error: {error}"))?;
    // Labels and features are indexed by position, so a node listed out of order
    // would silently take another node's row.
    if let Some((position, node)) = graph
        .nodes
        .iter()
        .enumerate()
        .find(|(position, node)| node.id != *position)
    {
        return Err(format!(
            "node at position {position} has id {}: plekton.graph/v1 lists nodes as ids \
             0..n_nodes, in order",
            node.id
        ));
    }
    let x = collect_node_features(&graph.nodes)
        .map_err(|error| error.to_string())?
        .map(|(feat_dim, flat)| Array2::from_shape_vec((graph.nodes.len(), feat_dim), flat))
        .transpose()
        .map_err(|error| format!("x shape error: {error}"))?;
    let metadata =
        serde_json::to_string(&graph.metadata).map_err(|error| format!("metadata: {error}"))?;

    let GraphOutput { nodes, edges, .. } = graph;
    let labels = Array1::from(
        nodes
            .into_iter()
            .map(|node| to_i64(node.community))
            .collect::<Vec<i64>>(),
    );
    let n_edges = edges.len();
    let flat = edges
        .into_iter()
        .map(|edge| [to_i64(edge.source), to_i64(edge.target)])
        .collect::<Vec<[i64; 2]>>()
        .into_flattened();
    let edges = Array2::from_shape_vec((n_edges, 2), flat)
        .map_err(|error| format!("edges shape error: {error}"))?;
    Ok((edges, labels, x, metadata))
}

// ─── Config-driven dispatch ──────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (config_json))]
fn generate_from_config(py: Python<'_>, config_json: &str) -> PyResult<String> {
    py.allow_threads(|| generate_from_config_native(config_json))
        .map_err(|e| err(&e))
}

// ─── Module registration ─────────────────────────────────────────────────────

#[pymodule]
fn _plekton(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_sbm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_sbm_from_fit, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ba, m)?)?;
    m.add_function(wrap_pyfunction!(generate_csbm, m)?)?;
    m.add_function(wrap_pyfunction!(graph_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(json_to_graph, m)?)?;
    m.add_function(wrap_pyfunction!(generate_from_config, m)?)?;

    mixing_py::register(m)?;
    graph_sparsify_py::register(py, m)?;
    fit_py::register(py, m)?;
    metrics_py::register(py, m)?;

    Ok(())
}
