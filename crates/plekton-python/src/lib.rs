// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Python bindings for `plekton`. Thin, zero-copy wrappers over pure-Rust crates.
//!
//! Zero-copy contract:
//!   * Every `Array1<T>` / `Array2<T>` produced here is a fresh owned, C-contiguous
//!     allocation handed to `NumPy` via `IntoPyArray::into_pyarray(py)`, transferring
//!     ownership without copying.
//!   * Never use `to_pyarray_bound` here — it clones.
//!
//! The compiled module is exposed as `plekton._plekton` and re-exported by
//! `python/plekton/__init__.py`.

#![allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 hands #[pyfunction] owned PyReadonly array guards by value"
)]

use std::fs;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use plekton::{
    arrays_to_graph_output, build_bidirectional_edge_flat, collect_node_features, generate_ba_core,
    generate_csbm as generate_csbm_core, generate_from_config_native, generate_sbm_core,
    generate_sbm_from_fit as generate_sbm_from_fit_core, BlockAssignment, BlockConnection,
    CsbmConfig, DegreeCorrection, GraphOutput, SbmCoreResult, SbmFit, SynthGraphError,
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

/// `(edge_index [2,2E], labels [N], effective_h)`.
type SbmArrays<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>, f64);

/// `(edge_index [2,2E], x [N,F], labels [N], effective_h)`.
type CsbmArrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i64>>,
    f64,
);

/// `(edge_index [2,2E], labels [N], optional x [N,F])`.
type PygArrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Option<Bound<'py, PyArray2<f32>>>,
);

fn sbm_result_to_arrays<'py>(py: Python<'py>, result: &SbmCoreResult) -> PyResult<SbmArrays<'py>> {
    let (flat, total_directed) = build_bidirectional_edge_flat(&result.canonical_edges);
    let edge_arr = Array2::from_shape_vec((2, total_directed), flat)
        .map_err(|error| PyValueError::new_err(format!("edge_index shape error: {error}")))?;
    let label_arr = Array1::from(
        result
            .labels
            .iter()
            .map(|&lbl| to_i64(lbl))
            .collect::<Vec<_>>(),
    );
    Ok((
        edge_arr.into_pyarray(py),
        label_arr.into_pyarray(py),
        result.effective_h,
    ))
}

// ─── Synthetic graph generators ──────────────────────────────────────────────

/// SBM / DC-SBM: returns `(edge_index [2,2E], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, n_communities, p_in, p_out, theta_exponent=None, seed=None))]
fn generate_sbm(
    py: Python<'_>,
    n_nodes: usize,
    n_communities: usize,
    p_in: f64,
    p_out: f64,
    theta_exponent: Option<f64>,
    seed: Option<u64>,
) -> PyResult<SbmArrays<'_>> {
    let result = py
        .allow_threads(|| {
            generate_sbm_core(n_nodes, n_communities, p_in, p_out, theta_exponent, seed)
        })
        .map_err(|e| err(&e))?;
    sbm_result_to_arrays(py, &result)
}

/// Fitted SBM / DC-SBM: returns `(edge_index [2,2E], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, block_sizes, b_matrix, theta=None, seed=None))]
fn generate_sbm_from_fit<'py>(
    py: Python<'py>,
    n_nodes: usize,
    block_sizes: Vec<usize>,
    b_matrix: PyReadonlyArray2<'py, f64>,
    theta: Option<PyReadonlyArray1<'py, f64>>,
    seed: Option<u64>,
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

    let result = py
        .allow_threads(move || {
            let dc = theta_vec
                .as_deref()
                .map_or(DegreeCorrection::Off, DegreeCorrection::Supplied);
            let fit = SbmFit {
                assignment: BlockAssignment::Sizes(&block_sizes),
                connection: BlockConnection::Matrix(&b_vec),
                degree_correction: dc,
            };
            generate_sbm_from_fit_core(n_nodes, &fit, seed)
        })
        .map_err(|e| err(&e))?;
    sbm_result_to_arrays(py, &result)
}

/// Barabási–Albert: returns `(edge_index [2,2E], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (n_nodes, m=2, m0=None, n_communities=1, seed=None))]
fn generate_ba(
    py: Python<'_>,
    n_nodes: usize,
    m: usize,
    m0: Option<usize>,
    n_communities: usize,
    seed: Option<u64>,
) -> PyResult<SbmArrays<'_>> {
    let initial = m0.unwrap_or(m);
    let result = py
        .allow_threads(|| generate_ba_core(n_nodes, m, initial, n_communities, seed))
        .map_err(|e| err(&e))?;
    sbm_result_to_arrays(py, &result)
}

/// cSBM: returns `(edge_index [2,2E], x [N,F], labels [N], effective_h)`.
#[pyfunction]
#[pyo3(signature = (
    n_nodes, n_communities, homophily, avg_degree, feat_dim, mu,
    feature_dist="gaussian", class_weights=None, theta_exponent=None,
    p_triangle=0.0, feat_noise_ratio=0.0, ensure_connected=false,
    p_in_override=None, p_out_override=None, b_matrix=None,
    centroids=None, covariance=None, seed=None
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
    seed: Option<u64>,
) -> PyResult<CsbmArrays<'py>> {
    let fd = feature_dist.to_owned();
    let result = py
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
                seed,
            };
            generate_csbm_core(&cfg)
        })
        .map_err(|e| err(&e))?;

    let (flat, total_directed) = build_bidirectional_edge_flat(&result.canonical_edges);
    let edge_arr = Array2::from_shape_vec((2, total_directed), flat)
        .map_err(|error| PyValueError::new_err(format!("edge_index shape error: {error}")))?;
    let x_arr = Array2::from_shape_vec((result.n_nodes, result.feat_dim), result.x_flat)
        .map_err(|error| PyValueError::new_err(format!("x shape error: {error}")))?;
    let label_arr = Array1::from(
        result
            .labels
            .iter()
            .map(|&lbl| to_i64(lbl))
            .collect::<Vec<_>>(),
    );

    Ok((
        edge_arr.into_pyarray(py),
        x_arr.into_pyarray(py),
        label_arr.into_pyarray(py),
        result.effective_h,
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

// ─── PyG tensor converter ────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (json_str))]
fn json_to_pyg<'py>(py: Python<'py>, json_str: &str) -> PyResult<PygArrays<'py>> {
    let graph: GraphOutput = serde_json::from_str(json_str)
        .map_err(|error| PyValueError::new_err(format!("JSON parse error: {error}")))?;

    let n_nodes = graph.nodes.len();
    let labels: Vec<i64> = graph
        .nodes
        .iter()
        .map(|node| to_i64(node.community))
        .collect();
    let label_arr = Array1::from(labels);

    let canonical: Vec<(i64, i64)> = graph
        .edges
        .iter()
        .map(|edge| (to_i64(edge.source), to_i64(edge.target)))
        .collect();
    let (flat, total_directed) = build_bidirectional_edge_flat(&canonical);
    let edge_arr = Array2::from_shape_vec((2, total_directed), flat)
        .map_err(|error| PyValueError::new_err(format!("edge_index shape error: {error}")))?;

    let x_arr = match collect_node_features(&graph.nodes).map_err(|e| err(&e))? {
        Some((feat_dim, x_flat)) => {
            let arr = Array2::from_shape_vec((n_nodes, feat_dim), x_flat)
                .map_err(|error| PyValueError::new_err(format!("x shape error: {error}")))?;
            Some(arr.into_pyarray(py))
        }
        None => None,
    };

    Ok((edge_arr.into_pyarray(py), label_arr.into_pyarray(py), x_arr))
}

// ─── Config-driven dispatch ──────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (config_json))]
fn generate_from_config(py: Python<'_>, config_json: &str) -> PyResult<String> {
    py.allow_threads(|| generate_from_config_native(config_json))
        .map_err(|e| err(&e))
}

// ─── Top-level SLPA convenience re-export ────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (n_nodes, edges, seed=0, rounds=100, threshold=0.2))]
fn overlapping_communities(
    py: Python<'_>,
    n_nodes: usize,
    edges: Vec<(usize, usize)>,
    seed: u64,
    rounds: usize,
    threshold: f64,
) -> Vec<Vec<usize>> {
    py.allow_threads(|| {
        plekton_fit::detect_overlapping(&edges, n_nodes, seed, rounds, threshold).communities
    })
}

// ─── Module registration ─────────────────────────────────────────────────────

#[pymodule]
fn _plekton(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_sbm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_sbm_from_fit, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ba, m)?)?;
    m.add_function(wrap_pyfunction!(generate_csbm, m)?)?;
    m.add_function(wrap_pyfunction!(graph_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(json_to_pyg, m)?)?;
    m.add_function(wrap_pyfunction!(generate_from_config, m)?)?;
    m.add_function(wrap_pyfunction!(overlapping_communities, m)?)?;

    mixing_py::register(m)?;
    graph_sparsify_py::register(py, m)?;
    fit_py::register(py, m)?;
    metrics_py::register(py, m)?;

    Ok(())
}
