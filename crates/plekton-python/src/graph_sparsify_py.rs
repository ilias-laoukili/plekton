// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! `plekton.sparsify` bindings — edge scoring & graph sparsification.

#![allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 hands #[pyfunction] owned PyReadonly array guards by value"
)]

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use plekton_sparsify::{
    approx_er_scores, metric_backbone as backbone_select, proximity_to_distance, score_edges,
    sparsify, ApproxErConfig, Metric, SparsifyError, Strategy,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyModule, Bound};

use crate::edges_py::read_edges;

fn map_err(error: &SparsifyError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn read_features<'a>(
    features: &'a PyReadonlyArray2<'a, f32>,
    n_nodes: usize,
) -> PyResult<(&'a [f32], usize)> {
    let shape = features.shape();
    if shape[0] != n_nodes {
        return Err(PyValueError::new_err(format!(
            "features has {} rows but n_nodes={n_nodes}",
            shape[0]
        )));
    }
    let feat_dim = shape[1];
    if feat_dim == 0 {
        return Err(PyValueError::new_err("feat_dim must be non-zero"));
    }
    let flat = features.as_slice().map_err(|_| {
        PyValueError::new_err("features must be C-contiguous (call np.ascontiguousarray)")
    })?;
    Ok((flat, feat_dim))
}

#[pyfunction]
#[pyo3(signature = (edges, n_nodes, metric, features=None, ignore_self_loops=true))]
fn similarity_scores<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    metric: &str,
    features: Option<PyReadonlyArray2<'py, f32>>,
    ignore_self_loops: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let metric = match metric {
        "jaccard" => Metric::Jaccard,
        "adamic_adar" => Metric::AdamicAdar,
        "degree_product" => Metric::DegreeProduct,
        "feature_cosine" => Metric::FeatureCosine,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown metric '{other}' (expected jaccard, adamic_adar, degree_product, feature_cosine)"
            )));
        }
    };

    let edge_pairs = read_edges(&edges, n_nodes)?;
    let features = match features {
        Some(ref f) => Some(read_features(f, n_nodes)?),
        None => None,
    };
    let scores = py
        .allow_threads(|| {
            score_edges(
                metric,
                n_nodes,
                &edge_pairs,
                features.as_ref().map(|(flat, dim)| (*flat, *dim)),
                ignore_self_loops,
            )
        })
        .map_err(|e| map_err(&e))?;

    Ok(Array1::from(scores).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (edges, n_nodes, epsilon=0.3, seed=42, max_dim=None))]
fn approx_effective_resistance<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    epsilon: f64,
    seed: u64,
    max_dim: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    let config = ApproxErConfig {
        epsilon,
        seed,
        max_dim,
        ..ApproxErConfig::default()
    };
    let scores = py.allow_threads(|| approx_er_scores(n_nodes, &edge_pairs, &config));
    Ok(Array1::from(scores).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (
    edges, n_nodes, scores, strategy, retention_ratio,
    keep_lowest=false, min_per_node=1, seed=42, preserve_connectivity=true
))]
#[allow(
    clippy::too_many_arguments,
    reason = "one flat call surface per strategy family"
)]
fn select_mask<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    scores: PyReadonlyArray1<'py, f64>,
    strategy: &str,
    retention_ratio: f64,
    keep_lowest: bool,
    min_per_node: usize,
    seed: u64,
    preserve_connectivity: bool,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    let score_slice = scores
        .as_slice()
        .map_err(|_| PyValueError::new_err("scores must be C-contiguous"))?;

    let selection = match strategy {
        "threshold" => Strategy::Threshold {
            retention_ratio,
            keep_lowest,
        },
        "degree_aware" => Strategy::DegreeAware {
            retention_ratio,
            min_per_node,
        },
        "sampled" => Strategy::Sampled {
            retention_ratio,
            seed,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown strategy '{other}' (expected threshold, degree_aware, sampled)"
            )));
        }
    };

    let result = py
        .allow_threads(|| {
            sparsify(
                n_nodes,
                &edge_pairs,
                score_slice,
                &selection,
                preserve_connectivity,
            )
        })
        .map_err(|e| map_err(&e))?;

    Ok(Array1::from(result.kept_mask).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (edges, n_nodes, distances, epsilon=1e-9, distances_are_similarity=false))]
fn metric_backbone_mask<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    distances: PyReadonlyArray1<'py, f64>,
    epsilon: f64,
    distances_are_similarity: bool,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    let raw = distances
        .as_slice()
        .map_err(|_| PyValueError::new_err("distances must be C-contiguous"))?;

    let result = py
        .allow_threads(|| {
            let owned;
            let weights: &[f64] = if distances_are_similarity {
                owned = proximity_to_distance(raw, false);
                &owned
            } else {
                raw
            };
            backbone_select(n_nodes, &edge_pairs, weights, epsilon)
        })
        .map_err(|e| map_err(&e))?;

    Ok(Array1::from(result.kept_mask).into_pyarray(py))
}

pub fn register<'py>(
    py: Python<'py>,
    parent: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new(py, "sparsify")?;
    m.add_function(wrap_pyfunction!(similarity_scores, &m)?)?;
    m.add_function(wrap_pyfunction!(approx_effective_resistance, &m)?)?;
    m.add_function(wrap_pyfunction!(select_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(metric_backbone_mask, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(m)
}
