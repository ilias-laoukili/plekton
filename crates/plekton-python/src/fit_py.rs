// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! `plekton.fit` bindings — DC-SBM blockmodel estimation and community detection.

#![allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 hands #[pyfunction] owned PyReadonly array guards by value"
)]

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use plekton_fit::{compact, detect_label_propagation, detect_overlapping, fit, FitNote, Partition};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyModule},
    Bound,
};

use crate::edges_py::{read_edges, read_labels};

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[pyfunction]
#[pyo3(signature = (edges, n_nodes, labels, attribute, seed, max_rounds))]
fn fit_blockmodel<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    labels: Option<PyReadonlyArray1<'py, i64>>,
    attribute: &str,
    seed: u64,
    max_rounds: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    let partition = match labels {
        Some(ref arr) => {
            let raw = read_labels(arr)?;
            let compacted = compact(&raw);
            Partition::Declared {
                attribute: attribute.to_owned(),
                labels: compacted,
            }
        }
        None => {
            py.allow_threads(|| detect_label_propagation(&edge_pairs, n_nodes, seed, max_rounds))
        }
    };

    let provenance = partition.provenance();
    let part_labels: Vec<i64> = partition.labels().iter().map(|&l| to_i64(l)).collect();
    let converged: Option<bool> = match &partition {
        Partition::Declared { .. } => None,
        Partition::Detected { detection, .. } => match *detection {
            plekton_fit::Detection::LabelPropagation { converged, .. } => Some(converged),
        },
    };

    let fitted = py
        .allow_threads(|| fit(&edge_pairs, n_nodes, partition))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let n_blocks = fitted.n_blocks();
    let blocks_per_node = fitted.blocks_per_node();
    let b_arr = Array2::from_shape_vec((n_blocks, n_blocks), fitted.b_matrix)
        .map_err(|e| PyValueError::new_err(format!("b_matrix shape error: {e}")))?;
    let theta_arr = Array1::from(fitted.theta);
    let labels_arr = Array1::from(part_labels);

    let notes_list = PyList::empty(py);
    for note in &fitted.notes {
        let item = PyDict::new(py);
        match *note {
            FitNote::BlockHasNoEdges { block, size } => {
                item.set_item("kind", "block_has_no_edges")?;
                item.set_item("block", block)?;
                item.set_item("size", size)?;
            }
            FitNote::BlockOfOne { block } => {
                item.set_item("kind", "block_of_one")?;
                item.set_item("block", block)?;
            }
        }
        notes_list.append(item)?;
    }

    let out = PyDict::new(py);
    out.set_item("block_sizes", fitted.block_sizes)?;
    out.set_item("b_matrix", b_arr.into_pyarray(py))?;
    out.set_item("theta", theta_arr.into_pyarray(py))?;
    out.set_item("labels", labels_arr.into_pyarray(py))?;
    out.set_item("n_nodes", fitted.n_nodes)?;
    out.set_item("n_edges", fitted.n_edges)?;
    out.set_item("n_blocks", n_blocks)?;
    out.set_item("blocks_per_node", blocks_per_node)?;
    out.set_item("provenance", provenance)?;
    out.set_item("converged", converged)?;
    out.set_item("notes", notes_list)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (edges, n_nodes, seed, max_rounds))]
fn label_propagation<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    seed: u64,
    max_rounds: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, bool)> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    let partition =
        py.allow_threads(|| detect_label_propagation(&edge_pairs, n_nodes, seed, max_rounds));
    let converged = match &partition {
        Partition::Detected {
            detection: plekton_fit::Detection::LabelPropagation { converged, .. },
            ..
        } => *converged,
        Partition::Declared { .. } => true,
    };
    let arr = Array1::from(
        partition
            .labels()
            .iter()
            .map(|&l| to_i64(l))
            .collect::<Vec<_>>(),
    );
    Ok((arr.into_pyarray(py), converged))
}

#[pyfunction]
#[pyo3(signature = (edges, n_nodes, seed, rounds, threshold))]
fn overlapping_communities<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    n_nodes: usize,
    seed: u64,
    rounds: usize,
    threshold: f64,
) -> PyResult<Vec<Vec<usize>>> {
    // Same bounds check as every other binding: an id outside the graph is an
    // error here, not an edge the detector quietly skips.
    let edge_pairs = read_edges(&edges, n_nodes)?;
    Ok(py.allow_threads(|| {
        detect_overlapping(&edge_pairs, n_nodes, seed, rounds, threshold).communities
    }))
}

pub fn register<'py>(
    py: Python<'py>,
    parent: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new(py, "fit")?;
    m.add_function(wrap_pyfunction!(fit_blockmodel, &m)?)?;
    m.add_function(wrap_pyfunction!(label_propagation, &m)?)?;
    m.add_function(wrap_pyfunction!(overlapping_communities, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(m)
}
