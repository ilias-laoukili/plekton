// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! `plekton.metrics` bindings — structural graph metrics over edge lists.

#![allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 hands #[pyfunction] owned PyReadonly array guards by value"
)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use plekton_metrics::{
    assortativity::degree_assortativity,
    clustering::compute_clustering_coefficient,
    components::count_components,
    degree_distribution::{degree_sequence, ks_distance, summarise},
    density::density as graph_density,
    homophily::compute_homophily,
    mixing::attribute_mixing as compute_attribute_mixing,
    transitivity::transitivity as compute_transitivity,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyModule},
    Bound,
};

use crate::edges_py::{read_edges, read_labels};

#[pyfunction]
fn homophily<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<f64>> {
    let lbls = read_labels(&labels)?;
    let edge_pairs = read_edges(&edges, lbls.len())?;
    Ok(py.allow_threads(|| compute_homophily(&edge_pairs, &lbls)))
}

#[pyfunction]
fn clustering<'py>(
    py: Python<'py>,
    n_nodes: usize,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<Option<f64>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    Ok(py.allow_threads(|| compute_clustering_coefficient(n_nodes, &edge_pairs)))
}

#[pyfunction]
fn transitivity<'py>(
    py: Python<'py>,
    n_nodes: usize,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<Option<f64>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    Ok(py.allow_threads(|| compute_transitivity(n_nodes, &edge_pairs)))
}

#[pyfunction]
fn assortativity<'py>(
    py: Python<'py>,
    n_nodes: usize,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<Option<f64>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    Ok(py.allow_threads(|| degree_assortativity(n_nodes, &edge_pairs)))
}

#[pyfunction]
fn components<'py>(
    py: Python<'py>,
    n_nodes: usize,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<usize> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    Ok(py.allow_threads(|| count_components(n_nodes, &edge_pairs)))
}

#[pyfunction]
fn density(n_nodes: usize, n_edges: usize) -> Option<f64> {
    graph_density(n_nodes, n_edges)
}

#[pyfunction]
fn attribute_mixing<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    labels: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let lbls = read_labels(&labels)?;
    let edge_pairs = read_edges(&edges, lbls.len())?;
    let measured = py.allow_threads(|| compute_attribute_mixing(&edge_pairs, &lbls));
    let Some(m) = measured else {
        return Ok(None);
    };
    let out = PyDict::new(py);
    out.set_item("within", m.within)?;
    out.set_item("random", m.random)?;
    out.set_item("r", m.r)?;
    Ok(Some(out))
}

#[pyfunction]
fn degree_summary<'py>(
    py: Python<'py>,
    n_nodes: usize,
    edges: PyReadonlyArray2<'py, i64>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let edge_pairs = read_edges(&edges, n_nodes)?;
    let summary = py.allow_threads(|| {
        let degs = degree_sequence(n_nodes, &edge_pairs);
        summarise(&degs)
    });
    let Some(s) = summary else {
        return Ok(None);
    };
    let out = PyDict::new(py);
    out.set_item("mean", s.mean)?;
    out.set_item("max", s.max)?;
    out.set_item("p50", s.p50)?;
    out.set_item("p90", s.p90)?;
    out.set_item("p99", s.p99)?;
    out.set_item("gini", s.gini)?;
    Ok(Some(out))
}

#[pyfunction]
fn degree_ks<'py>(
    py: Python<'py>,
    n_nodes_a: usize,
    edges_a: PyReadonlyArray2<'py, i64>,
    n_nodes_b: usize,
    edges_b: PyReadonlyArray2<'py, i64>,
) -> PyResult<Option<f64>> {
    let pairs_a = read_edges(&edges_a, n_nodes_a)?;
    let pairs_b = read_edges(&edges_b, n_nodes_b)?;
    if n_nodes_a == 0 || n_nodes_b == 0 {
        return Err(PyValueError::new_err(
            "both graphs must have at least one node to compare degree distributions",
        ));
    }
    Ok(py.allow_threads(|| {
        let deg_a = degree_sequence(n_nodes_a, &pairs_a);
        let deg_b = degree_sequence(n_nodes_b, &pairs_b);
        ks_distance(&deg_a, &deg_b)
    }))
}

pub fn register<'py>(
    py: Python<'py>,
    parent: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyModule>> {
    let m = PyModule::new(py, "metrics")?;
    m.add_function(wrap_pyfunction!(homophily, &m)?)?;
    m.add_function(wrap_pyfunction!(clustering, &m)?)?;
    m.add_function(wrap_pyfunction!(transitivity, &m)?)?;
    m.add_function(wrap_pyfunction!(assortativity, &m)?)?;
    m.add_function(wrap_pyfunction!(components, &m)?)?;
    m.add_function(wrap_pyfunction!(density, &m)?)?;
    m.add_function(wrap_pyfunction!(attribute_mixing, &m)?)?;
    m.add_function(wrap_pyfunction!(degree_summary, &m)?)?;
    m.add_function(wrap_pyfunction!(degree_ks, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(m)
}
