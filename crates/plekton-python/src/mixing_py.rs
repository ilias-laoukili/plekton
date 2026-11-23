// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Native `PyO3` binding for `mixing_graph` — zero-copy `NumPy` arrays out, no JSON round-trip.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use plekton::{
    generate_mixing_graph, generate_mixing_graph_json, MixingRequest, MixingTarget, StrengthSpec,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyModule},
    Bound,
};

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn parse_mixing_targets(dict: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, MixingTarget>> {
    let mut out = BTreeMap::new();
    for (k, v) in dict.iter() {
        let attr: String = k.extract()?;
        let spec = v.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!(
                "mixing['{attr}'] must be a dict with 'within' or 'matrix'"
            ))
        })?;
        let mut target = MixingTarget::default();
        for (sk, sv) in spec.iter() {
            let field: String = sk.extract()?;
            match field.as_str() {
                "within" => target.within = sv.extract()?,
                "matrix" => target.matrix = sv.extract()?,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown mixing target key '{other}' for attribute '{attr}'"
                    )));
                }
            }
        }
        out.insert(attr, target);
    }
    Ok(out)
}

fn parse_strength(dict: &Bound<'_, PyDict>) -> PyResult<StrengthSpec> {
    let mut mean: Option<f64> = None;
    let mut spread: f64 = 0.0;
    let mut same: BTreeMap<String, f64> = BTreeMap::new();
    for (k, v) in dict.iter() {
        let field: String = k.extract()?;
        match field.as_str() {
            "mean" => mean = Some(v.extract()?),
            "spread" => spread = v.extract()?,
            "same" => same = v.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown strength key '{other}' (expected mean, spread, same)"
                )));
            }
        }
    }
    let Some(mean) = mean else {
        return Err(PyValueError::new_err(
            "strength requires a 'mean' in (0, 1)",
        ));
    };
    Ok(StrengthSpec { mean, spread, same })
}

/// Native `mixing_graph`: accepts Python dicts/lists directly and returns a dict
/// with zero-copy `NumPy` `edges` (`int64[E, 2]`) and optional `strengths` (`float64[E]`).
#[pyfunction]
#[pyo3(signature = (
    nodes,
    attributes=None,
    groups=None,
    degrees=None,
    mean_degree=None,
    degree_spread=0.0,
    max_degree=None,
    mixing=None,
    clustering=None,
    strength=None,
    directed=false,
    popularity_spread=0.0,
    seed=None
))]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors every field of MixingRequest on one flat Python call surface"
)]
fn mixing_graph_native<'py>(
    py: Python<'py>,
    nodes: usize,
    attributes: Option<BTreeMap<String, Vec<String>>>,
    groups: Option<Vec<Option<String>>>,
    degrees: Option<Vec<usize>>,
    mean_degree: Option<f64>,
    degree_spread: f64,
    max_degree: Option<usize>,
    mixing: Option<&Bound<'py, PyDict>>,
    clustering: Option<f64>,
    strength: Option<&Bound<'py, PyDict>>,
    directed: bool,
    popularity_spread: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    let request = MixingRequest {
        nodes,
        seed,
        attributes: attributes.unwrap_or_default(),
        groups,
        degrees,
        mean_degree,
        degree_spread,
        max_degree,
        mixing: match mixing {
            Some(dict) => parse_mixing_targets(dict)?,
            None => BTreeMap::new(),
        },
        directed,
        popularity_spread,
        clustering,
        strength: match strength {
            Some(dict) => Some(parse_strength(dict)?),
            None => None,
        },
    };

    let result = py
        .allow_threads(|| generate_mixing_graph(&request))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let n_edges = result.edges.len();
    let mut flat_edges = Vec::with_capacity(n_edges * 2);
    for (u, v) in result.edges {
        flat_edges.push(to_i64(u));
        flat_edges.push(to_i64(v));
    }
    let edges_arr = Array2::from_shape_vec((n_edges, 2), flat_edges)
        .map_err(|e| PyValueError::new_err(format!("edges shape error: {e}")))?;

    let out = PyDict::new(py);
    out.set_item("edges", edges_arr.into_pyarray(py))?;
    out.set_item("directed", result.directed)?;
    if let Some(strengths) = result.strengths {
        out.set_item("strengths", Array1::from(strengths).into_pyarray(py))?;
    }
    out.set_item("wanted_degree", result.wanted_degree)?;
    out.set_item("mean_degree", result.mean_degree)?;

    let mixing_dict = PyDict::new(py);
    for (attr, report) in result.mixing {
        let entry = PyDict::new(py);
        entry.set_item("target", report.target)?;
        entry.set_item("within", report.within)?;
        entry.set_item("random", report.random)?;
        entry.set_item("r", report.r)?;
        mixing_dict.set_item(attr, entry)?;
    }
    out.set_item("mixing", mixing_dict)?;

    let clustering_dict = PyDict::new(py);
    clustering_dict.set_item("target", result.clustering.target)?;
    clustering_dict.set_item("transitivity", result.clustering.transitivity)?;
    out.set_item("clustering", clustering_dict)?;

    Ok(out)
}

/// JSON-in / JSON-out entry point for `mixing_graph` interoperability.
#[pyfunction]
#[pyo3(signature = (request_json))]
fn generate_mixing_graph_json_py(py: Python<'_>, request_json: &str) -> PyResult<String> {
    py.allow_threads(|| generate_mixing_graph_json(request_json))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mixing_graph_native, m)?)?;
    m.add_function(wrap_pyfunction!(generate_mixing_graph_json_py, m)?)?;
    Ok(())
}
