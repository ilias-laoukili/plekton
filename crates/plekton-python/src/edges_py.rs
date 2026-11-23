// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Shared `NumPy` edge-array and label-array readers for `PyO3` bindings.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    PyResult,
};

/// Read an `[E, 2]` int64 edge array into `(usize, usize)` pairs, validating
/// that every endpoint lies in `[0, n_nodes)`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "each index is bounds-checked as 0 <= idx < n_nodes before the cast"
)]
pub fn read_edges(
    edges: &PyReadonlyArray2<'_, i64>,
    n_nodes: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let view = edges.as_array();
    if view.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "edges must have shape [E, 2], got [_, {}]",
            view.ncols()
        )));
    }
    let bound = i64::try_from(n_nodes).unwrap_or(i64::MAX);
    let mut pairs = Vec::with_capacity(view.nrows());
    for row in view.rows() {
        let (u, v) = (row[0], row[1]);
        if u < 0 || v < 0 || u >= bound || v >= bound {
            return Err(PyIndexError::new_err(format!(
                "edge ({u}, {v}) out of bounds for n_nodes={n_nodes}"
            )));
        }
        pairs.push((u as usize, v as usize));
    }
    Ok(pairs)
}

/// Read a 1-D non-negative int64 label array into `Vec<usize>`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "each label is checked >= 0 before casting to usize"
)]
pub fn read_labels(labels: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    let view = labels.as_array();
    let mut out = Vec::with_capacity(view.len());
    for &lbl in view {
        if lbl < 0 {
            return Err(PyValueError::new_err(format!(
                "labels must be non-negative, got {lbl}"
            )));
        }
        out.push(lbl as usize);
    }
    Ok(out)
}
