// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Edge-selection operators.
//!
//! Each operator takes per-edge `scores` (and, for [`degree_aware`], the graph
//! structure) plus a target `num_keep`, and returns a boolean keep-mask aligned
//! with the edge list. The [`crate::sparsify`] façade drives them and applies
//! the optional connectivity guarantee afterwards.

pub mod degree_aware;
pub mod sampled;
pub mod threshold;

/// Indices `0..scores.len()` sorted by **descending** score.
///
/// Ties break by ascending index so the ordering is fully deterministic (and
/// `f64::total_cmp` gives a total order even in the presence of `NaN`).
pub fn indices_by_score_desc(scores: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    indices
}
