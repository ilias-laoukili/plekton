// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Hard top-k / bottom-k threshold selection.

/// Keep exactly `num_keep` edges by score.
///
/// When `keep_lowest` is `false`, the `num_keep` highest-scoring edges are kept;
/// when `true`, the `num_keep` lowest-scoring edges are kept. Ties break by
/// ascending edge index for determinism. If `num_keep >= scores.len()` every
/// edge is kept.
pub fn select(scores: &[f64], num_keep: usize, keep_lowest: bool) -> Vec<bool> {
    let n_edges = scores.len();
    if num_keep >= n_edges {
        return vec![true; n_edges];
    }

    // Ascending order by score (ties by ascending index).
    let mut ascending: Vec<usize> = (0..n_edges).collect();
    ascending.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)));

    let selected = if keep_lowest {
        &ascending[..num_keep]
    } else {
        &ascending[n_edges - num_keep..]
    };

    let mut mask = vec![false; n_edges];
    for &edge_index in selected {
        mask[edge_index] = true;
    }
    mask
}
