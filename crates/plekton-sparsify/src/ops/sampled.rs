// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Weighted sampling without replacement (Efraimidis-Spirakis).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Floor applied to edge weights so a zero-scoring edge still has a (tiny)
/// non-zero chance of being kept.
const WEIGHT_FLOOR: f64 = 1e-8;

/// Sample exactly `num_keep` edges without replacement, with selection
/// probability proportional to edge score.
///
/// Uses the Efraimidis-Spirakis one-pass scheme: each edge `i` draws
/// `u_i ~ Uniform(0, 1)` and is assigned the key `ln(u_i) / w_i` (with
/// `w_i = max(score_i, 1e-8)`); the `num_keep` edges with the largest keys are
/// kept. Larger weights push the key toward `0` (its maximum), so high-scoring
/// edges are favoured exactly in proportion to their weight.
///
/// The `u_i` are drawn in edge order from a seeded [`ChaCha8Rng`], so the result
/// is bit-identical across platforms and `rand` versions for a given `seed`.
/// If `num_keep >= scores.len()` every edge is kept.
pub fn select(scores: &[f64], num_keep: usize, seed: u64) -> Vec<bool> {
    let n_edges = scores.len();
    if num_keep >= n_edges {
        return vec![true; n_edges];
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Draw one uniform per edge, in order, so the stream is deterministic.
    let mut keyed: Vec<(f64, usize)> = (0..n_edges)
        .map(|edge_index| {
            let weight = scores[edge_index].max(WEIGHT_FLOOR);
            // Clamp away from exactly 0.0 so ln() stays finite.
            let uniform: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
            (uniform.ln() / weight, edge_index)
        })
        .collect();

    // Largest keys win; ties break by ascending index for determinism.
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    let mut mask = vec![false; n_edges];
    for &(_, edge_index) in &keyed[..num_keep] {
        mask[edge_index] = true;
    }
    mask
}
