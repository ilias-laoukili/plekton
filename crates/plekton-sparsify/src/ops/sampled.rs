// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Weighted sampling without replacement (Efraimidis-Spirakis).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Floor applied to edge weights so a zero-scoring edge still has a (tiny)
/// non-zero chance of being kept.
const WEIGHT_FLOOR: f64 = 1e-8;

/// The stream of `ChaCha8Rng::seed_from_u64(seed)` that sampling reads.
///
/// [`crate::approx_er`] draws its projection from stream 0, the default. Were
/// sampling on stream 0 too, one seed given to both — as
/// [`crate::sparsify_resistance_sampled`] does, and as a Python caller does by
/// passing the same `seed` to both functions — would make the sampling re-read
/// the very draws that produced the scores it samples by.
const SAMPLING_STREAM: u64 = 1;

/// The generator [`select`] draws from: `ChaCha8Rng::seed_from_u64(seed)`, moved
/// to [`SAMPLING_STREAM`].
fn sampling_rng(seed: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(SAMPLING_STREAM);
    rng
}

/// Sample exactly `num_keep` edges without replacement, favouring high scores.
///
/// Uses the Efraimidis-Spirakis one-pass scheme: each edge `i` draws
/// `u_i ~ Uniform(0, 1)` and is assigned the key `ln(u_i) / w_i` (with
/// `w_i = max(score_i, 1e-8)`); the `num_keep` edges with the largest keys are
/// kept. The result is distributed like `num_keep` successive draws without
/// replacement, each picking one of the remaining edges with probability
/// proportional to its weight. The probability that an edge ends up kept
/// therefore grows with its weight but is **not** proportional to it: it
/// saturates towards 1 for heavy edges, so an edge twice as heavy as another is
/// not kept twice as often.
///
/// The `u_i` are drawn in edge order from [`sampling_rng`], so the same `seed`
/// reproduces the same selection. That is not promised across minor versions of
/// `rand`, which may change sampled values. If `num_keep >= scores.len()` every
/// edge is kept.
pub fn select(scores: &[f64], num_keep: usize, seed: u64) -> Vec<bool> {
    let n_edges = scores.len();
    if num_keep >= n_edges {
        return vec![true; n_edges];
    }

    let mut rng = sampling_rng(seed);

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

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;

    #[test]
    fn sampling_reads_its_own_stream() {
        // The projection in `approx_er` reads stream 0 of `seed_from_u64(seed)`;
        // given the same seed, sampling must not see those draws.
        let mut sampling = sampling_rng(42);
        let mut projection = ChaCha8Rng::seed_from_u64(42);
        assert_eq!(sampling.get_stream(), SAMPLING_STREAM);
        assert_eq!(projection.get_stream(), 0);
        assert_ne!(sampling.next_u64(), projection.next_u64());
    }

    #[test]
    fn selection_golden() {
        // Pinned after sampling moved to stream 1, so a change of stream, of the
        // key formula or of the tie rule shows up here.
        let scores: Vec<f64> = (1..=12).map(|i| f64::from(i) / 10.0).collect();
        let kept: Vec<usize> = select(&scores, 6, 3)
            .iter()
            .enumerate()
            .filter_map(|(edge_index, &keep)| keep.then_some(edge_index))
            .collect();
        assert_eq!(kept, [2, 3, 4, 8, 10, 11]);
    }
}
