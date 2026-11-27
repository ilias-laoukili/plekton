// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Shared graph-metric primitives for the `plekton` workspace.
//!
//! This crate provides canonical, reusable implementations of structural
//! metrics over plain edge lists with zero coupling to any graph type.
//!
//! # Modules
//!
//! | Module | Metric |
//! |---|---|
//! | [`components`]     | Connected-component count (BFS, O(V + E)) |
//! | [`homophily`]      | Edge homophily — fraction of intra-community edges |
//! | [`clustering`]     | Mean local clustering coefficient (`O(E · max_degree)`) |
//! | [`transitivity`]   | Global clustering coefficient / transitivity |
//! | [`degree_distribution`] | Degree sequence, its summary, and a two-sample KS distance |
//! | [`density`]        | Graph density — ratio of edges to maximum possible |
//! | [`assortativity`]  | Degree assortativity (Newman's r) |
//! | [`mixing`]         | Attribute mixing: within share, random rate, Newman's r |
//! | [`similarity`]     | Per-edge similarity scores (Jaccard, Adamic-Adar, …) |
//! | [`skipped`]        | `SkippedMetric` — why a metric came back `None` |
//!
//! All functions operate on a plain `&[(usize, usize)]` edge list and a
//! `usize` node count, with no coupling to any specific graph type.

// ── Pragmatic clippy exemptions (Fortress resolution, graph-metrics) ─────────

pub mod assortativity;
pub mod clustering;
pub mod components;
pub mod degree_distribution;
pub mod density;
pub mod homophily;
pub mod mixing;
pub mod similarity;
pub mod skipped;
pub mod transitivity;

/// `usize` count -> `f64`, exact for every count below 2^53.
///
/// It used to go through `u32` and saturate at `u32::MAX`, so any count from
/// 2^32 up came out wrong; `f64` holds every integer up to 2^53 exactly, far
/// beyond any node or edge count.
#[inline]
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "exact below 2^53, which no node or edge count gets near"
)]
pub(crate) const fn count_to_f64(n: usize) -> f64 {
    n as f64
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn counts_above_u32_max_stay_exact() {
        let count: usize = (1 << 32) + 1;
        assert_eq!(
            super::count_to_f64(count).to_bits(),
            4_294_967_297.0_f64.to_bits()
        );
    }
}
