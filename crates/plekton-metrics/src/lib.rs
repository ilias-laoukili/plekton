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

/// Lossless `usize` count -> `f64` (graph node/edge counts never approach 2^32),
/// used instead of a precision-losing `as` cast throughout the metric formulas.
#[inline]
#[must_use]
pub(crate) fn count_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}
