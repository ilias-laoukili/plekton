// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
// Dev micro-benchmark binary: panic-on-setup and timing/throughput casts are fine
// here — this is not shipped runtime code.
//! SBM micro-benchmark — measures `generate_sbm_core` wall-time at N=60211.
//!
//! Run with:
//!   cargo run --release -p plekton --bin `sbm_bench`
//!
//! Prints mean ± stddev over REPS repetitions.

use std::time::Instant;

/// Binary entry point — runs `REPETITIONS` SBM generations at fixed parameters
/// and prints wall-time statistics. Calibrates `p_in` / `p_out` so the expected
/// average degree is roughly 10 across the whole graph (see derivation below).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// Total node count — chosen to match the size of the OGB-Arxiv subgraph
    /// commonly used in homophily papers, so micro-benchmarks reflect realistic
    /// research-time loads.
    const NODE_COUNT: usize = 60_211;
    /// Number of communities; held constant so re-runs are comparable.
    const COMMUNITY_COUNT: usize = 4;
    /// Repetition count for mean/stddev — 10 is the smallest sample where
    /// stddev is meaningful while keeping the bench under a few seconds.
    const REPETITIONS: usize = 10;

    // `p_in` / `p_out` calibration so that avg_degree ≈ 10 across the full graph.
    //
    // For COMMUNITY_COUNT blocks of size NODE_COUNT / COMMUNITY_COUNT each:
    //   avg_degree ≈ (NODE_COUNT/COMMUNITY_COUNT - 1) · p_in
    //              + NODE_COUNT · (COMMUNITY_COUNT-1)/COMMUNITY_COUNT · p_out
    //
    // With p_out = p_in / 10 and COMMUNITY_COUNT=4, NODE_COUNT=60211:
    //   avg ≈ 15052 · p_in + 45157 · 0.1 · p_in
    //       = 19567.7 · p_in
    //   p_in ≈ 10 / 19567.7 ≈ 0.000_511
    let p_in: f64 = 0.000_511;
    let p_out: f64 = p_in / 10.0;

    // Warm-up — first call pays for allocator + cache priming; not measured.
    let _ = plekton::generate_sbm_core(NODE_COUNT, COMMUNITY_COUNT, p_in, p_out, None, Some(0));

    let mut sample_milliseconds: Vec<f64> = Vec::with_capacity(REPETITIONS);
    let mut total_edge_count: usize = 0;
    for repetition_index in 0..REPETITIONS {
        let start_instant = Instant::now();
        let result = plekton::generate_sbm_core(
            NODE_COUNT,
            COMMUNITY_COUNT,
            p_in,
            p_out,
            None,
            Some(repetition_index as u64),
        )?;
        let elapsed_seconds = start_instant.elapsed().as_secs_f64();
        sample_milliseconds.push(elapsed_seconds * 1_000.0);
        total_edge_count += result.canonical_edges.len();
    }

    let mean_milliseconds = sample_milliseconds.iter().sum::<f64>()
        / num_traits::ToPrimitive::to_f64(&(REPETITIONS)).unwrap_or(0.0);
    let variance_milliseconds = sample_milliseconds
        .iter()
        .map(|sample_ms| (sample_ms - mean_milliseconds).powi(2))
        .sum::<f64>()
        / num_traits::ToPrimitive::to_f64(&(REPETITIONS)).unwrap_or(0.0);
    let stddev_milliseconds = variance_milliseconds.sqrt();
    let mean_edge_count = total_edge_count / REPETITIONS;

    println!(
        "SBM benchmark  N={NODE_COUNT}  k={COMMUNITY_COUNT}  p_in={p_in:.6}  p_out={p_out:.6}",
    );
    println!("  reps        : {REPETITIONS}");
    println!("  avg edges   : {mean_edge_count}");
    println!("  mean time   : {mean_milliseconds:.3} ms");
    println!("  stddev      : {stddev_milliseconds:.3} ms");
    println!(
        "  min / max   : {:.3} / {:.3} ms",
        sample_milliseconds
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        sample_milliseconds
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    Ok(())
}
