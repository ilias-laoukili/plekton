// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
// Golden-test scaffolding: production-grade lints (panic-on-setup, numeric casts,
// exact float compares) are noise in fixed-seed assertions, not signal.
//! Fixed-seed full-output goldens for every generator (audit B1, the gate).
//!
//! **Re-frozen for `ChaCha8Rng`.** The generators now draw from
//! `rand_chacha::ChaCha8Rng` instead of `rand::rngs::SmallRng`, so that a seed
//! gives the same graph on every platform: rand says `SmallRng` "should not be
//! considered reproducible due to dependence on platform and possible replacement
//! in future library versions" (it is Xoshiro256++ on 64-bit targets and
//! Xoshiro128++ on 32-bit ones). Every constant below was re-captured after that switch, which changed
//! nothing else — not the draws, not their order, not the sampling loops. The
//! previous values depended on `SmallRng` and cannot be reproduced with
//! `ChaCha8Rng`. The labels did not move: block layouts draw no random numbers.
//! What these goldens do and do not cover across platforms is stated in
//! `docs/models.md` ("Determinism").
//!
//! The crate's headline guarantee is "identical output for the same seed across
//! platforms" (`lib.rs` module docs). Before this file that guarantee was pinned
//! by *nothing executable end-to-end*: the in-crate tests check edge **counts**,
//! shapes, and properties (`effective_h ∈ [0,1]`, features finite) — none freeze
//! the actual emitted graph. A reorder of the block loop, a change to RNG
//! consumption order, or a relocation that silently perturbs sampling would pass
//! every existing test while changing the seeded output.
//!
//! These goldens close that gap. Each asserts the **exact** `canonical_edges` and
//! `labels` for a fixed seed (cSBM also pins `x_flat` within 1e-6). They cover the
//! plain-SBM, DC-SBM (θ path), BA, cSBM-gaussian, and cSBM-DC code paths — i.e.
//! every branch the B4 DRY extraction and the B5 module split touch.
//!
//! Rule, enforced here: any change to a seeded generator's output must be a
//! deliberate, reviewed re-freeze of these constants — **not** a silent drift. A
//! refactor that cannot reproduce these values is a behavior change, not a
//! relocation.
//!
//! Captured by a throwaway program that calls the generators at the seeds below
//! and prints these constants — first an in-repo `golden_capture` example (since
//! deleted), and for the `ChaCha8Rng` re-freeze a standalone crate outside the
//! repository that depends on this one by path. Re-capture the same way if a
//! generator's contract is ever intentionally revised.

use plekton::{generate_ba_core, generate_csbm, generate_sbm_core, CsbmConfig};

/// Maximum tolerated absolute error on a pinned `x_flat` feature value. Edges and
/// labels are integer-exact; features are f32 and compared with this slack to
/// absorb any last-bit platform rounding without masking a real reorder.
const FEAT_TOL: f32 = 1e-6;

fn assert_feats_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "x_flat length drifted");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= FEAT_TOL,
            "x_flat[{i}] drifted: got {g}, want {w} (|Δ|={})",
            (g - w).abs()
        );
    }
}

// ── SBM (plain) — generate_sbm_core(12, 3, 0.5, 0.1, None, Some(42)) ──────────

const SBM_PLAIN_EDGES: &[(i64, i64)] = &[
    (0, 1),
    (0, 2),
    (0, 3),
    (2, 3),
    (2, 6),
    (2, 10),
    (4, 5),
    (4, 6),
    (4, 7),
    (5, 6),
    (5, 11),
    (8, 10),
    (8, 11),
    (9, 10),
];
const SBM_PLAIN_LABELS: &[usize] = &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

#[test]
fn sbm_plain_golden() -> Result<(), Box<dyn std::error::Error>> {
    let r = generate_sbm_core(12, 3, 0.5, 0.1, None, Some(42))?;
    assert_eq!(
        r.canonical_edges, SBM_PLAIN_EDGES,
        "SBM plain edges drifted"
    );
    assert_eq!(r.labels, SBM_PLAIN_LABELS, "SBM plain labels drifted");
    assert!((r.effective_h.ok_or("no edge, so no homophily")? - 11.0 / 14.0).abs() < 1e-12);

    // Same seed twice ⇒ byte-identical.
    let r2 = generate_sbm_core(12, 3, 0.5, 0.1, None, Some(42))?;
    assert_eq!(r.canonical_edges, r2.canonical_edges);
    assert_eq!(r.labels, r2.labels);
    Ok(())
}

// ── DC-SBM (θ path) — generate_sbm_core(16, 4, 0.6, 0.1, Some(3.0), Some(7)) ──
// Re-frozen after upgrading DC-SBM from O(N²) single-ceiling rejection sampling
// to exact O(N + E) dyadic weight-bucketed geometric skip sampling.

const SBM_DC_EDGES: &[(i64, i64)] = &[
    (1, 2),
    (0, 3),
    (1, 3),
    (0, 1),
    (3, 10),
    (0, 14),
    (0, 13),
    (5, 7),
    (4, 6),
    (5, 6),
    (6, 7),
    (6, 11),
    (6, 10),
    (4, 14),
    (4, 13),
    (5, 13),
    (9, 11),
    (9, 10),
    (10, 11),
    (8, 10),
    (8, 12),
    (8, 13),
    (12, 14),
    (14, 15),
    (13, 14),
];
const SBM_DC_LABELS: &[usize] = &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];

#[test]
fn sbm_dc_golden() -> Result<(), Box<dyn std::error::Error>> {
    let r = generate_sbm_core(16, 4, 0.6, 0.1, Some(3.0), Some(7))?;
    assert_eq!(r.canonical_edges, SBM_DC_EDGES, "DC-SBM edges drifted");
    assert_eq!(r.labels, SBM_DC_LABELS, "DC-SBM labels drifted");
    assert!((r.effective_h.ok_or("no edge, so no homophily")? - 15.0 / 25.0).abs() < 1e-12);
    Ok(())
}

// ── BA — generate_ba_core(12, 2, 2, 3, Some(42)) ─────────────────────────────

const BA_EDGES: &[(i64, i64)] = &[
    (0, 1),
    (1, 2),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 4),
    (3, 4),
    (4, 5),
    (1, 5),
    (4, 6),
    (2, 6),
    (3, 7),
    (1, 7),
    (0, 8),
    (1, 8),
    (1, 9),
    (3, 9),
    (7, 10),
    (2, 10),
    (1, 11),
    (8, 11),
];
const BA_LABELS: &[usize] = &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

#[test]
fn ba_golden() -> Result<(), Box<dyn std::error::Error>> {
    let r = generate_ba_core(12, 2, 2, 3, Some(42))?;
    assert_eq!(r.canonical_edges, BA_EDGES, "BA edges drifted");
    assert_eq!(r.labels, BA_LABELS, "BA labels drifted");
    Ok(())
}

// ── cSBM gaussian, p_triangle=0 ──────────────────────────────────────────────
// generate_csbm(n=10, K=2, h=0.7, d=4.0, feat_dim=3, mu=1.0, "gaussian", seed=42)

const CSBM_GAUSS_EDGES: &[(i64, i64)] = &[
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (3, 4),
    (0, 6),
    (0, 7),
    (1, 8),
    (3, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (5, 9),
    (6, 7),
    (6, 8),
    (7, 8),
];
const CSBM_GAUSS_LABELS: &[usize] = &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
const CSBM_GAUSS_XFLAT: &[f32] = &[
    1.645_770_5,
    -0.324_031_4,
    1.037_415_5,
    -0.191_874_53,
    -0.103_758_15,
    -0.460_407_14,
    -0.485_909_16,
    0.770_808_2,
    0.874_656_9,
    0.929_752_47,
    0.113_860_86,
    -0.388_095_9,
    0.841_769_75,
    1.134_007_7,
    0.245_135_47,
    -1.826_830_6,
    0.195_686_97,
    -1.938_794_3,
    -0.729_111_6,
    0.111_483_514,
    -1.850_351_1,
    -0.506_565_9,
    -0.304_385_13,
    0.338_030_6,
    0.488_909,
    -0.016_131_211,
    0.679_695_9,
    0.071_621_13,
    1.263_257_4,
    2.022_967_8,
];

#[test]
fn csbm_gaussian_p_triangle_zero_golden() -> Result<(), Box<dyn std::error::Error>> {
    let r = generate_csbm(&CsbmConfig {
        n_nodes: 10,
        n_communities: 2,
        homophily: 0.7,
        avg_degree: 4.0,
        feat_dim: 3,
        mu: 1.0,
        seed: Some(42),
        ..CsbmConfig::default()
    })?;
    assert_eq!(
        r.canonical_edges, CSBM_GAUSS_EDGES,
        "cSBM gaussian edges drifted"
    );
    assert_eq!(r.labels, CSBM_GAUSS_LABELS, "cSBM gaussian labels drifted");
    assert_feats_close(&r.x_flat, CSBM_GAUSS_XFLAT);
    assert!((r.effective_h.ok_or("no edge, so no homophily")? - 13.0 / 18.0).abs() < 1e-12);
    Ok(())
}

// ── cSBM DC (θ normalization inside cSBM), p_triangle=0 ───────────────────────
// generate_csbm(n=12, K=3, h=0.6, d=4.0, feat_dim=3, mu=1.0, theta_exponent=3.0, seed=7)
// Re-frozen after upgrading DC-SBM to exact O(N + E) dyadic bucketing.

const CSBM_DC_EDGES: &[(i64, i64)] = &[
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (0, 1),
    (2, 7),
    (3, 6),
    (0, 5),
    (0, 6),
    (1, 6),
    (4, 5),
    (5, 7),
    (4, 6),
    (5, 6),
    (6, 7),
    (7, 9),
    (4, 10),
    (6, 8),
    (6, 10),
    (8, 11),
    (10, 11),
    (8, 10),
];
const CSBM_DC_LABELS: &[usize] = &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
const CSBM_DC_XFLAT: &[f32] = &[
    -1.392_553_3,
    0.825_725_3,
    -0.103_968_285,
    -0.208_703_83,
    0.236_087_72,
    -1.209_826_6,
    -1.203_924_9,
    0.654_393_7,
    0.293_139_52,
    -0.136_738_09,
    -1.080_820_7,
    -0.680_160_8,
    -0.597_402_93,
    -1.214_115_7,
    -1.216_719_7,
    -0.718_434_5,
    -1.140_754_3,
    0.588_790_1,
    -0.079_437_26,
    1.681_389_5,
    1.601_043_3,
    -0.383_168_5,
    0.775_601_45,
    0.475_082_9,
    1.376_516_7,
    1.525_655_2,
    -0.444_414_2,
    0.446_257_2,
    -0.730_413_6,
    -0.918_697_66,
    -1.264_178,
    -0.721_830_5,
    -1.604_960_3,
    1.785_607_5,
    0.916_848_9,
    0.188_314_7,
];

#[test]
fn csbm_dc_p_triangle_zero_golden() -> Result<(), Box<dyn std::error::Error>> {
    let r = generate_csbm(&CsbmConfig {
        n_nodes: 12,
        n_communities: 3,
        homophily: 0.6,
        avg_degree: 4.0,
        feat_dim: 3,
        mu: 1.0,
        theta_exponent: Some(3.0),
        seed: Some(7),
        ..CsbmConfig::default()
    })?;
    assert_eq!(r.canonical_edges, CSBM_DC_EDGES, "cSBM DC edges drifted");
    assert_eq!(r.labels, CSBM_DC_LABELS, "cSBM DC labels drifted");
    assert_feats_close(&r.x_flat, CSBM_DC_XFLAT);
    assert!((r.effective_h.ok_or("no edge, so no homophily")? - 13.0 / 22.0).abs() < 1e-12);
    Ok(())
}

// ── cSBM p_triangle > 0 — the F-DET1 regression guard ────────────────────────
// Before the fix these constants did not exist and could not: the triangle-
// closing loop iterated a randomly-seeded `HashSet`, so the same seed produced a
// different edge list (and ordering) every process run. The values below are the
// frozen output of the now-deterministic implementation. If this test ever fails,
// the determinism fix has regressed (or the contract was deliberately re-frozen).

const CSBM_TRI_EDGES: &[(i64, i64)] = &[
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (3, 4),
    (3, 6),
    (3, 8),
    (2, 11),
    (3, 10),
    (5, 6),
    (5, 8),
    (5, 9),
    (6, 8),
    (7, 8),
    (7, 9),
    (8, 9),
    (7, 11),
    (8, 11),
    (10, 11),
    (10, 12),
    (10, 13),
    (10, 14),
    (11, 13),
    (12, 13),
    (12, 14),
    (13, 14),
    (2, 3),
    (3, 11),
    (8, 10),
];

#[test]
fn csbm_p_triangle_positive_golden() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = CsbmConfig {
        n_nodes: 15,
        n_communities: 3,
        homophily: 0.6,
        avg_degree: 4.0,
        feat_dim: 3,
        mu: 1.0,
        p_triangle: 0.5,
        seed: Some(11),
        ..CsbmConfig::default()
    };
    let r = generate_csbm(&cfg)?;
    assert_eq!(
        r.canonical_edges, CSBM_TRI_EDGES,
        "cSBM p_triangle>0 edges drifted — the F-DET1 determinism fix regressed"
    );

    // Same seed twice in one process ⇒ identical. Against the pre-fix code this
    // alone failed: two `HashSet`s in a process get distinct randomized orders.
    let r2 = generate_csbm(&cfg)?;
    assert_eq!(
        r.canonical_edges, r2.canonical_edges,
        "cSBM p_triangle>0 not reproducible"
    );
    Ok(())
}

// ── cSBM p_triangle > 0 AND ensure_connected — combined non-RNG-order path ────

const CSBM_TRI_CONN_EDGES: &[(i64, i64)] = &[
    (0, 1),
    (1, 3),
    (0, 6),
    (3, 9),
    (1, 14),
    (2, 19),
    (4, 19),
    (5, 7),
    (6, 9),
    (7, 9),
    (8, 9),
    (10, 13),
    (11, 12),
    (10, 17),
    (12, 19),
    (16, 19),
    (18, 19),
    (0, 14),
    (1, 9),
    (0, 9),
    (3, 14),
    (1, 6),
    (6, 7),
    (3, 8),
    (13, 17),
    (11, 19),
    (16, 18),
    (0, 2),
    (0, 10),
    (15, 18),
];

#[test]
fn csbm_p_triangle_and_ensure_connected_golden() -> Result<(), Box<dyn std::error::Error>> {
    let r = generate_csbm(&CsbmConfig {
        n_nodes: 20,
        n_communities: 4,
        homophily: 0.5,
        avg_degree: 2.0,
        feat_dim: 4,
        mu: 1.0,
        p_triangle: 1.0,
        ensure_connected: true,
        seed: Some(3),
        ..CsbmConfig::default()
    })?;
    assert_eq!(
        r.canonical_edges, CSBM_TRI_CONN_EDGES,
        "cSBM p_triangle>0 + ensure_connected edges drifted"
    );

    // ensure_connected actually delivers a single component.
    let n = r.n_nodes;
    let mut adj = vec![Vec::new(); n];
    for &(s, d) in &r.canonical_edges {
        adj[usize::try_from(s).unwrap_or(usize::MAX)]
            .push(usize::try_from(d).unwrap_or(usize::MAX));
        adj[usize::try_from(d).unwrap_or(usize::MAX)]
            .push(usize::try_from(s).unwrap_or(usize::MAX));
    }
    let mut seen = vec![false; n];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut count = 1;
    while let Some(u) = stack.pop() {
        for &v in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                count += 1;
                stack.push(v);
            }
        }
    }
    assert_eq!(count, n, "ensure_connected must yield one component");
    Ok(())
}
