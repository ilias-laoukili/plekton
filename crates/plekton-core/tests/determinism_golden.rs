// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
// Golden-test scaffolding: production-grade lints (panic-on-setup, numeric casts,
// exact float compares) are noise in fixed-seed assertions, not signal.
//! Fixed-seed full-output goldens for every generator (audit B1, the gate).
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
//! Captured once from a `golden_capture` example tool (since deleted) at the seeds
//! below; regenerate by temporarily restoring that tool if a generator's
//! contract is ever intentionally revised.

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
    (1, 2),
    (1, 3),
    (1, 11),
    (4, 6),
    (4, 7),
    (5, 6),
    (6, 9),
    (6, 10),
    (8, 11),
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
    assert!((r.effective_h - 0.666_666_666_666_666_6).abs() < 1e-12);

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
    (0, 1),
    (1, 2),
    (1, 3),
    (1, 6),
    (2, 9),
    (1, 8),
    (0, 13),
    (5, 6),
    (4, 7),
    (4, 8),
    (4, 12),
    (4, 13),
    (4, 14),
    (5, 14),
    (6, 13),
    (9, 10),
    (8, 10),
    (9, 14),
    (12, 13),
    (13, 14),
];
const SBM_DC_LABELS: &[usize] = &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];

#[test]
fn sbm_dc_golden() -> Result<(), Box<dyn std::error::Error>> {
    let r = generate_sbm_core(16, 4, 0.6, 0.1, Some(3.0), Some(7))?;
    assert_eq!(r.canonical_edges, SBM_DC_EDGES, "DC-SBM edges drifted");
    assert_eq!(r.labels, SBM_DC_LABELS, "DC-SBM labels drifted");
    assert!((r.effective_h - 0.45).abs() < 1e-12);
    Ok(())
}

// ── BA — generate_ba_core(12, 2, 2, 3, Some(42)) ─────────────────────────────

const BA_EDGES: &[(i64, i64)] = &[
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (0, 3),
    (0, 4),
    (2, 4),
    (4, 5),
    (1, 5),
    (3, 6),
    (2, 6),
    (4, 7),
    (6, 7),
    (5, 8),
    (6, 8),
    (0, 9),
    (3, 9),
    (1, 10),
    (3, 10),
    (10, 11),
    (0, 11),
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
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (0, 9),
    (2, 8),
    (2, 9),
    (3, 9),
    (4, 9),
    (5, 6),
    (6, 8),
    (6, 9),
    (7, 8),
    (7, 9),
    (8, 9),
];
const CSBM_GAUSS_LABELS: &[usize] = &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
const CSBM_GAUSS_XFLAT: &[f32] = &[
    0.259_209_13,
    0.042_492_22,
    -0.677_347,
    0.078_212_43,
    -0.385_764_36,
    0.030_644_94,
    1.027_896_6,
    0.987_736_2,
    0.648_283_84,
    0.088_507_18,
    1.291_327,
    -0.180_598_69,
    0.517_657_16,
    -0.680_796_86,
    0.611_200_33,
    -2.087_250_7,
    0.758_643_27,
    -0.201_007_9,
    -1.179_080_7,
    -0.553_441_76,
    0.264_838_2,
    2.501_431_2,
    1.787_367_3,
    0.055_590_26,
    0.131_629_29,
    -0.165_177_08,
    0.387_736_74,
    -2.152_816_8,
    0.377_019_2,
    0.283_628_94,
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
    assert!((r.effective_h - 0.75).abs() < 1e-12);
    Ok(())
}

// ── cSBM DC (θ normalization inside cSBM), p_triangle=0 ───────────────────────
// generate_csbm(n=12, K=3, h=0.6, d=4.0, feat_dim=3, mu=1.0, theta_exponent=3.0, seed=7)
// Re-frozen after upgrading DC-SBM to exact O(N + E) dyadic bucketing.

const CSBM_DC_EDGES: &[(i64, i64)] = &[
    (0, 2),
    (0, 1),
    (1, 2),
    (2, 3),
    (1, 3),
    (2, 6),
    (0, 7),
    (2, 7),
    (3, 4),
    (1, 7),
    (3, 7),
    (2, 8),
    (1, 10),
    (5, 6),
    (4, 7),
    (5, 7),
    (6, 7),
    (6, 8),
    (7, 11),
    (7, 8),
    (9, 11),
    (10, 11),
    (8, 10),
    (8, 11),
];
const CSBM_DC_LABELS: &[usize] = &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
const CSBM_DC_XFLAT: &[f32] = &[
    0.163_342_6,
    1.015_269_8,
    1.758_786_7,
    1.663_835_4,
    0.020_409_737,
    1.429_774,
    -0.688_591_8,
    0.803_830_44,
    0.875_597_6,
    0.654_548_94,
    -1.843_533_9,
    1.386_91,
    -0.968_590_56,
    1.242_621_8,
    -0.708_920_9,
    -1.097_479_1,
    0.113_769_31,
    -0.478_212_15,
    -0.897_173_7,
    -0.050_010_256,
    0.708_646_4,
    -1.751_602_5,
    0.551_142_2,
    0.756_085_63,
    0.922_982_04,
    -1.215_311_3,
    1.208_424_6,
    -0.974_207_76,
    -1.881_875,
    0.488_095_37,
    1.538_877_7,
    2.288_670_8,
    2.270_681,
    0.929_079_2,
    1.711_040_4,
    2.069_808_7,
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
    assert!((r.effective_h - 13.0 / 24.0).abs() < 1e-12);
    Ok(())
}

// ── cSBM p_triangle > 0 — the F-DET1 regression guard ────────────────────────
// Before the fix these constants did not exist and could not: the triangle-
// closing loop iterated a randomly-seeded `HashSet`, so the same seed produced a
// different edge list (and ordering) every process run. The values below are the
// frozen output of the now-deterministic implementation. If this test ever fails,
// the determinism fix has regressed (or the contract was deliberately re-frozen).

const CSBM_TRI_EDGES: &[(i64, i64)] = &[
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (0, 5),
    (1, 8),
    (2, 6),
    (3, 6),
    (3, 8),
    (4, 5),
    (2, 11),
    (2, 12),
    (4, 12),
    (5, 7),
    (5, 9),
    (6, 9),
    (8, 9),
    (6, 12),
    (6, 14),
    (7, 10),
    (9, 13),
    (10, 12),
    (10, 13),
    (11, 12),
    (11, 13),
    (11, 14),
    (12, 14),
    (4, 8),
    (1, 9),
    (2, 13),
    (2, 10),
    (0, 9),
    (2, 7),
    (8, 13),
    (6, 10),
    (6, 13),
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
    (1, 5),
    (3, 8),
    (3, 9),
    (4, 9),
    (1, 19),
    (5, 7),
    (5, 8),
    (7, 8),
    (6, 13),
    (5, 17),
    (10, 13),
    (12, 13),
    (13, 14),
    (10, 17),
    (13, 17),
    (14, 16),
    (15, 16),
    (15, 17),
    (16, 18),
    (0, 19),
    (0, 5),
    (8, 9),
    (3, 4),
    (5, 9),
    (5, 13),
    (12, 14),
    (13, 15),
    (14, 15),
    (16, 17),
    (5, 15),
    (0, 2),
    (11, 16),
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
