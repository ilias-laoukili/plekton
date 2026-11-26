// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Fixed-seed goldens for the two randomised detectors: label propagation and
//! SLPA.
//!
//! Both draw from a `rand_chacha::ChaCha8Rng` built with `seed_from_u64(seed)`.
//! Label propagation shuffles the node visiting order once per round; SLPA does
//! too, and also draws every spoken label and every tie. The other tests check
//! that a seed is reproducible and that the outcome has the right shape, but none
//! pins the outcome itself, so a change to the generator or to the order it is
//! consumed in would pass them all. These goldens pin it.
//!
//! The input is a 6 × 6 grid built here, not loaded. It has no community
//! structure for the detectors to agree on, so what they return depends on the
//! draws. A ring would not do: label propagation floods a 30-node ring with a
//! single label for every one of 2 000 seeds. Seed 7 is used because its
//! label-propagation partition has four blocks and was unique among the first
//! 1 000 seeds when frozen; seed 42, by contrast, gives the most common outcome,
//! one block, which about one seed in five produces.
//!
//! Rule, as for `crates/plekton-core/tests/determinism_golden.rs`: any change to
//! these constants must be a deliberate, reviewed re-freeze, never a silent drift.

use plekton_fit::{detect_label_propagation, detect_overlapping, Detection, Partition};

const GRID_SIDE: usize = 6;
const N_NODES: usize = GRID_SIDE * GRID_SIDE;
const SEED: u64 = 7;

const LP_MAX_ROUNDS: usize = 30;
const LP_ROUNDS_USED: usize = 5;
const LP_LABELS: &[usize] = &[
    0, 0, 1, 1, 1, 1, //
    2, 2, 1, 1, 1, 1, //
    2, 2, 1, 1, 1, 1, //
    2, 2, 1, 1, 1, 1, //
    2, 2, 2, 2, 3, 3, //
    2, 2, 2, 2, 3, 3,
];

const SLPA_ROUNDS: usize = 100;
const SLPA_THRESHOLD: f64 = 0.2;
const SLPA_COMMUNITIES: &[&[usize]] = &[
    &[0, 1, 6, 7],
    &[
        2, 3, 4, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 20, 21, 22, 26, 27,
    ],
    &[5, 11],
    &[7, 12, 13, 18, 19, 24, 25, 30, 31],
    &[12, 13, 18, 19, 24, 25, 26, 30, 31, 32],
    &[22, 23, 27, 28, 29],
    &[34, 35],
];

/// The 4-neighbour grid, edges listed row by row: right neighbour, then down.
fn grid(side: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for row in 0..side {
        for col in 0..side {
            let node = row * side + col;
            if col + 1 < side {
                edges.push((node, node + 1));
            }
            if row + 1 < side {
                edges.push((node, node + side));
            }
        }
    }
    edges
}

#[test]
fn label_propagation_golden() {
    let edges = grid(GRID_SIDE);
    let partition = detect_label_propagation(&edges, N_NODES, SEED, LP_MAX_ROUNDS);
    assert_eq!(
        partition,
        Partition::Detected {
            detection: Detection::LabelPropagation {
                seed: SEED,
                rounds_used: LP_ROUNDS_USED,
                converged: true,
            },
            labels: LP_LABELS.to_vec(),
        },
        "label propagation drifted"
    );

    // Same seed twice ⇒ identical.
    assert_eq!(
        partition,
        detect_label_propagation(&edges, N_NODES, SEED, LP_MAX_ROUNDS)
    );
    // The golden exercises the generator: the next seed gives another partition.
    let next = detect_label_propagation(&edges, N_NODES, SEED + 1, LP_MAX_ROUNDS);
    assert_ne!(
        next.labels(),
        LP_LABELS,
        "the golden does not depend on the seed"
    );
}

#[test]
fn slpa_golden() {
    let edges = grid(GRID_SIDE);
    let found = detect_overlapping(&edges, N_NODES, SEED, SLPA_ROUNDS, SLPA_THRESHOLD);
    let expected: Vec<Vec<usize>> = SLPA_COMMUNITIES.iter().map(|c| c.to_vec()).collect();
    assert_eq!(found.communities, expected, "SLPA communities drifted");
    for (index, members) in found.communities.iter().enumerate() {
        for &node in members {
            assert!(
                found.memberships[node].contains(&index),
                "node {node} is in community {index} but its memberships say {:?}",
                found.memberships[node]
            );
        }
    }

    // Same seed twice ⇒ identical.
    assert_eq!(
        found,
        detect_overlapping(&edges, N_NODES, SEED, SLPA_ROUNDS, SLPA_THRESHOLD)
    );
    // The golden exercises the generator: the next seed gives other communities.
    let next = detect_overlapping(&edges, N_NODES, SEED + 1, SLPA_ROUNDS, SLPA_THRESHOLD);
    assert_ne!(
        next.communities, expected,
        "the golden does not depend on the seed"
    );
}
