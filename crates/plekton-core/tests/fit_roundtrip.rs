// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
// Integration-test scaffolding: production-grade lints (panic-on-setup, numeric
// casts, exact float compares, literal grouping) are noise in tests, not signal.
//! Generate with known parameters, fit the result, and check the parameters come
//! back.
//!
//! **This is the only test that catches an estimator that is confidently wrong.**
//! Every test inside `graph-fit` checks the estimator against arithmetic the same
//! author wrote; this one checks it against a sampler written years earlier, on a
//! graph the estimator did not choose.
//!
//! It lives in `plekton-core` rather than in `plekton-fit` because `plekton-fit`
//! keeps a zero internal dependency count, and only the test needs both ends.

use plekton::{generate_sbm_from_fit, BlockAssignment, BlockConnection, DegreeCorrection, SbmFit};
use plekton_fit::{fit, Partition};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Tolerance on a recovered mixing-matrix entry, as a fraction of the truth.
///
/// The estimate is a binomial proportion over `n_a×n_b` pairs. With the block
/// sizes below, the smallest cell has ~10 000 pairs at p = 0.02, so the standard
/// error is under 0.0015 — about 7 % of that entry. 20 % is comfortably outside
/// the noise and still tight enough that an off-by-a-factor estimator fails.
const B_RELATIVE_TOLERANCE: f64 = 0.20;

/// Tolerance on the recovered per-node θ, as a Pearson correlation floor.
///
/// θ is recovered from a single realisation of a degree, so the per-node value is
/// noisy by construction; what has to survive is the *ranking*. A generator whose
/// hubs came back as leaves would not clear 0.7.
const THETA_CORRELATION_FLOOR: f64 = 0.70;

/// Weight given to the seeded hubs in the degree-correction round trip.
const HUB_THETA: f64 = 8.0;

fn labels_from_sizes(sizes: &[usize]) -> Vec<usize> {
    let mut labels = Vec::new();
    for (block, size) in sizes.iter().enumerate() {
        labels.extend(std::iter::repeat_n(block, *size));
    }
    labels
}

fn declared(sizes: &[usize]) -> Partition {
    Partition::Declared {
        attribute: "the partition the graph was generated with".to_owned(),
        labels: labels_from_sizes(sizes),
    }
}

fn to_usize_edges(edges: &[(i64, i64)]) -> Result<Vec<(usize, usize)>, std::num::TryFromIntError> {
    edges
        .iter()
        .map(|(source, target)| Ok((usize::try_from(*source)?, usize::try_from(*target)?)))
        .collect()
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    let count = f64::from(u32::try_from(left.len()).unwrap_or(u32::MAX));
    if count < 2.0 {
        return 0.0;
    }
    let mean_l = left.iter().sum::<f64>() / count;
    let mean_r = right.iter().sum::<f64>() / count;
    let mut covariance = 0.0;
    let mut var_l = 0.0;
    let mut var_r = 0.0;
    for (value_l, value_r) in left.iter().zip(right) {
        let (delta_l, delta_r) = (value_l - mean_l, value_r - mean_r);
        covariance = delta_l.mul_add(delta_r, covariance);
        var_l = delta_l.mul_add(delta_l, var_l);
        var_r = delta_r.mul_add(delta_r, var_r);
    }
    if var_l <= 0.0 || var_r <= 0.0 {
        return 0.0;
    }
    covariance / (var_l * var_r).sqrt()
}

#[test]
fn a_mixing_matrix_survives_a_generate_then_fit_round_trip() -> TestResult {
    let sizes = vec![400usize, 300, 200];
    let n_nodes: usize = sizes.iter().sum();
    // Deliberately not homophilous everywhere: block 0 talks to block 2 more than
    // it talks to itself, which no (p_in, p_out) pair can express. If the round
    // trip only worked for uniform matrices it would be testing the old surface.
    let truth = vec![
        0.05, 0.02, 0.08, //
        0.02, 0.12, 0.03, //
        0.08, 0.03, 0.06,
    ];
    let generated = generate_sbm_from_fit(
        n_nodes,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&sizes),
            connection: BlockConnection::Matrix(&truth),
            degree_correction: DegreeCorrection::Off,
        },
        Some(20_260_901),
    )?;

    let fitted = fit(
        &to_usize_edges(&generated.canonical_edges)?,
        n_nodes,
        declared(&sizes),
    )?;

    assert_eq!(fitted.block_sizes, sizes);
    assert_eq!(fitted.n_blocks(), 3);
    for (index, (recovered, expected)) in fitted.b_matrix.iter().zip(&truth).enumerate() {
        let relative = (recovered - expected).abs() / expected;
        assert!(
            relative < B_RELATIVE_TOLERANCE,
            "B[{},{}]: generated with {expected}, recovered {recovered} \
             ({:.1} % off)",
            index / 3,
            index % 3,
            relative * 100.0
        );
    }
    Ok(())
}

#[test]
fn a_degree_correction_survives_a_generate_then_fit_round_trip() -> TestResult {
    let sizes = vec![500usize, 500];
    let n_nodes: usize = sizes.iter().sum();
    let truth = vec![
        0.08, 0.01, //
        0.01, 0.08,
    ];
    // A spread wide enough that a hub is unmistakable, with unit mean per block so
    // it is already in the sampler's gauge.
    let theta: Vec<f64> = (0..n_nodes)
        .map(|node| {
            let within = node % 500;
            if within < 25 {
                HUB_THETA
            } else {
                // Whatever keeps the block mean at 1: the 25 hubs already account
                // for 25 * HUB_THETA of the 500 units a unit-mean block must hold.
                25.0f64.mul_add(-HUB_THETA, 500.0) / 475.0
            }
        })
        .collect();

    let generated = generate_sbm_from_fit(
        n_nodes,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&sizes),
            connection: BlockConnection::Matrix(&truth),
            degree_correction: DegreeCorrection::Supplied(&theta),
        },
        Some(7_777),
    )?;

    let fitted = fit(
        &to_usize_edges(&generated.canonical_edges)?,
        n_nodes,
        declared(&sizes),
    )?;

    let correlation = pearson(&theta, &fitted.theta);
    assert!(
        correlation > THETA_CORRELATION_FLOOR,
        "theta went in and came back with correlation {correlation:.3}, below \
         {THETA_CORRELATION_FLOOR}"
    );

    // And the ranking, which is the part a generator actually consumes: the 25
    // seeded hubs of block 0 must dominate its 475 others.
    let hub_mean: f64 = fitted.theta[..25].iter().sum::<f64>() / 25.0;
    let rest_mean: f64 = fitted.theta[25..500].iter().sum::<f64>() / 475.0;
    assert!(
        hub_mean > rest_mean * 3.0,
        "hubs recovered at {hub_mean:.2} against {rest_mean:.2} for the rest"
    );
    Ok(())
}

#[test]
fn a_fit_regenerates_a_graph_of_the_same_shape() -> TestResult {
    // The whole loop: generate, fit, generate from the fit. The second graph is
    // not the first — it is a different draw — but its density and its block
    // structure must land close, or the fit carries nothing.
    let sizes = vec![300usize, 300];
    let n_nodes: usize = sizes.iter().sum();
    let truth = vec![
        0.10, 0.01, //
        0.01, 0.10,
    ];
    let original = generate_sbm_from_fit(
        n_nodes,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&sizes),
            connection: BlockConnection::Matrix(&truth),
            degree_correction: DegreeCorrection::Off,
        },
        Some(11),
    )?;
    let fitted = fit(
        &to_usize_edges(&original.canonical_edges)?,
        n_nodes,
        declared(&sizes),
    )?;

    let regenerated = generate_sbm_from_fit(
        n_nodes,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&fitted.block_sizes),
            connection: BlockConnection::Matrix(&fitted.b_matrix),
            degree_correction: DegreeCorrection::Supplied(&fitted.theta),
        },
        Some(12),
    )?;

    let original_edges = f64::from(u32::try_from(original.canonical_edges.len())?);
    let regenerated_edges = f64::from(u32::try_from(regenerated.canonical_edges.len())?);
    let relative = (regenerated_edges - original_edges).abs() / original_edges;
    assert!(
        relative < 0.15,
        "the regenerated graph has {regenerated_edges} edges against {original_edges} \
         ({:.1} % off)",
        relative * 100.0
    );
    Ok(())
}

#[test]
fn a_detected_partition_also_completes_the_loop() -> TestResult {
    // The declared arm hands the fitter the answer. This one does not: label
    // propagation has to find the blocks before anything can be estimated.
    let sizes = vec![150usize, 150];
    let n_nodes: usize = sizes.iter().sum();
    let truth = vec![
        0.20, 0.004, //
        0.004, 0.20,
    ];
    let generated = generate_sbm_from_fit(
        n_nodes,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&sizes),
            connection: BlockConnection::Matrix(&truth),
            degree_correction: DegreeCorrection::Off,
        },
        Some(31),
    )?;
    let edges = to_usize_edges(&generated.canonical_edges)?;
    let partition = plekton_fit::detect_label_propagation(&edges, n_nodes, 5, 30);
    let fitted = fit(&edges, n_nodes, partition)?;

    assert!(
        fitted.n_blocks() >= 2,
        "two well-separated blocks should not collapse to one: {} found",
        fitted.n_blocks()
    );
    assert!(
        fitted.blocks_per_node() < 0.10,
        "a detection that returns nearly one block per node has found nothing: {:.3}",
        fitted.blocks_per_node()
    );
    Ok(())
}
