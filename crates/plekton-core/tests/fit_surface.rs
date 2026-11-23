// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
// Integration-test scaffolding: production-grade lints (panic-on-setup, numeric
// casts, exact float compares, literal grouping) are noise in tests, not signal.
//! The surface a fitted generator needs: a supplied mixing matrix, supplied block
//! sizes, and a supplied θ.
//!
//! Every test here answers one question: **can a fit be handed back to the
//! generator and come out as the thing that was fitted?** Before this surface
//! existed the answer was no — `generate_sbm_core` took two scalars and drew its
//! own θ, so "fit your generator" would have been a report with nowhere to go.

use plekton::{
    generate_sbm_core, generate_sbm_from_fit, BlockAssignment, BlockConnection, DegreeCorrection,
    SbmFit,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A K×K matrix with `p_in` on the diagonal and `p_out` everywhere else — the
/// matrix form of what `BlockConnection::Uniform` expresses with two scalars.
fn uniform_matrix(k: usize, p_in: f64, p_out: f64) -> Vec<f64> {
    (0..k * k)
        .map(|flat| if flat / k == flat % k { p_in } else { p_out })
        .collect()
}

fn block_sizes_of(labels: &[usize], k: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; k];
    for &label in labels {
        sizes[label] += 1;
    }
    sizes
}

// ── The matrix arm reduces to the scalar arm ──────────────────────────────────

#[test]
fn a_uniform_matrix_generates_exactly_what_two_scalars_do() -> TestResult {
    let matrix = uniform_matrix(4, 0.6, 0.05);
    let by_matrix = generate_sbm_from_fit(
        40,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 4 },
            connection: BlockConnection::Matrix(&matrix),
            degree_correction: DegreeCorrection::Off,
        },
        Some(11),
    )?;
    let by_scalars = generate_sbm_core(40, 4, 0.6, 0.05, None, Some(11))?;

    assert_eq!(
        by_matrix.canonical_edges, by_scalars.canonical_edges,
        "a matrix that says what p_in/p_out say must sample the same graph, or the \
         matrix arm is a second generator wearing the first one's name"
    );
    assert_eq!(by_matrix.labels, by_scalars.labels);
    Ok(())
}

#[test]
fn a_matrix_expresses_what_two_scalars_cannot() -> TestResult {
    // Block 0 talks to block 2 and to nothing else. No (p_in, p_out) pair says this.
    let matrix = vec![
        0.0, 0.0, 0.9, //
        0.0, 0.9, 0.0, //
        0.9, 0.0, 0.0,
    ];
    let result = generate_sbm_from_fit(
        60,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 3 },
            connection: BlockConnection::Matrix(&matrix),
            degree_correction: DegreeCorrection::Off,
        },
        Some(3),
    )?;
    let labels = &result.labels;
    let mut block_pair_counts = std::collections::BTreeMap::<(usize, usize), usize>::new();
    for (src, dst) in &result.canonical_edges {
        let (src, dst) = (usize::try_from(*src)?, usize::try_from(*dst)?);
        let (lo, hi) = (labels[src].min(labels[dst]), labels[src].max(labels[dst]));
        *block_pair_counts.entry((lo, hi)).or_default() += 1;
    }
    let zero_one = block_pair_counts.get(&(0, 1)).copied().unwrap_or(0);
    let zero_two = block_pair_counts.get(&(0, 2)).copied().unwrap_or(0);
    assert_eq!(zero_one, 0, "B[0,1] = 0 must produce no 0-1 edges");
    assert!(zero_two > 0, "B[0,2] = 0.9 must produce 0-2 edges");
    Ok(())
}

// ── Supplied block sizes ──────────────────────────────────────────────────────

#[test]
fn supplied_block_sizes_are_the_sizes_generated() -> TestResult {
    let sizes = vec![50usize, 7, 3];
    let matrix = uniform_matrix(3, 0.3, 0.02);
    let result = generate_sbm_from_fit(
        60,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&sizes),
            connection: BlockConnection::Matrix(&matrix),
            degree_correction: DegreeCorrection::Off,
        },
        Some(5),
    )?;
    assert_eq!(
        block_sizes_of(&result.labels, 3),
        sizes,
        "a fitted partition that comes back re-balanced is not the partition that was fitted"
    );
    Ok(())
}

#[test]
fn block_sizes_that_do_not_cover_the_nodes_are_refused() -> TestResult {
    let sizes = vec![10usize, 10];
    let err = generate_sbm_from_fit(
        25,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&sizes),
            connection: BlockConnection::Uniform {
                p_in: 0.5,
                p_out: 0.1,
            },
            degree_correction: DegreeCorrection::Off,
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("sum to 20"),
        "the refusal must name the shortfall: {err}"
    );
    Ok(())
}

#[test]
fn an_empty_block_is_refused_by_name() -> TestResult {
    let sizes = vec![10usize, 0, 10];
    let err = generate_sbm_from_fit(
        20,
        &SbmFit {
            assignment: BlockAssignment::Sizes(&sizes),
            connection: BlockConnection::Uniform {
                p_in: 0.5,
                p_out: 0.1,
            },
            degree_correction: DegreeCorrection::Off,
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("block 1 has size 0"),
        "the refusal must name the block: {err}"
    );
    Ok(())
}

// ── Supplied θ ──────────────────────────────────────────────────────────────

#[test]
fn a_supplied_theta_of_ones_is_the_uncorrected_graph() -> TestResult {
    // The `Supplied` arm must consume no RNG. If it drew even one value, the
    // sampler downstream would land on different numbers and this would differ.
    let theta = vec![1.0f64; 40];
    let corrected = generate_sbm_from_fit(
        40,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 4 },
            connection: BlockConnection::Uniform {
                p_in: 0.5,
                p_out: 0.05,
            },
            degree_correction: DegreeCorrection::Supplied(&theta),
        },
        Some(9),
    )?;
    let plain = generate_sbm_core(40, 4, 0.5, 0.05, None, Some(9))?;
    assert_eq!(
        corrected.canonical_edges, plain.canonical_edges,
        "a supplied theta of all ones must consume no RNG and change nothing"
    );
    Ok(())
}

#[test]
fn rescaling_a_supplied_theta_changes_nothing() -> TestResult {
    // Only theta's shape within a block is identifiable: the sampler uses
    // B[a,b]·theta_i·theta_j, so any per-block rescaling is absorbed by B. If
    // this failed, a fitter would have to guess the generator's normalisation.
    let base: Vec<f64> = (0..40).map(|node| 0.5 + f64::from(node % 7)).collect();
    let scaled: Vec<f64> = base.iter().map(|weight| weight * 3.0).collect();
    let fit_of = |theta: &[f64]| {
        generate_sbm_from_fit(
            40,
            &SbmFit {
                assignment: BlockAssignment::Balanced { n_communities: 4 },
                connection: BlockConnection::Uniform {
                    p_in: 0.5,
                    p_out: 0.05,
                },
                degree_correction: DegreeCorrection::Supplied(theta),
            },
            Some(4),
        )
    };
    assert_eq!(
        fit_of(&base)?.canonical_edges,
        fit_of(&scaled)?.canonical_edges,
        "theta and 3*theta must give the same graph"
    );
    Ok(())
}

#[test]
fn a_supplied_theta_actually_corrects_degrees() -> TestResult {
    // One node given 10x the weight of its block-mates must end up a hub. Without
    // this, `Supplied` could be silently ignored and every other test would pass.
    let mut theta = vec![1.0f64; 60];
    theta[0] = 20.0;
    let result = generate_sbm_from_fit(
        60,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 2 },
            connection: BlockConnection::Uniform {
                p_in: 0.02,
                p_out: 0.01,
            },
            degree_correction: DegreeCorrection::Supplied(&theta),
        },
        Some(2),
    )?;
    let mut degree = vec![0usize; 60];
    for (src, dst) in &result.canonical_edges {
        degree[usize::try_from(*src)?] += 1;
        degree[usize::try_from(*dst)?] += 1;
    }
    let others_max = degree[1..].iter().copied().max().unwrap_or(0);
    assert!(
        degree[0] > others_max,
        "the node weighted 20x should be the hub: got {} against a maximum of {others_max}",
        degree[0]
    );
    Ok(())
}

#[test]
fn a_theta_of_the_wrong_length_is_refused() -> TestResult {
    let theta = vec![1.0f64; 39];
    let err = generate_sbm_from_fit(
        40,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 4 },
            connection: BlockConnection::Uniform {
                p_in: 0.5,
                p_out: 0.05,
            },
            degree_correction: DegreeCorrection::Supplied(&theta),
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("one entry per node"),
        "the refusal must say what was expected: {err}"
    );
    Ok(())
}

#[test]
fn a_negative_theta_is_refused_with_its_index() -> TestResult {
    let mut theta = vec![1.0f64; 20];
    theta[7] = -0.5;
    let err = generate_sbm_from_fit(
        20,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 2 },
            connection: BlockConnection::Uniform {
                p_in: 0.5,
                p_out: 0.05,
            },
            degree_correction: DegreeCorrection::Supplied(&theta),
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("theta[7]"),
        "the refusal must name the entry: {err}"
    );
    Ok(())
}

#[test]
fn a_block_of_all_zero_theta_is_refused() -> TestResult {
    let mut theta = vec![1.0f64; 20];
    for weight in theta.iter_mut().take(10) {
        *weight = 0.0;
    }
    let err = generate_sbm_from_fit(
        20,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 2 },
            connection: BlockConnection::Uniform {
                p_in: 0.5,
                p_out: 0.05,
            },
            degree_correction: DegreeCorrection::Supplied(&theta),
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("no rescaling"),
        "the refusal must say why zero is not rescalable: {err}"
    );
    Ok(())
}

// ── Matrix validation ─────────────────────────────────────────────────────────

#[test]
fn an_asymmetric_matrix_is_refused_at_its_entry() -> TestResult {
    let matrix = vec![
        0.5, 0.2, //
        0.1, 0.5,
    ];
    let err = generate_sbm_from_fit(
        20,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 2 },
            connection: BlockConnection::Matrix(&matrix),
            degree_correction: DegreeCorrection::Off,
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("B[0,1]"),
        "the refusal must name the entry: {err}"
    );
    Ok(())
}

#[test]
fn a_matrix_of_the_wrong_shape_is_refused() -> TestResult {
    let matrix = vec![0.5, 0.2, 0.2];
    let err = generate_sbm_from_fit(
        20,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 2 },
            connection: BlockConnection::Matrix(&matrix),
            degree_correction: DegreeCorrection::Off,
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("got 3"),
        "the refusal must report the length it got: {err}"
    );
    Ok(())
}

#[test]
fn a_matrix_entry_outside_zero_one_is_refused() -> TestResult {
    let matrix = vec![
        1.5, 0.2, //
        0.2, 0.5,
    ];
    let err = generate_sbm_from_fit(
        20,
        &SbmFit {
            assignment: BlockAssignment::Balanced { n_communities: 2 },
            connection: BlockConnection::Matrix(&matrix),
            degree_correction: DegreeCorrection::Off,
        },
        Some(1),
    )
    .err()
    .ok_or("expected a refusal")?;
    assert!(
        err.to_string().contains("[0, 1]"),
        "the refusal must state the range: {err}"
    );
    Ok(())
}

// ── Determinism, which every claim above rests on ─────────────────────────────

#[test]
fn a_fit_generates_the_same_graph_twice() -> TestResult {
    let sizes = vec![30usize, 20, 10];
    let matrix = vec![
        0.30, 0.02, 0.01, //
        0.02, 0.40, 0.03, //
        0.01, 0.03, 0.50,
    ];
    let theta: Vec<f64> = (0..60).map(|node| 0.25 + f64::from(node % 5)).collect();
    let run = || {
        generate_sbm_from_fit(
            60,
            &SbmFit {
                assignment: BlockAssignment::Sizes(&sizes),
                connection: BlockConnection::Matrix(&matrix),
                degree_correction: DegreeCorrection::Supplied(&theta),
            },
            Some(1234),
        )
    };
    let (first, second) = (run()?, run()?);
    assert_eq!(first.canonical_edges, second.canonical_edges);
    assert_eq!(first.labels, second.labels);
    Ok(())
}
