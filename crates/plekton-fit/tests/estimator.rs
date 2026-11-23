// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
// Integration-test scaffolding: production-grade lints (panic-on-setup, numeric
// casts, exact float compares, literal grouping) are noise in tests, not signal.
//! What the estimator measures, and what it refuses to measure.

use plekton_fit::{
    detect_label_propagation, fit, Detection, Fit, FitError, FitNote, Partition, MAX_MIXING_ENTRIES,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn declared(labels: &[usize]) -> Partition {
    Partition::Declared {
        attribute: "test".to_owned(),
        labels: labels.to_vec(),
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

/// Two disjoint triangles, one per block.
fn two_triangles() -> (Vec<(usize, usize)>, Vec<usize>) {
    (
        vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        vec![0, 0, 0, 1, 1, 1],
    )
}

// ── The mixing matrix ─────────────────────────────────────────────────────────

#[test]
fn a_complete_bipartite_graph_reads_as_one_off_diagonal() -> TestResult {
    // K(3,3): every cross pair present, no within-block pair. B must be exactly
    // [[0, 1], [1, 0]] — anything else means the possible-pair denominator is wrong.
    let mut edges = Vec::new();
    for left in 0..3 {
        for right in 3..6 {
            edges.push((left, right));
        }
    }
    let fitted = fit(&edges, 6, declared(&[0, 0, 0, 1, 1, 1]))?;
    assert!(
        close(fitted.b_matrix[0], 0.0),
        "B[0,0] = {}",
        fitted.b_matrix[0]
    );
    assert!(
        close(fitted.b_matrix[1], 1.0),
        "B[0,1] = {}",
        fitted.b_matrix[1]
    );
    assert!(
        close(fitted.b_matrix[2], 1.0),
        "B[1,0] = {}",
        fitted.b_matrix[2]
    );
    assert!(
        close(fitted.b_matrix[3], 0.0),
        "B[1,1] = {}",
        fitted.b_matrix[3]
    );
    Ok(())
}

#[test]
fn two_triangles_read_as_full_within_and_empty_between() -> TestResult {
    let (edges, labels) = two_triangles();
    let fitted = fit(&edges, 6, declared(&labels))?;
    assert!(
        close(fitted.b_matrix[0], 1.0),
        "every within-block pair is present"
    );
    assert!(close(fitted.b_matrix[1], 0.0), "no edge crosses");
    assert_eq!(fitted.block_sizes, vec![3, 3]);
    assert_eq!(fitted.n_edges, 6);
    Ok(())
}

#[test]
fn the_mixing_matrix_is_symmetric() -> TestResult {
    let edges = vec![(0, 4), (1, 4), (2, 5), (3, 5), (0, 1), (4, 5)];
    let labels = [0, 0, 1, 1, 2, 2];
    let fitted = fit(&edges, 6, declared(&labels))?;
    let k = fitted.n_blocks();
    for row in 0..k {
        for col in 0..k {
            assert!(
                close(
                    fitted.b_matrix[row * k + col],
                    fitted.b_matrix[col * k + row]
                ),
                "B[{row},{col}] != B[{col},{row}]"
            );
        }
    }
    Ok(())
}

#[test]
fn a_duplicated_edge_is_counted_once() -> TestResult {
    let (mut edges, labels) = two_triangles();
    edges.push((1, 0)); // the same undirected pair, written the other way round
    edges.push((0, 1));
    edges.push((0, 0)); // and a self-loop, which the model has no term for
    let fitted = fit(&edges, 6, declared(&labels))?;
    assert_eq!(
        fitted.n_edges, 6,
        "a repeated pair inflates exactly the density being measured"
    );
    Ok(())
}

// ── The degree correction ─────────────────────────────────────────────────────

#[test]
fn theta_has_unit_mean_in_every_block() -> TestResult {
    // A star inside block 0 and a path inside block 1: both have uneven degrees,
    // so a normalisation bug would show as a block mean away from 1.
    let edges = vec![(0, 1), (0, 2), (0, 3), (4, 5), (5, 6), (6, 7)];
    let labels = [0, 0, 0, 0, 1, 1, 1, 1];
    let fitted = fit(&edges, 8, declared(&labels))?;
    let mut offset = 0;
    for (block, size) in fitted.block_sizes.iter().enumerate() {
        let mean: f64 = fitted.theta[offset..offset + size].iter().sum::<f64>()
            / f64::from(u32::try_from(*size)?);
        assert!(
            close(mean, 1.0),
            "block {block} has mean theta {mean}, not 1"
        );
        offset += size;
    }
    Ok(())
}

#[test]
fn theta_ranks_the_hub_above_its_leaves() -> TestResult {
    let edges = vec![(0, 1), (0, 2), (0, 3)];
    let fitted = fit(&edges, 4, declared(&[0, 0, 0, 0]))?;
    assert!(
        fitted.theta[0] > fitted.theta[1],
        "the centre of a star must carry the larger weight: {:?}",
        fitted.theta
    );
    Ok(())
}

#[test]
fn theta_comes_back_in_generator_node_order() -> TestResult {
    // Labels interleaved: node 0 -> block 1, node 1 -> block 0, and so on. The
    // generator lays blocks out contiguously, so theta must be permuted to match
    // or every weight lands on the wrong node.
    let edges = vec![(1, 3), (1, 5), (0, 2)];
    let labels = [1, 0, 1, 0, 1, 0];
    let fitted = fit(&edges, 6, declared(&labels))?;
    assert_eq!(fitted.block_sizes, vec![3, 3]);
    // Block 0 holds nodes 1, 3, 5 with degrees 2, 1, 1 -> mean 4/3.
    assert!(
        close(fitted.theta[0], 2.0 / (4.0 / 3.0)),
        "the first theta must belong to node 1, the first member of block 0: {:?}",
        fitted.theta
    );
    Ok(())
}

// ── Conditions reported rather than hidden ────────────────────────────────────

#[test]
fn a_block_with_no_edges_is_named() -> TestResult {
    let edges = vec![(0, 1), (1, 2)];
    let fitted = fit(&edges, 5, declared(&[0, 0, 0, 1, 1]))?;
    assert!(
        fitted
            .notes
            .contains(&FitNote::BlockHasNoEdges { block: 1, size: 2 }),
        "a uniform theta on an edgeless block is a stand-in, not a measurement: {:?}",
        fitted.notes
    );
    Ok(())
}

#[test]
fn a_block_of_one_node_is_named() -> TestResult {
    let edges = vec![(0, 1), (0, 2)];
    let fitted = fit(&edges, 3, declared(&[0, 0, 1]))?;
    assert!(
        fitted.notes.contains(&FitNote::BlockOfOne { block: 1 }),
        "B[1,1] = 0 there is arithmetic, not observation: {:?}",
        fitted.notes
    );
    Ok(())
}

// ── Refusals ──────────────────────────────────────────────────────────────────

#[test]
fn a_label_vector_that_misses_nodes_is_refused() -> TestResult {
    let err = fit(&[(0, 1)], 4, declared(&[0, 0, 1]))
        .err()
        .ok_or("expected a refusal")?;
    assert_eq!(
        err,
        FitError::LabelCount {
            labels: 3,
            n_nodes: 4
        }
    );
    Ok(())
}

#[test]
fn an_edge_outside_the_graph_is_refused() -> TestResult {
    let err = fit(&[(0, 9)], 4, declared(&[0, 0, 0, 0]))
        .err()
        .ok_or("expected a refusal")?;
    assert_eq!(
        err,
        FitError::EdgeOutOfRange {
            node: 9,
            n_nodes: 4
        }
    );
    Ok(())
}

#[test]
fn a_partition_finer_than_the_budget_is_refused_before_allocating() -> TestResult {
    // One node per block, just past the ceiling. The point is that this refuses
    // rather than asking for the allocation: a detection algorithm returns exactly
    // this shape on a graph with no structure.
    let side = 2_001usize;
    let labels: Vec<usize> = (0..side).collect();
    let err = fit(&[], side, declared(&labels))
        .err()
        .ok_or("expected a refusal")?;
    assert_eq!(
        err,
        FitError::MixingMatrixTooLarge {
            n_blocks: side,
            projected: side * side,
        }
    );
    assert!(side * side > MAX_MIXING_ENTRIES);
    Ok(())
}

#[test]
fn the_refusal_says_to_coarsen_rather_than_to_raise_the_ceiling() -> TestResult {
    let labels: Vec<usize> = (0..2_001).collect();
    let err = fit(&[], 2_001, declared(&labels))
        .err()
        .ok_or("expected a refusal")?;
    let text = err.to_string();
    assert!(
        text.contains("Coarsen the partition rather than raising the ceiling"),
        "the remedy must not be to raise the cap: {text}"
    );
    Ok(())
}

// ── The degenerate partition ──────────────────────────────────────────────────

#[test]
fn one_block_per_node_reports_a_ratio_of_one() -> TestResult {
    let labels: Vec<usize> = (0..6).collect();
    let fitted = fit(&[(0, 1), (2, 3)], 6, declared(&labels))?;
    assert!(
        close(fitted.blocks_per_node(), 1.0),
        "a partition this fine reproduces the input graph; the ratio is what catches it"
    );
    Ok(())
}

#[test]
fn a_coarse_partition_reports_a_small_ratio() -> TestResult {
    let (edges, labels) = two_triangles();
    let fitted = fit(&edges, 6, declared(&labels))?;
    assert!(close(fitted.blocks_per_node(), 2.0 / 6.0));
    Ok(())
}

// ── Label propagation ─────────────────────────────────────────────────────────

#[test]
fn label_propagation_separates_two_cliques() {
    let mut edges = Vec::new();
    for left in 0..8 {
        for right in (left + 1)..8 {
            edges.push((left, right));
        }
    }
    for left in 8..16 {
        for right in (left + 1)..16 {
            edges.push((left, right));
        }
    }
    edges.push((0, 8)); // one bridge
    let partition = detect_label_propagation(&edges, 16, 7, 20);
    assert_eq!(
        partition.n_blocks(),
        2,
        "two cliques joined by one edge are two blocks: {:?}",
        partition.labels()
    );
    let labels = partition.labels();
    assert!(labels[..8].iter().all(|label| *label == labels[0]));
    assert!(labels[8..].iter().all(|label| *label == labels[8]));
}

#[test]
fn label_propagation_is_deterministic_for_a_seed() {
    let edges: Vec<(usize, usize)> = (0..30).map(|node| (node, (node + 1) % 30)).collect();
    let first = detect_label_propagation(&edges, 30, 42, 20);
    let second = detect_label_propagation(&edges, 30, 42, 20);
    assert_eq!(first, second);
}

#[test]
fn a_partition_that_did_not_converge_says_so() -> TestResult {
    let edges: Vec<(usize, usize)> = (0..30).map(|node| (node, (node + 1) % 30)).collect();
    let partition = detect_label_propagation(&edges, 30, 1, 1);
    let Partition::Detected {
        detection: Detection::LabelPropagation { rounds_used, .. },
        ..
    } = partition
    else {
        return Err("expected a detected partition".into());
    };
    assert_eq!(rounds_used, 1);
    Ok(())
}

#[test]
fn an_isolated_node_keeps_its_own_block() {
    let partition = detect_label_propagation(&[(0, 1), (1, 2)], 4, 5, 10);
    let labels = partition.labels();
    assert_ne!(
        labels[3], labels[0],
        "nothing in the graph places node 3; putting it anywhere would be an invention"
    );
}

#[test]
fn a_declared_partition_says_which_attribute_it_came_from() -> TestResult {
    let (edges, labels) = two_triangles();
    let partition = Partition::Declared {
        attribute: "subnet".to_owned(),
        labels,
    };
    assert_eq!(partition.provenance(), "declared");
    let fitted: Fit = fit(&edges, 6, partition)?;
    let Partition::Declared { attribute, .. } = &fitted.partition else {
        return Err("the fit must carry the partition it used".into());
    };
    assert_eq!(attribute, "subnet");
    Ok(())
}

#[test]
fn a_detected_partition_names_its_algorithm() -> TestResult {
    let partition = detect_label_propagation(&[(0, 1)], 2, 3, 5);
    assert_eq!(partition.provenance(), "detected");
    let Partition::Detected { detection, .. } = partition else {
        return Err("expected a detected partition".into());
    };
    assert_eq!(detection.algorithm(), "label_propagation");
    Ok(())
}
