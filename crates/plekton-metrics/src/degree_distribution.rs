// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Degree-distribution comparison.
//!
//! Every other metric in this crate reduces a graph to one number. Two graphs can
//! agree on all of them and still have completely different degree distributions
//! — one regular, one scale-free — which is exactly the difference a blockmodel
//! with degree correction is supposed to capture.
//!
//! Nothing here existed before: comparing two graphs' degree distributions was
//! not something this workspace could do, so a fit could not be checked on the
//! quantity it most directly targets.

/// Per-node degree from a canonical undirected edge list.
///
/// Self-loops count once, matching the convention the rest of this crate uses.
#[must_use]
pub fn degree_sequence(n_nodes: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut degree = vec![0usize; n_nodes];
    for &(source, target) in edges {
        if source < n_nodes {
            degree[source] += 1;
        }
        if target < n_nodes && target != source {
            degree[target] += 1;
        }
    }
    degree
}

/// A degree distribution reduced to the handful of numbers worth comparing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegreeSummary {
    /// Mean degree.
    pub mean: f64,
    /// Largest degree — the hub, which is what degree correction exists for.
    pub max: usize,
    /// Median.
    pub p50: usize,
    /// 90th percentile.
    pub p90: usize,
    /// 99th percentile: where a heavy tail shows and the mean does not.
    pub p99: usize,
    /// Gini coefficient of the degrees, in `[0, 1]`. `0` is a regular graph;
    /// higher means degree is concentrated in fewer nodes.
    pub gini: f64,
}

/// Summarises a degree sequence. Returns `None` for an empty sequence, which has
/// no mean to report — not a mean of zero.
#[must_use]
pub fn summarise(degrees: &[usize]) -> Option<DegreeSummary> {
    if degrees.is_empty() {
        return None;
    }
    let mut sorted: Vec<usize> = degrees.to_vec();
    sorted.sort_unstable();
    let count = crate::count_to_f64(sorted.len());
    let total: usize = sorted.iter().sum();

    // Gini over a sorted sequence: (2*Σ i*x_i) / (n * Σ x_i) - (n + 1) / n.
    let gini = if total == 0 {
        0.0
    } else {
        let weighted: f64 = sorted
            .iter()
            .enumerate()
            .map(|(index, degree)| crate::count_to_f64(index + 1) * crate::count_to_f64(*degree))
            .sum();
        ((2.0 * weighted) / (count * crate::count_to_f64(total)) - (count + 1.0) / count)
            .clamp(0.0, 1.0)
    };

    Some(DegreeSummary {
        mean: crate::count_to_f64(total) / count,
        max: sorted.last().copied().unwrap_or(0),
        p50: percentile(&sorted, 0.50),
        p90: percentile(&sorted, 0.90),
        p99: percentile(&sorted, 0.99),
        gini,
    })
}

/// Nearest-rank percentile of an already-sorted sequence.
fn percentile(sorted: &[usize], fraction: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let last = sorted.len() - 1;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the product is clamped into 0..=last before the cast, and last is a \
                  slice index that already fits usize"
    )]
    let rank = (fraction * crate::count_to_f64(last)).round().max(0.0) as usize;
    sorted[rank.min(last)]
}

/// Two-sample Kolmogorov—Smirnov statistic between two degree sequences.
///
/// The largest vertical gap between the two empirical CDFs, in `[0, 1]`. `0.0`
/// means the two distributions are identical; `1.0` means they do not overlap at
/// all.
///
/// This is a **distance, not a test**: no p-value is computed, because the two
/// samples here are not independent draws from anything — one is a real graph and
/// the other was fitted to it. Reporting a p-value would dress a descriptive gap
/// as an inferential result.
///
/// Returns `None` when either sequence is empty.
#[must_use]
pub fn ks_distance(left: &[usize], right: &[usize]) -> Option<f64> {
    if left.is_empty() || right.is_empty() {
        return None;
    }
    let mut left_sorted: Vec<usize> = left.to_vec();
    let mut right_sorted: Vec<usize> = right.to_vec();
    left_sorted.sort_unstable();
    right_sorted.sort_unstable();

    let (left_count, right_count) = (
        crate::count_to_f64(left_sorted.len()),
        crate::count_to_f64(right_sorted.len()),
    );
    let (mut left_index, mut right_index) = (0usize, 0usize);
    let mut largest_gap = 0.0f64;

    while left_index < left_sorted.len() && right_index < right_sorted.len() {
        let step = left_sorted[left_index].min(right_sorted[right_index]);
        while left_index < left_sorted.len() && left_sorted[left_index] <= step {
            left_index += 1;
        }
        while right_index < right_sorted.len() && right_sorted[right_index] <= step {
            right_index += 1;
        }
        let gap = (crate::count_to_f64(left_index) / left_count
            - crate::count_to_f64(right_index) / right_count)
            .abs();
        largest_gap = largest_gap.max(gap);
    }
    // Whichever sequence still has entries left runs its CDF to 1 while the other
    // stays put, so the final gap is the remaining fraction.
    let tail = if left_index < left_sorted.len() {
        1.0 - crate::count_to_f64(left_index) / left_count
    } else {
        1.0 - crate::count_to_f64(right_index) / right_count
    };
    Some(largest_gap.max(tail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_star_concentrates_its_degree() -> Result<(), Box<dyn std::error::Error>> {
        let degrees = degree_sequence(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        assert_eq!(degrees, vec![4, 1, 1, 1, 1]);
        let summary = summarise(&degrees).ok_or("a summary")?;
        assert_eq!(summary.max, 4);
        assert_eq!(summary.p50, 1);
        // K(1,4): degrees [1,1,1,1,4] give exactly 0.3 by the sorted-sequence
        // formula. Pinned as a value rather than a threshold, so a change to the
        // estimator has to be deliberate.
        assert!(
            (summary.gini - 0.3).abs() < 1e-9,
            "gini was {}",
            summary.gini
        );
        Ok(())
    }

    #[test]
    fn a_regular_graph_has_gini_zero() -> Result<(), Box<dyn std::error::Error>> {
        let degrees = degree_sequence(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
        let summary = summarise(&degrees).ok_or("a summary")?;
        assert!(summary.gini < 1e-9, "gini was {}", summary.gini);
        Ok(())
    }

    #[test]
    fn an_empty_sequence_has_no_mean_rather_than_a_mean_of_zero() {
        assert_eq!(summarise(&[]), None);
    }

    #[test]
    fn identical_sequences_have_zero_distance() {
        let degrees = vec![1usize, 2, 2, 3, 9];
        assert_eq!(ks_distance(&degrees, &degrees), Some(0.0));
    }

    #[test]
    fn disjoint_sequences_have_distance_one() {
        assert_eq!(ks_distance(&[1, 1, 1], &[9, 9, 9]), Some(1.0));
    }

    #[test]
    fn a_shifted_sequence_lands_between() {
        let distance = ks_distance(&[1, 2, 3, 4], &[3, 4, 5, 6]);
        assert!(
            distance.is_some_and(|gap| (0.0..1.0).contains(&gap) && gap > 0.4),
            "expected a partial overlap, got {distance:?}"
        );
    }

    #[test]
    fn an_empty_side_has_no_distance() {
        assert_eq!(ks_distance(&[], &[1, 2]), None);
        assert_eq!(ks_distance(&[1, 2], &[]), None);
    }
}
