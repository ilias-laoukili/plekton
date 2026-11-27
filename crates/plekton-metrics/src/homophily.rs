// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Edge-homophily computation.

use std::convert::Infallible;

/// Compute graph homophily: the fraction of edges whose two endpoints belong
/// to the same community (label class).
///
/// Defined for every labelling as soon as there is an edge, a single community
/// included: when every node is in one community, every edge stays inside it and
/// the homophily is `1.0`. Whether a graph carries labels at all is for the
/// caller to say — this function always has some — so a caller without labels
/// must not pass a stand-in such as all-zero communities and report the result.
///
/// Returns `None` rather than `0.0` only when there is no edge to measure,
/// because `0.0` is also a real answer — a perfectly heterophilous graph:
///
/// | case | result |
/// |---|---|
/// | no edges to measure | `None` |
/// | edges, a single community | `Some(1.0)` |
/// | edges and labels exist, none intra-community | `Some(0.0)` |
///
/// Pair the `None` with a [`crate::skipped::SkippedMetric`] saying so.
///
/// # Arguments
/// * `edges`       – Canonical edge list; each pair satisfies `source < target`.
/// * `communities` – Community index for each node, indexed by node ID.
///
/// # Examples
/// ```
/// use plekton_metrics::homophily::compute_homophily;
///
/// // Two edges: one intra-community, one inter-community → 0.5
/// let communities = [0, 0, 1];
/// assert_eq!(
///     compute_homophily(&[(0, 1), (1, 2)], &communities),
///     Some(0.5)
/// );
///
/// // A single community: every edge stays inside it
/// assert_eq!(compute_homophily(&[(0, 1)], &[0, 0]), Some(1.0));
///
/// // No edge, nothing to measure
/// assert_eq!(compute_homophily(&[], &[0, 1]), None);
/// ```
#[must_use]
pub fn compute_homophily(edges: &[(usize, usize)], communities: &[usize]) -> Option<f64> {
    let Ok(homophily) = try_homophily_of(
        edges.iter().map(|&edge| Ok::<_, Infallible>(edge)),
        communities,
    );
    homophily
}

/// [`compute_homophily`] over edges that may fail to convert, without copying
/// them.
///
/// This is the one implementation of edge homophily: [`compute_homophily`] calls
/// it, and so do the `plekton` generators, whose `(i64, i64)` edges reach it
/// through a fallible conversion to node indices. The count stops at the first
/// `Err`, which is returned.
///
/// # Errors
/// The first `Err` that `edges` yields.
pub fn try_homophily_of<I, E>(edges: I, communities: &[usize]) -> Result<Option<f64>, E>
where
    I: IntoIterator<Item = Result<(usize, usize), E>>,
{
    let (mut n_edges, mut n_matching) = (0_usize, 0_usize);
    for edge in edges {
        let (source, target) = edge?;
        n_edges += 1;
        if communities[source] == communities[target] {
            n_matching += 1;
        }
    }
    Ok((n_edges > 0).then(|| crate::count_to_f64(n_matching) / crate::count_to_f64(n_edges)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_community_is_fully_homophilous() {
        // All-zero communities are one community, not "unlabelled".
        assert_eq!(
            compute_homophily(&[(0, 1), (1, 2)], &[0, 0, 0]).map(f64::to_bits),
            Some(1.0_f64.to_bits())
        );
        assert_eq!(
            compute_homophily(&[(0, 1)], &[3, 3]).map(f64::to_bits),
            Some(1.0_f64.to_bits())
        );
    }

    #[test]
    fn no_edge_means_no_homophily() {
        assert_eq!(compute_homophily(&[], &[0, 1]), None);
        assert_eq!(compute_homophily(&[], &[]), None);
    }

    #[test]
    fn a_failed_conversion_stops_the_count() {
        let edges: [Result<(usize, usize), &str>; 3] =
            [Ok((0, 1)), Err("negative endpoint"), Ok((1, 2))];
        assert_eq!(
            try_homophily_of(edges, &[0, 0, 1]),
            Err("negative endpoint")
        );
    }
}
