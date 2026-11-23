// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Edge-homophily computation.

/// Compute graph homophily: the fraction of edges whose two endpoints belong
/// to the same community (label class).
///
/// Returns `None` rather than `0.0` where the fraction does not exist, because
/// `0.0` is also a real answer — a perfectly heterophilous graph — and a reader
/// given a bare `0.0` cannot tell it from a graph nobody measured:
///
/// | case | before | now |
/// |---|---|---|
/// | no edges to measure | `0.0` | `None` |
/// | every node in community `0`, i.e. nothing is labelled | `0.0` | `None` |
/// | edges and labels exist, none intra-community | `0.0` | `Some(0.0)` |
///
/// The second row is the important one: an all-zero assignment means the source
/// carried no labels, so every "matching" community is an artefact of the default
/// rather than a property of the graph.
///
/// Pair the `None` with a [`crate::skipped::SkippedMetric`] to say which case it was.
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
/// // Nothing is labelled, so there is no community for an edge to stay inside
/// assert_eq!(compute_homophily(&[(0, 1)], &[0, 0]), None);
/// ```
#[must_use]
pub fn compute_homophily(edges: &[(usize, usize)], communities: &[usize]) -> Option<f64> {
    if edges.is_empty() {
        return None;
    }
    // If no node carries a label (all community = 0), homophily is meaningless.
    let any_labeled = communities.iter().any(|&community| community != 0);
    if !any_labeled {
        return None;
    }

    let n_matching_edges = edges
        .iter()
        .filter(|&&(source, target)| communities[source] == communities[target])
        .count();

    Some(crate::count_to_f64(n_matching_edges) / crate::count_to_f64(edges.len()))
}
