// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Attribute mixing: how ties fall within and across the categories of one
//! node attribute.
//!
//! Unlike [`crate::homophily::compute_homophily`], every label is a real
//! category here (label `0` is not "unlabelled"), because the caller labels
//! nodes by an attribute it chose: sex, age band, milieu.

/// Mixing of a set of ties on one categorical attribute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mixing {
    /// Share of ties whose two ends are in the same category.
    pub within: f64,
    /// The share `within` would have if ties ignored the attribute: the sum of
    /// the squared shares of tie ends held by each category.
    pub random: f64,
    /// Newman's assortativity coefficient: `(within - random) / (1 - random)`.
    /// `1` when every tie stays inside its category, `0` for random mixing,
    /// negative when ties avoid their own category. `None` when every tie end
    /// is in a single category (the coefficient is undefined).
    pub r: Option<f64>,
}

/// Measures the mixing of `edges` on the attribute given by `labels`
/// (one category index per node).
///
/// Returns `None` when there is no edge to measure.
///
/// # Examples
/// ```
/// use plekton_metrics::mixing::attribute_mixing;
///
/// // Two categories, every tie across them: fully heterophilous.
/// let m = attribute_mixing(&[(0, 2), (1, 3)], &[0, 0, 1, 1]).unwrap();
/// assert_eq!(m.within, 0.0);
/// assert_eq!(m.r, Some(-1.0));
/// ```
#[must_use]
pub fn attribute_mixing(edges: &[(usize, usize)], labels: &[usize]) -> Option<Mixing> {
    if edges.is_empty() {
        return None;
    }
    let n_categories = labels.iter().max().map_or(0, |&m| m + 1);
    let mut ends = vec![0usize; n_categories];
    let mut same = 0usize;
    for &(a, b) in edges {
        ends[labels[a]] += 1;
        ends[labels[b]] += 1;
        if labels[a] == labels[b] {
            same += 1;
        }
    }
    let total = crate::count_to_f64(2 * edges.len());
    let random: f64 = ends
        .iter()
        .map(|&e| {
            let share = crate::count_to_f64(e) / total;
            share * share
        })
        .sum();
    let within = crate::count_to_f64(same) / crate::count_to_f64(edges.len());
    let r = (random < 1.0).then(|| (within - random) / (1.0 - random));
    Some(Mixing { within, random, r })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_edges_nothing_to_measure() {
        assert_eq!(attribute_mixing(&[], &[0, 1]), None);
    }

    #[test]
    fn all_within_is_fully_assortative() -> Result<(), &'static str> {
        let m = attribute_mixing(&[(0, 1), (2, 3)], &[0, 0, 1, 1]).ok_or("no edges")?;
        assert!((m.within - 1.0).abs() < 1e-12);
        assert!((m.random - 0.5).abs() < 1e-12);
        assert!((m.r.ok_or("undefined")? - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn single_category_has_no_coefficient() -> Result<(), &'static str> {
        let m = attribute_mixing(&[(0, 1)], &[0, 0]).ok_or("no edges")?;
        assert_eq!(m.r, None);
        Ok(())
    }
}
