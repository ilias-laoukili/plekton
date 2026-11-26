// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Edge-similarity scoring primitives for graph sparsification.
//!
//! Each function assigns a score to every edge of an undirected graph, ranking
//! edges by structural (or feature) importance. These scores are the input to
//! selection strategies in the `graph-sparsify` crate.
//!
//! # Metrics
//!
//! | Function | Score for edge `(u, v)` |
//! |---|---|
//! | [`jaccard_scores`]        | `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|` — neighbourhood overlap |
//! | [`adamic_adar_scores`]    | `Σ_{w ∈ N(u) ∩ N(v)} 1 / ln(deg(w) + 1)` — hub-penalised overlap |
//! | [`degree_product_scores`] | `deg(u) · deg(v)` — degree-preferential score |
//! | [`feature_cosine_scores`] | `max(cos(f_u, f_v), 0)` — node-feature alignment |
//!
//! # Self-loops
//!
//! `jaccard_scores`, `adamic_adar_scores` and `degree_product_scores` take an
//! `ignore_self_loops` flag (**default policy: `true` at call sites**). When
//! `true`, edges `(u, u)` do not contribute to any neighbourhood and receive a
//! score of `0.0`. Stripping them matters because a self-loop would otherwise
//! count a node as its own neighbour and inflate the scores.
//!
//! All neighbourhood-based functions expect a canonical edge list (`source <
//! target`, each undirected edge once) and return a `Vec<f64>` aligned with the
//! input `edges` slice.

use hashbrown::HashSet;
use rayon::prelude::*;

use crate::count_to_f64;

/// Build per-node neighbour sets from a canonical edge list.
///
/// Duplicate edges collapse (set semantics), so neighbourhoods are those of the
/// binary adjacency matrix. When `ignore_self_loops` is
/// `true`, `(u, u)` edges are skipped so a node is never its own neighbour.
fn build_neighbour_sets(
    n_nodes: usize,
    edges: &[(usize, usize)],
    ignore_self_loops: bool,
) -> Vec<HashSet<usize>> {
    let mut neighbour_sets: Vec<HashSet<usize>> = vec![HashSet::new(); n_nodes];
    for &(u, v) in edges {
        if u == v {
            if !ignore_self_loops {
                neighbour_sets[u].insert(u);
            }
        } else {
            neighbour_sets[u].insert(v);
            neighbour_sets[v].insert(u);
        }
    }
    neighbour_sets
}

/// Build per-node neighbour lists, each sorted in ascending id order without
/// repeats.
///
/// The same neighbourhoods as [`build_neighbour_sets`], in an order that does not
/// change from one run to the next, so a floating-point sum taken over them
/// always rounds the same way.
fn build_sorted_neighbours(
    n_nodes: usize,
    edges: &[(usize, usize)],
    ignore_self_loops: bool,
) -> Vec<Vec<usize>> {
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(u, v) in edges {
        if u == v {
            if !ignore_self_loops {
                neighbours[u].push(u);
            }
        } else {
            neighbours[u].push(v);
            neighbours[v].push(u);
        }
    }
    neighbours.par_iter_mut().for_each(|list| {
        list.sort_unstable();
        list.dedup();
    });
    neighbours
}

/// Count common neighbours of `u` and `v` by iterating the smaller set.
///
/// Cost is `O(min(deg(u), deg(v)))`.
#[inline]
fn intersection_size(a: &HashSet<usize>, b: &HashSet<usize>) -> usize {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small.iter().filter(|&&w| large.contains(&w)).count()
}

/// Compute the Jaccard similarity of every edge.
///
/// `J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|`, with the union derived as
/// `deg(u) + deg(v) − |N(u) ∩ N(v)|`. Edges whose union is empty score `0.0`.
///
/// # Arguments
/// * `n_nodes` – Total number of nodes (indices in `edges` must be `< n_nodes`).
/// * `edges`   – Canonical edge list; returned scores are aligned with it.
/// * `ignore_self_loops` – Skip `(u, u)` edges (recommended default `true`).
///
/// # Examples
/// ```
/// use plekton_metrics::similarity::jaccard_scores;
///
/// // Triangle: each edge shares exactly one common neighbour → 1/3.
/// let s = jaccard_scores(3, &[(0, 1), (0, 2), (1, 2)], true);
/// assert!((s[0] - 1.0 / 3.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn jaccard_scores(
    n_nodes: usize,
    edges: &[(usize, usize)],
    ignore_self_loops: bool,
) -> Vec<f64> {
    let neighbour_sets = build_neighbour_sets(n_nodes, edges, ignore_self_loops);
    let degrees: Vec<usize> = neighbour_sets.iter().map(HashSet::len).collect();

    edges
        .par_iter()
        .map(|&(u, v)| {
            if ignore_self_loops && u == v {
                return 0.0;
            }
            let intersection = intersection_size(&neighbour_sets[u], &neighbour_sets[v]);
            let union = degrees[u] + degrees[v] - intersection;
            if union == 0 {
                0.0
            } else {
                count_to_f64(intersection) / count_to_f64(union)
            }
        })
        .collect()
}

/// Compute the Adamic-Adar index of every edge.
///
/// `AA(u, v) = Σ_{w ∈ N(u) ∩ N(v)} 1 / max(ln(deg(w) + 1), 1e-10)`. Common
/// neighbours of low degree contribute more, penalising hubs. Note the `+1`
/// inside the logarithm: the classic Adamic–Adar index weights each common
/// neighbour by `1 / ln deg(w)`, so these scores are not interchangeable with
/// implementations of the classic form.
///
/// The common neighbours of an edge are summed in ascending id order, so a
/// score is the same bits in every run and every process.
///
/// # Arguments
/// See [`jaccard_scores`].
///
/// # Examples
/// ```
/// use plekton_metrics::similarity::adamic_adar_scores;
///
/// // Triangle edge (0,1): one common neighbour of degree 2 → 1/ln(3).
/// let s = adamic_adar_scores(3, &[(0, 1), (0, 2), (1, 2)], true);
/// assert!((s[0] - 1.0 / 3.0_f64.ln()).abs() < 1e-12);
/// ```
#[must_use]
pub fn adamic_adar_scores(
    n_nodes: usize,
    edges: &[(usize, usize)],
    ignore_self_loops: bool,
) -> Vec<f64> {
    let neighbours = build_sorted_neighbours(n_nodes, edges, ignore_self_loops);

    edges
        .par_iter()
        .map(|&(u, v)| {
            if ignore_self_loops && u == v {
                return 0.0;
            }
            let (a, b) = (&neighbours[u], &neighbours[v]);
            let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            // `small` is sorted, so the common neighbours come in ascending id
            // order whichever list is the smaller one.
            small
                .iter()
                .filter(|w| large.binary_search(w).is_ok())
                .map(|&w| {
                    // `ln_1p(deg)` == `ln(deg + 1)`, computed without the
                    // intermediate `1.0 +` rounding.
                    let log_degree = count_to_f64(neighbours[w].len()).ln_1p().max(1e-10);
                    1.0 / log_degree
                })
                .sum()
        })
        .collect()
}

/// Compute the degree-product score `deg(u) · deg(v)` of every edge.
///
/// Degrees are taken from the binary neighbourhood (duplicate edges collapse).
/// Each degree is converted to `f64` before multiplication so the product
/// cannot overflow `usize` on dense graphs.
///
/// # Arguments
/// See [`jaccard_scores`].
#[must_use]
pub fn degree_product_scores(
    n_nodes: usize,
    edges: &[(usize, usize)],
    ignore_self_loops: bool,
) -> Vec<f64> {
    let neighbour_sets = build_neighbour_sets(n_nodes, edges, ignore_self_loops);
    let degrees: Vec<usize> = neighbour_sets.iter().map(HashSet::len).collect();

    edges
        .par_iter()
        .map(|&(u, v)| {
            if ignore_self_loops && u == v {
                return 0.0;
            }
            count_to_f64(degrees[u]) * count_to_f64(degrees[v])
        })
        .collect()
}

/// Compute the clamped feature cosine similarity of every edge.
///
/// For edge `(u, v)`, the score is `max(cos(f_u, f_v), 0)` where `f_i` is node
/// `i`'s feature row. Negative cosines are clamped to `0.0` (treated as "no
/// similarity") to keep the score range consistent with Jaccard/AA.
///
/// Features are supplied as a **flat, row-major** slice of length
/// `n_nodes · feat_dim` (node `i`'s row is `features[i · feat_dim ..]`). This
/// layout is deliberate: it maps directly onto a contiguous `NumPy`/`ndarray`
/// buffer for zero-copy transfer across the `PyO3` boundary. Per-node L2 norms
/// are precomputed once in `O(n_nodes · feat_dim)`; each edge then costs a
/// single `O(feat_dim)` dot product. A zero-norm row uses a `1e-10` floor to
/// avoid division by zero.
///
/// # Arguments
/// * `edges`    – Edge list; returned scores are aligned with it.
/// * `features` – Flat row-major features, length `n_nodes · feat_dim`.
/// * `feat_dim` – Feature dimension (must be non-zero and divide `features.len()`).
///
/// # Panics
/// Panics if `feat_dim == 0`, so callers must validate it (the `PyO3` wrapper
/// surfaces this as a `ValueError` rather than a panic).
#[must_use]
pub fn feature_cosine_scores(
    edges: &[(usize, usize)],
    features: &[f32],
    feat_dim: usize,
) -> Vec<f64> {
    assert!(feat_dim > 0, "feat_dim must be non-zero");
    let n_nodes = features.len() / feat_dim;

    let row = |i: usize| &features[i * feat_dim..(i + 1) * feat_dim];

    let norms: Vec<f64> = (0..n_nodes)
        .into_par_iter()
        .map(|i| {
            row(i)
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt()
                .max(1e-10)
        })
        .collect();

    edges
        .par_iter()
        .map(|&(u, v)| {
            let dot: f64 = row(u)
                .iter()
                .zip(row(v))
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            (dot / (norms[u] * norms[v])).max(0.0)
        })
        .collect()
}

/// Fraction of the original edges retained after sparsification.
///
/// Returns `sparse.len() / original.len()`, or `0.0` when the original graph
/// has no edges.
#[must_use]
pub fn edge_retention(original: &[(usize, usize)], sparse: &[(usize, usize)]) -> f64 {
    if original.is_empty() {
        return 0.0;
    }
    count_to_f64(sparse.len()) / count_to_f64(original.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
    // Diamond: 0-1, 0-2, 1-2, 1-3, 2-3. Edge (1,2) shares neighbours {0, 3}.
    const DIAMOND: [(usize, usize); 5] = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)];

    #[test]
    fn jaccard_triangle_is_one_third() {
        let s = jaccard_scores(3, &TRIANGLE, true);
        for score in s {
            assert!((score - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn jaccard_path_has_no_overlap() {
        // 0-1-2 path: endpoints share no common neighbour.
        let s = jaccard_scores(3, &[(0, 1), (1, 2)], true);
        assert_eq!(s, vec![0.0, 0.0]);
    }

    #[test]
    fn jaccard_diamond_middle_edge_is_half() {
        // Edge (1,2) is index 2. |{0,3}| / |{0,1,2,3}| = 2/4 = 0.5.
        let s = jaccard_scores(4, &DIAMOND, true);
        assert!((s[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn adamic_adar_triangle_single_neighbour_degree_two() {
        let s = adamic_adar_scores(3, &TRIANGLE, true);
        let expected = 1.0 / 3.0_f64.ln(); // one common neighbour of degree 2
        for score in s {
            assert!((score - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn adamic_adar_diamond_middle_edge_two_neighbours() {
        // Common neighbours {0, 3}, both degree 2 → 2 / ln(3).
        let s = adamic_adar_scores(4, &DIAMOND, true);
        let expected = 2.0 / 3.0_f64.ln();
        assert!((s[2] - expected).abs() < 1e-12);
    }

    #[test]
    fn degree_product_triangle() {
        // Every node has degree 2 → product 4 for each edge.
        let s = degree_product_scores(3, &TRIANGLE, true);
        assert_eq!(s, vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn self_loop_policy_changes_scores() {
        // Triangle + a self-loop on node 0.
        let edges = [(0, 1), (0, 2), (1, 2), (0, 0)];

        // ignore=true: triangle scores unchanged, self-loop edge scores 0.
        let ignored = jaccard_scores(3, &edges, true);
        assert!((ignored[0] - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(ignored[3].to_bits(), 0.0_f64.to_bits());

        // ignore=false: 0 becomes its own neighbour, so N(0)={0,1,2} and edge
        // (0,1) overlap grows to {0,2} / {0,1,2} = 2/3.
        let kept = jaccard_scores(3, &edges, false);
        assert!((kept[0] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn feature_cosine_orthogonal_and_parallel() {
        // node0=[1,0], node1=[1,0], node2=[0,1].
        let features = [1.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let s = feature_cosine_scores(&[(0, 1), (0, 2), (1, 2)], &features, 2);
        assert!((s[0] - 1.0).abs() < 1e-9); // parallel
        assert!(s[1].abs() < 1e-9); // orthogonal → clamped 0
        assert!(s[2].abs() < 1e-9);
    }

    #[test]
    fn edge_retention_ratio() {
        assert!((edge_retention(&TRIANGLE, &[(0, 1)]) - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(edge_retention(&[], &[]).to_bits(), 0.0_f64.to_bits());
    }
}
