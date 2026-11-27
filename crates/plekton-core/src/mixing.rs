// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Graphs whose ties follow a mixing target per node attribute.
//!
//! The SBM generators take one block label and one homophily, or a fitted
//! block matrix. Social layers need more: marriages heterophilous on sex and
//! homophilous on age; friendships homophilous on milieu, age, place and
//! opinion at once; ties only inside a household or a workplace; a cap of one
//! partner. This generator takes:
//!
//! * a category per node for each attribute;
//! * an optional group per node: ties stay inside a group, and a node without
//!   a group takes no tie;
//! * a wanted degree per node (given, or drawn around a mean), with a cap;
//! * per attribute, the share of ties joining two nodes of the same category
//!   (`within`), or the share of ties between every pair of categories
//!   (`matrix`);
//! * optionally, a clustering target (the share of paths of two ties that
//!   close into a triangle: friends of friends who are friends);
//! * optionally, a strength per tie, higher when the two ends are alike;
//! * optionally, ties that go one way (`directed`: who follows whom), with a
//!   heavy tail of popular nodes.
//!
//! Above the random rate (the within share if ties ignored the attribute) a
//! layer is homophilous on that attribute; below it, heterophilous.
//!
//! How (undirected): nodes sharing all their targeted categories form a block.
//! Iterative proportional fitting finds how many tie ends should go between
//! each pair of blocks, so that every block keeps its degrees and every
//! attribute meets its target. Ties are then placed block pair by block pair,
//! inside groups, up to each node's wanted degree; a second pass fills the
//! degrees still wanted, only where the targets allow ties. A clustering
//! target then rewires pairs of ties to close triangles, keeping every degree
//! and the mixing.
//!
//! How (directed): each node follows its wanted number of others, drawn with
//! weight `popularity x factors`, where the popularity of a node is
//! log-normal and the factors, one per attribute and pair of categories, are
//! fitted so that every attribute meets its target.
//!
//! The result reports, for every attribute, the target next to the measured
//! within share, random rate and assortativity, and the target clustering
//! next to the measured transitivity.

use std::collections::{BTreeMap, HashMap, HashSet};

use num_traits::ToPrimitive;
use plekton_metrics::{
    mixing::attribute_mixing, skipped::SkippedMetric, transitivity::transitivity,
};
use rand::{distributions::WeightedIndex, seq::SliceRandom, Rng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal, Normal};
use serde::{Deserialize, Serialize};

use crate::{sampling::seeded_rng, SynthGraphError};

/// Rounds of iterative proportional fitting.
const FIT_ROUNDS: usize = 30;
/// Attempts to place one tie before giving it up.
const PLACE_TRIES: usize = 20;
/// Rewiring attempts per tie when closing triangles.
const REWIRE_TRIES: usize = 200;
/// How far closing triangles may move the mixing of a targeted attribute:
/// the share of ties whose pair of categories changed.
const MIX_SLACK: f64 = 0.02;

/// Category names and the category index of every node, per attribute.
type Codes<'a> = BTreeMap<&'a str, (Vec<String>, Vec<usize>)>;

/// Target mixing of one attribute: exactly one of `within` or `matrix`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixingTarget {
    /// Share of ties joining two nodes of the same category, in `[0, 1]`.
    /// `0` on sex for marriages: every tie joins a man and a woman.
    #[serde(default)]
    pub within: Option<f64>,
    /// Share of ties between each pair of categories, e.g.
    /// `{"female": {"male": 1.0}}`. Made symmetric and normalised to sum to 1.
    #[serde(default)]
    pub matrix: Option<BTreeMap<String, BTreeMap<String, f64>>>,
}

/// Strength of every tie, in `(0, 1)`: on the logit scale, the logit of
/// `mean`, plus normal noise of σ `spread`, plus `same[a]` when both ends
/// share their category of attribute `a`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrengthSpec {
    /// Typical strength, in `(0, 1)`.
    pub mean: f64,
    /// σ of the noise on the logit scale; `0` gives every tie `mean`.
    #[serde(default)]
    pub spread: f64,
    /// Added on the logit scale when both ends share the attribute's category.
    #[serde(default)]
    pub same: BTreeMap<String, f64>,
}

/// What to generate. Every per-node list has `nodes` entries.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixingRequest {
    /// Number of nodes.
    pub nodes: usize,
    /// Seed for reproducible output; `None` draws one from the OS.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Category of every node, per attribute name.
    #[serde(default)]
    pub attributes: BTreeMap<String, Vec<String>>,
    /// Group of every node; ties stay inside groups; `null` takes no tie.
    #[serde(default)]
    pub groups: Option<Vec<Option<String>>>,
    /// Wanted degree of every node. Replaces `mean_degree`.
    #[serde(default)]
    pub degrees: Option<Vec<usize>>,
    /// Mean wanted degree, used when `degrees` is not given.
    #[serde(default)]
    pub mean_degree: Option<f64>,
    /// Spread of wanted degrees around `mean_degree` (σ of a log-normal);
    /// `0` gives everyone the same degree.
    #[serde(default)]
    pub degree_spread: f64,
    /// Cap on every node's degree (`1` for a matching).
    #[serde(default)]
    pub max_degree: Option<usize>,
    /// Target mixing per attribute name.
    #[serde(default)]
    pub mixing: BTreeMap<String, MixingTarget>,
    /// Ties go one way, `(from, to)`: each node follows its wanted degree of
    /// others (out-degree), and is followed according to its popularity.
    #[serde(default)]
    pub directed: bool,
    /// σ of the log-normal popularity of the nodes (directed only); the larger,
    /// the heavier the tail of the most followed.
    #[serde(default)]
    pub popularity_spread: f64,
    /// Target transitivity, in `[0, 1]` (undirected only).
    #[serde(default)]
    pub clustering: Option<f64>,
    /// Strength of every tie.
    #[serde(default)]
    pub strength: Option<StrengthSpec>,
}

/// Measured mixing of one attribute, next to its target.
#[derive(Debug, Clone, Serialize)]
pub struct AttributeReport {
    /// Target within share, if the attribute had a target.
    pub target: Option<f64>,
    /// Measured share of ties inside a category.
    pub within: Option<f64>,
    /// Within share if ties ignored the attribute.
    pub random: Option<f64>,
    /// Newman's assortativity.
    pub r: Option<f64>,
}

/// Clustering: target and measured transitivity.
#[derive(Debug, Clone, Serialize)]
pub struct ClusteringReport {
    /// Target transitivity, if one was given.
    pub target: Option<f64>,
    /// Measured transitivity (`None` for a directed graph, or without a path
    /// of two ties).
    pub transitivity: Option<f64>,
}

/// A generated graph and its report.
#[derive(Debug, Clone, Serialize)]
pub struct MixingGraph {
    /// Ties, sorted: `(a, b)` with `a < b`, or `(from, to)` when directed.
    pub edges: Vec<(usize, usize)>,
    /// Whether ties go one way.
    pub directed: bool,
    /// Strength of every tie, in the order of `edges`, when asked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strengths: Option<Vec<f64>>,
    /// Mean wanted degree of the nodes that can take ties (out-degree when
    /// directed); `None` when no node can take one — no node at all, or every
    /// wanted degree 0 — so the mean would be over nothing.
    pub wanted_degree: Option<f64>,
    /// Mean degree reached by the same nodes; `None` exactly when
    /// [`Self::wanted_degree`] is.
    pub mean_degree: Option<f64>,
    /// Mixing of every attribute given, targeted or not.
    pub mixing: BTreeMap<String, AttributeReport>,
    /// Clustering, target and measured.
    pub clustering: ClusteringReport,
    /// Why [`Self::wanted_degree`] and [`Self::mean_degree`] are `None`, when
    /// they are.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub metrics_skipped: Vec<SkippedMetric>,
}

/// Generates a graph from a [`MixingRequest`].
///
/// # Errors
/// Returns [`SynthGraphError::InvalidParameter`] when a per-node list has the
/// wrong length, a target names an attribute that is not given, a `within`
/// is outside `[0, 1]`, a target has both or neither of `within` and
/// `matrix`, neither `degrees` nor `mean_degree` is given, a clustering
/// target is outside `[0, 1]` or given for a directed graph, or a strength
/// is malformed.
pub fn generate_mixing_graph(request: &MixingRequest) -> Result<MixingGraph, SynthGraphError> {
    validate(request)?;
    let n = request.nodes;
    let mut rng = seeded_rng(request.seed);

    let groups = group_ids(request);
    let wanted = wanted_degrees(request, &groups, &mut rng)?;

    // Category index of every node, per attribute.
    let mut codes: BTreeMap<&str, (Vec<String>, Vec<usize>)> = BTreeMap::new();
    for (name, values) in &request.attributes {
        let mut names: Vec<String> = values.clone();
        names.sort();
        names.dedup();
        let index: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let per_node = values
            .iter()
            .map(|v| index.get(v.as_str()).copied().unwrap_or(0))
            .collect();
        codes.insert(name.as_str(), (names, per_node));
    }

    let edges = if request.directed {
        directed_ties(request, &codes, &wanted, &groups, &mut rng)?
    } else {
        let edges = undirected_ties(request, &codes, &wanted, &groups, &mut rng)?;
        match request.clustering {
            Some(target) => {
                let labels: Vec<&[usize]> = request
                    .mixing
                    .keys()
                    .map(|a| codes[a.as_str()].1.as_slice())
                    .collect();
                close_triangles(edges, n, target, &labels, &mut rng)
            }
            None => edges,
        }
    };
    let strengths = match &request.strength {
        Some(spec) => Some(tie_strengths(spec, &edges, &codes, &mut rng)?),
        None => None,
    };

    let reaching: Vec<usize> = (0..n).filter(|&u| wanted[u] > 0).collect();
    let mut degree = vec![0usize; n];
    for &(a, b) in &edges {
        degree[a] += 1;
        if !request.directed {
            degree[b] += 1;
        }
    }
    // A mean over the nodes that can take ties: none when there are none.
    let mean = |xs: &[usize]| {
        (!reaching.is_empty())
            .then(|| to_f64(reaching.iter().map(|&u| xs[u]).sum()) / to_f64(reaching.len()))
    };
    let metrics_skipped = if reaching.is_empty() {
        let because = "no node can take a tie, so there is no node to average over";
        vec![
            SkippedMetric::undefined("wanted_degree", because),
            SkippedMetric::undefined("mean_degree", because),
        ]
    } else {
        Vec::new()
    };

    let mut mixing = BTreeMap::new();
    for (name, (names, per_node)) in &codes {
        let measured = attribute_mixing(&edges, per_node);
        let target = request.mixing.get(*name).map(|t| target_within(t, names));
        mixing.insert(
            (*name).to_string(),
            AttributeReport {
                target,
                within: measured.map(|m| m.within),
                random: measured.map(|m| m.random),
                r: measured.and_then(|m| m.r),
            },
        );
    }

    let clustering = ClusteringReport {
        target: request.clustering,
        transitivity: if request.directed {
            None
        } else {
            transitivity(n, &edges)
        },
    };
    Ok(MixingGraph {
        wanted_degree: mean(&wanted),
        mean_degree: mean(&degree),
        directed: request.directed,
        edges,
        strengths,
        mixing,
        clustering,
        metrics_skipped,
    })
}

/// Undirected ties: blocks, fitted tie budgets between blocks, placement.
fn undirected_ties(
    request: &MixingRequest,
    codes: &Codes<'_>,
    wanted: &[usize],
    groups: &[Option<usize>],
    rng: &mut ChaCha8Rng,
) -> Result<Vec<(usize, usize)>, SynthGraphError> {
    let n = request.nodes;
    // Blocks: nodes sharing all their targeted categories.
    let targeted: Vec<&str> = request.mixing.keys().map(String::as_str).collect();
    let mut block_index: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut block_of = vec![0usize; n];
    for (node, block) in block_of.iter_mut().enumerate() {
        let key: Vec<usize> = targeted.iter().map(|a| codes[a].1[node]).collect();
        let next = block_index.len();
        *block = *block_index.entry(key).or_insert(next);
    }
    let n_blocks = block_index.len();
    let mut labels = vec![Vec::new(); n_blocks];
    for (key, &b) in &block_index {
        labels[b].clone_from(key);
    }
    let mut members = vec![Vec::new(); n_blocks];
    let mut stubs = vec![0.0; n_blocks];
    for node in 0..n {
        members[block_of[node]].push(node);
        stubs[block_of[node]] += to_f64(wanted[node]);
    }

    // Target matrix of every targeted attribute, over its categories.
    let total: f64 = stubs.iter().sum();
    let mut targets = Vec::new();
    for (a, name) in targeted.iter().enumerate() {
        let names = &codes[name].0;
        let mut shares = vec![0.0; names.len()];
        if total > 0.0 {
            for (b, label) in labels.iter().enumerate() {
                shares[label[a]] += stubs[b] / total;
            }
        }
        targets.push(target_matrix(&request.mixing[*name], names, &shares)?);
    }

    let expected = fit(&stubs, &labels, &targets);
    Ok(place(&expected, &members, &block_of, wanted, groups, rng))
}

/// The within share a target asks for.
fn target_within(target: &MixingTarget, names: &[String]) -> f64 {
    if let Some(w) = target.within {
        return w;
    }
    let (mut diagonal, mut total) = (0.0, 0.0);
    for (u, row) in target.matrix.iter().flatten() {
        for (v, x) in row {
            if names.contains(u) && names.contains(v) {
                total += x;
                if u == v {
                    diagonal += x;
                }
            }
        }
    }
    if total > 0.0 {
        diagonal / total
    } else {
        0.0
    }
}

/// JSON in, JSON out: the entry point of the Python bindings.
///
/// # Errors
/// Returns [`SynthGraphError::Serialization`] for malformed JSON, and the
/// errors of [`generate_mixing_graph`].
pub fn generate_mixing_graph_json(request_json: &str) -> Result<String, SynthGraphError> {
    let request: MixingRequest = serde_json::from_str(request_json)?;
    Ok(serde_json::to_string(&generate_mixing_graph(&request)?)?)
}

fn validate(request: &MixingRequest) -> Result<(), SynthGraphError> {
    let n = request.nodes;
    let bad = |m: String| Err(SynthGraphError::InvalidParameter(m));
    for (name, values) in &request.attributes {
        if values.len() != n {
            return bad(format!(
                "attribute '{name}' has {} values for {n} nodes",
                values.len()
            ));
        }
    }
    if request.groups.as_ref().is_some_and(|g| g.len() != n) {
        return bad(format!("groups must have {n} entries"));
    }
    match (&request.degrees, request.mean_degree) {
        (Some(d), _) if d.len() != n => return bad(format!("degrees must have {n} entries")),
        (None, None) => return bad("give degrees or mean_degree".into()),
        (None, Some(m)) if !(m.is_finite() && m >= 0.0) => {
            return bad("mean_degree must be >= 0".into())
        }
        _ => {}
    }
    if !(request.degree_spread.is_finite() && request.degree_spread >= 0.0) {
        return bad("degree_spread must be >= 0".into());
    }
    for (name, t) in &request.mixing {
        if !request.attributes.contains_key(name) {
            return bad(format!(
                "mixing target for '{name}', which is not among the attributes"
            ));
        }
        match (t.within, &t.matrix) {
            (Some(w), None) if (0.0..=1.0).contains(&w) => {}
            (Some(_), None) => return bad(format!("'{name}': within must be in [0, 1]")),
            (None, Some(m))
                if m.values()
                    .flat_map(BTreeMap::values)
                    .all(|x| x.is_finite() && *x >= 0.0) => {}
            (None, Some(_)) => return bad(format!("'{name}': matrix shares must be >= 0")),
            _ => return bad(format!("'{name}': give exactly one of within and matrix")),
        }
    }
    if !(request.popularity_spread.is_finite() && request.popularity_spread >= 0.0) {
        return bad("popularity_spread must be >= 0".into());
    }
    match request.clustering {
        Some(_) if request.directed => {
            return bad("a clustering target is for undirected graphs".into())
        }
        Some(c) if !(0.0..=1.0).contains(&c) => return bad("clustering must be in [0, 1]".into()),
        _ => {}
    }
    if let Some(strength) = &request.strength {
        if !(strength.mean > 0.0 && strength.mean < 1.0) {
            return bad("strength mean must be in (0, 1)".into());
        }
        if !(strength.spread.is_finite() && strength.spread >= 0.0) {
            return bad("strength spread must be >= 0".into());
        }
        if let Some(name) = strength
            .same
            .keys()
            .find(|a| !request.attributes.contains_key(*a))
        {
            return bad(format!(
                "strength for '{name}', which is not among the attributes"
            ));
        }
    }
    Ok(())
}

/// Group index of every node; `None` for nodes that take no tie.
fn group_ids(request: &MixingRequest) -> Vec<Option<usize>> {
    let Some(groups) = &request.groups else {
        return vec![Some(0); request.nodes];
    };
    let mut index: BTreeMap<&str, usize> = BTreeMap::new();
    groups
        .iter()
        .map(|g| {
            g.as_deref().map(|name| {
                let next = index.len();
                *index.entry(name).or_insert(next)
            })
        })
        .collect()
}

/// Wanted degree of every node, capped by its group size minus one and by
/// `max_degree`. Drawn around `mean_degree` when `degrees` is not given.
fn wanted_degrees(
    request: &MixingRequest,
    groups: &[Option<usize>],
    rng: &mut ChaCha8Rng,
) -> Result<Vec<usize>, SynthGraphError> {
    let n = request.nodes;
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for g in groups.iter().flatten() {
        *sizes.entry(*g).or_insert(0) += 1;
    }
    let raw: Vec<usize> = if let Some(d) = &request.degrees {
        d.clone()
    } else {
        let mean = request.mean_degree.unwrap_or(0.0);
        let spread = LogNormal::new(0.0, request.degree_spread)
            .map_err(|e| SynthGraphError::InvalidParameter(e.to_string()))?;
        let weights: Vec<f64> = (0..n).map(|_| spread.sample(rng)).collect();
        let mean_weight = weights.iter().sum::<f64>() / to_f64(n.max(1));
        weights
            .iter()
            .map(|w| whole((mean * w / mean_weight).round()).max(1))
            .collect()
    };
    let cap = request.max_degree.unwrap_or(usize::MAX);
    Ok((0..n)
        .map(|u| groups[u].map_or(0, |g| raw[u].min(sizes[&g] - 1).min(cap)))
        .collect())
}

/// Target share of tie ends between every pair of categories, symmetric,
/// summing to 1, given the share of tie ends each category holds.
fn target_matrix(
    target: &MixingTarget,
    names: &[String],
    shares: &[f64],
) -> Result<Vec<Vec<f64>>, SynthGraphError> {
    let c = names.len();
    if let Some(matrix) = &target.matrix {
        let index: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let mut m = vec![vec![0.0; c]; c];
        for (u, row) in matrix {
            for (v, x) in row {
                if let (Some(&i), Some(&j)) = (index.get(u.as_str()), index.get(v.as_str())) {
                    m[i][j] += x / 2.0;
                    m[j][i] += x / 2.0;
                }
            }
        }
        let total: f64 = m.iter().flatten().sum();
        if total <= 0.0 {
            return Err(SynthGraphError::InvalidParameter(
                "a mixing matrix names no category present in the nodes".into(),
            ));
        }
        return Ok(m
            .into_iter()
            .map(|row| row.into_iter().map(|x| x / total).collect())
            .collect());
    }

    let within = target.within.unwrap_or(0.0);
    let rate: f64 = shares.iter().map(|s| s * s).sum();
    let mixed = |u: usize, v: usize| shares[u] * shares[v];
    if within >= rate {
        // Move part of the random mixing onto the diagonal.
        let k = if rate < 1.0 {
            (within - rate) / (1.0 - rate)
        } else {
            0.0
        };
        return Ok((0..c)
            .map(|u| {
                (0..c)
                    .map(|v| {
                        (1.0 - k).mul_add(mixed(u, v), if u == v { k * shares[u] } else { 0.0 })
                    })
                    .collect()
            })
            .collect());
    }
    // Move part of the random mixing off the diagonal.
    let k = 1.0 - within / rate;
    let apart = across(shares);
    Ok((0..c)
        .map(|u| {
            (0..c)
                .map(|v| (1.0 - k).mul_add(mixed(u, v), k * apart[u][v]))
                .collect()
        })
        .collect())
}

/// Mixing with no tie inside a category, keeping each category's share.
/// Exact when no category holds more than half the tie ends.
fn across(shares: &[f64]) -> Vec<Vec<f64>> {
    let c = shares.len();
    let mut m: Vec<Vec<f64>> = (0..c)
        .map(|u| {
            (0..c)
                .map(|v| if u == v { 0.0 } else { shares[u] * shares[v] })
                .collect()
        })
        .collect();
    for _ in 0..200 {
        let f: Vec<f64> = (0..c)
            .map(|u| {
                let row: f64 = m[u].iter().sum();
                if row > 0.0 {
                    (shares[u] / row).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        for u in 0..c {
            for v in 0..c {
                m[u][v] *= f[u] * f[v];
            }
        }
    }
    m
}

/// Expected tie ends between blocks: keeps each block's stubs and comes as
/// close as it can to every attribute's target at once.
fn fit(stubs: &[f64], labels: &[Vec<usize>], targets: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    let k = stubs.len();
    let total: f64 = stubs.iter().sum();
    if total <= 0.0 {
        return vec![vec![0.0; k]; k];
    }
    let mut e: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| stubs[i] * stubs[j] / total).collect())
        .collect();
    for _ in 0..FIT_ROUNDS {
        for (a, t) in targets.iter().enumerate() {
            let c = t.len();
            let mut now = vec![vec![0.0; c]; c];
            for i in 0..k {
                for j in 0..k {
                    now[labels[i][a]][labels[j][a]] += e[i][j];
                }
            }
            let scale: Vec<Vec<f64>> = (0..c)
                .map(|u| {
                    (0..c)
                        .map(|v| {
                            if now[u][v] > 0.0 {
                                t[u][v] * total / now[u][v]
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect();
            for i in 0..k {
                for j in 0..k {
                    e[i][j] *= scale[labels[i][a]][labels[j][a]];
                }
            }
        }
        let f: Vec<f64> = (0..k)
            .map(|i| {
                let row: f64 = e[i].iter().sum();
                if row > 0.0 {
                    (stubs[i] / row).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        for i in 0..k {
            for j in 0..k {
                e[i][j] *= f[i] * f[j];
            }
        }
    }
    e
}

/// Places the ties in two passes, only between nodes of the same group and
/// never beyond a node's wanted degree.
///
/// 1. Budgets: about `expected[i][j]` ties between blocks `i` and `j`.
/// 2. Filling: degrees still wanted (a household rarely holds exactly the mix
///    the budgets assume) are filled between block pairs the targets allow
///    (`expected > 0`, so a `within = 0` stays exact), preferring the pairs
///    still furthest below their budget.
fn place(
    expected: &[Vec<f64>],
    members: &[Vec<usize>],
    block_of: &[usize],
    wanted: &[usize],
    groups: &[Option<usize>],
    rng: &mut ChaCha8Rng,
) -> Vec<(usize, usize)> {
    let n_blocks = members.len();
    let budget = |i: usize, j: usize| {
        if i == j {
            expected[i][i] / 2.0
        } else {
            expected[i][j]
        }
    };
    let mut placed = vec![vec![0.0f64; n_blocks]; n_blocks];
    let mut degree = vec![0usize; wanted.len()];
    let mut ties: HashSet<(usize, usize)> = HashSet::new();
    let mut by_group: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut in_group: HashMap<usize, Vec<usize>> = HashMap::new();
    for (b, block) in members.iter().enumerate() {
        for &u in block {
            if let Some(g) = groups[u] {
                by_group.entry((b, g)).or_default().push(u);
                in_group.entry(g).or_default().push(u);
            }
        }
    }
    // 1. Budgets.
    let mut pairs: Vec<(usize, usize)> = (0..n_blocks)
        .flat_map(|i| (i..n_blocks).map(move |j| (i, j)))
        .collect();
    pairs.shuffle(rng);
    for (i, j) in pairs {
        let target = budget(i, j);
        let count = whole(target.floor()) + usize::from(rng.gen::<f64>() < target - target.floor());
        'ties: for _ in 0..count {
            for _ in 0..PLACE_TRIES {
                let free: Vec<usize> = members[i]
                    .iter()
                    .copied()
                    .filter(|&u| degree[u] < wanted[u])
                    .collect();
                let Some(&u) = free.choose(rng) else {
                    break 'ties;
                };
                let Some(g) = groups[u] else { continue };
                let others: Vec<usize> = by_group.get(&(j, g)).map_or_else(Vec::new, |vs| {
                    vs.iter()
                        .copied()
                        .filter(|&v| v != u && degree[v] < wanted[v] && !ties.contains(&pair(u, v)))
                        .collect()
                });
                if let Some(&v) = others.choose(rng) {
                    add_tie(u, v, block_of, &mut ties, &mut degree, &mut placed);
                    break;
                }
            }
        }
    }

    // 2. Filling.
    let mut order: Vec<usize> = (0..wanted.len())
        .filter(|&u| degree[u] < wanted[u])
        .collect();
    order.shuffle(rng);
    for u in order {
        let Some(g) = groups[u] else { continue };
        while degree[u] < wanted[u] {
            let candidates: Vec<(usize, f64)> = in_group[&g]
                .iter()
                .copied()
                .filter(|&v| v != u && degree[v] < wanted[v] && !ties.contains(&pair(u, v)))
                .filter_map(|v| {
                    let (bu, bv) = pair(block_of[u], block_of[v]);
                    let target = budget(bu, bv);
                    let short = (target - placed[bu][bv]).max(0.0);
                    (target > 0.0).then(|| (v, 1e-3f64.mul_add(target, short)))
                })
                .collect();
            let Ok(pick) = WeightedIndex::new(candidates.iter().map(|c| c.1)) else {
                break;
            };
            let v = candidates[pick.sample(rng)].0;
            add_tie(u, v, block_of, &mut ties, &mut degree, &mut placed);
        }
    }

    let mut edges: Vec<(usize, usize)> = ties.into_iter().collect();
    edges.sort_unstable();
    edges
}

fn add_tie(
    u: usize,
    v: usize,
    block_of: &[usize],
    ties: &mut HashSet<(usize, usize)>,
    degree: &mut [usize],
    placed: &mut [Vec<f64>],
) {
    ties.insert(pair(u, v));
    degree[u] += 1;
    degree[v] += 1;
    let (i, j) = pair(block_of[u], block_of[v]);
    placed[i][j] += 1.0;
}

const fn pair(u: usize, v: usize) -> (usize, usize) {
    if u < v {
        (u, v)
    } else {
        (v, u)
    }
}

fn to_f64(n: usize) -> f64 {
    n.to_f64().unwrap_or(0.0)
}

fn whole(x: f64) -> usize {
    x.to_usize().unwrap_or(0)
}

// ── Clustering ──────────────────────────────────────────────────────────────

/// Rewires ties to close triangles until the transitivity reaches `target`.
///
/// One step: `v` and `w` share a neighbour (the hub) but no tie; ties `v-x`
/// and `w-y` become `v-w` and `x-y`. Every degree stays the same, and so do
/// the groups (all ties stay inside a group, so the hub, `v`, `w`, `x` and `y`
/// share one). A step is kept when it adds triangles and leaves the pairs of
/// categories of every targeted attribute (`labels`) within `MIX_SLACK` of
/// what they were.
fn close_triangles(
    edges: Vec<(usize, usize)>,
    nodes: usize,
    target: f64,
    labels: &[&[usize]],
    rng: &mut ChaCha8Rng,
) -> Vec<(usize, usize)> {
    let mut ties: HashSet<(usize, usize)> = edges.iter().copied().collect();
    let mut adjacent = vec![Vec::new(); nodes];
    for &(a, b) in &edges {
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let common = |adjacent: &[Vec<usize>], ties: &HashSet<(usize, usize)>, a: usize, b: usize| {
        adjacent[a]
            .iter()
            .filter(|&&c| c != b && ties.contains(&pair(b, c)))
            .count()
    };
    let paths: f64 = adjacent
        .iter()
        .map(|vs| to_f64(vs.len() * vs.len().saturating_sub(1) / 2))
        .sum();
    let mut triangles: usize = edges
        .iter()
        .map(|&(a, b)| common(&adjacent, &ties, a, b))
        .sum::<usize>()
        / 3;
    let hubs: Vec<usize> = (0..nodes).filter(|&h| adjacent[h].len() >= 2).collect();
    if hubs.is_empty() {
        return edges;
    }
    // Change, per attribute, in the count of ties of each pair of categories.
    let mut moved: Vec<HashMap<(usize, usize), i64>> = vec![HashMap::new(); labels.len()];
    let mut distance = vec![0i64; labels.len()];
    let slack = MIX_SLACK * to_f64(edges.len());

    for _ in 0..REWIRE_TRIES * edges.len() {
        if 3.0 * to_f64(triangles) >= target * paths {
            break;
        }
        let hub = *hubs.choose(rng).unwrap_or(&0);
        let mut two = adjacent[hub].choose_multiple(rng, 2).copied();
        let (Some(v), Some(w)) = (two.next(), two.next()) else {
            continue;
        };
        if ties.contains(&pair(v, w)) {
            continue;
        }
        let Some(&x) = adjacent[v].choose(rng) else {
            continue;
        };
        let Some(&y) = adjacent[w].choose(rng) else {
            continue;
        };
        if x == hub || x == w || y == hub || y == v || x == y || ties.contains(&pair(x, y)) {
            continue;
        }
        // The mixing first: it is cheaper.
        let changes = [(v, x, -1), (w, y, -1), (v, w, 1), (x, y, 1)];
        let new_distance: Vec<i64> = labels
            .iter()
            .enumerate()
            .map(|(a, l)| {
                let mut local: HashMap<(usize, usize), i64> = HashMap::new();
                for &(p, q, sign) in &changes {
                    *local.entry(pair(l[p], l[q])).or_insert(0) += sign;
                }
                local.into_iter().fold(distance[a], |sum, (cell, delta)| {
                    let before = moved[a].get(&cell).copied().unwrap_or(0);
                    sum + (before + delta).abs() - before.abs()
                })
            })
            .collect();
        if new_distance.iter().any(|&d| to_f64(whole_i64(d)) > slack) {
            continue;
        }
        // Then the triangles, removing and adding one tie at a time.
        let mut lost = common(&adjacent, &ties, v, x);
        remove_tie(v, x, &mut adjacent, &mut ties);
        lost += common(&adjacent, &ties, w, y);
        remove_tie(w, y, &mut adjacent, &mut ties);
        let mut gained = common(&adjacent, &ties, v, w);
        insert_tie(v, w, &mut adjacent, &mut ties);
        gained += common(&adjacent, &ties, x, y);
        insert_tie(x, y, &mut adjacent, &mut ties);
        if gained > lost {
            triangles = triangles + gained - lost;
            for (a, l) in labels.iter().enumerate() {
                for &(p, q, sign) in &changes {
                    *moved[a].entry(pair(l[p], l[q])).or_insert(0) += sign;
                }
            }
            distance = new_distance;
        } else {
            remove_tie(x, y, &mut adjacent, &mut ties);
            remove_tie(v, w, &mut adjacent, &mut ties);
            insert_tie(w, y, &mut adjacent, &mut ties);
            insert_tie(v, x, &mut adjacent, &mut ties);
        }
    }
    let mut edges: Vec<(usize, usize)> = ties.into_iter().collect();
    edges.sort_unstable();
    edges
}

fn remove_tie(a: usize, b: usize, adjacent: &mut [Vec<usize>], ties: &mut HashSet<(usize, usize)>) {
    ties.remove(&pair(a, b));
    for (from, to) in [(a, b), (b, a)] {
        if let Some(i) = adjacent[from].iter().position(|&c| c == to) {
            adjacent[from].swap_remove(i);
        }
    }
}

fn insert_tie(a: usize, b: usize, adjacent: &mut [Vec<usize>], ties: &mut HashSet<(usize, usize)>) {
    ties.insert(pair(a, b));
    adjacent[a].push(b);
    adjacent[b].push(a);
}

fn whole_i64(x: i64) -> usize {
    x.to_usize().unwrap_or(0)
}

// ── Directed ties ───────────────────────────────────────────────────────────

/// One targeted attribute of a directed graph.
struct Follow<'a> {
    /// Category of every node.
    labels: &'a [usize],
    /// Target within share, or
    within: Option<f64>,
    /// target share of ties per pair of categories, follower first.
    goal: Option<Vec<Vec<f64>>>,
    /// Weight factor per pair of categories, follower first.
    factors: Vec<Vec<f64>>,
}

impl Follow<'_> {
    fn factor(&self, u: usize, v: usize) -> f64 {
        self.factors[self.labels[u]][self.labels[v]]
    }

    /// One round of fitting, given the expected ties `now` per pair of categories.
    fn refit(&mut self, now: &[Vec<f64>]) {
        let total: f64 = now.iter().flatten().sum();
        if total <= 0.0 {
            return;
        }
        let c = self.factors.len();
        if let Some(goal) = &self.goal {
            let goal_total: f64 = goal.iter().flatten().sum();
            for i in 0..c {
                for j in 0..c {
                    self.factors[i][j] *= if now[i][j] > 0.0 && goal_total > 0.0 {
                        (goal[i][j] / goal_total) / (now[i][j] / total)
                    } else {
                        0.0
                    };
                }
            }
        } else if let Some(w) = self.within {
            let share: f64 = (0..c).map(|i| now[i][i]).sum::<f64>() / total;
            for (i, row) in self.factors.iter_mut().enumerate() {
                for (j, f) in row.iter_mut().enumerate() {
                    if i == j && share > 0.0 {
                        *f *= w / share;
                    } else if i != j && share < 1.0 {
                        *f *= (1.0 - w) / (1.0 - share);
                    }
                }
            }
        }
    }
}

/// Directed ties: each node follows its wanted degree of others in its group,
/// drawn without replacement with weight `popularity[v] x factors(u, v)`.
/// The factors (one per targeted attribute and pair of categories, follower
/// first) are fitted so that the expected ties meet every target.
fn directed_ties(
    request: &MixingRequest,
    codes: &Codes<'_>,
    wanted: &[usize],
    groups: &[Option<usize>],
    rng: &mut ChaCha8Rng,
) -> Result<Vec<(usize, usize)>, SynthGraphError> {
    let n = request.nodes;
    let spread = LogNormal::new(0.0, request.popularity_spread)
        .map_err(|e| SynthGraphError::InvalidParameter(e.to_string()))?;
    let popularity: Vec<f64> = (0..n).map(|_| spread.sample(rng)).collect();
    let mut in_group: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (u, g) in groups.iter().enumerate() {
        if let Some(g) = g {
            in_group.entry(*g).or_default().push(u);
        }
    }

    let mut targeted = Vec::new();
    for (name, target) in &request.mixing {
        let (names, per_node) = &codes[name.as_str()];
        let c = names.len();
        let goal = target.matrix.as_ref().map(|matrix| {
            let mut m = vec![vec![0.0; c]; c];
            for (u, row) in matrix {
                for (v, x) in row {
                    if let (Some(i), Some(j)) = (
                        names.iter().position(|s| s == u),
                        names.iter().position(|s| s == v),
                    ) {
                        m[i][j] += x;
                    }
                }
            }
            m
        });
        targeted.push(Follow {
            labels: per_node,
            within: target.within,
            goal,
            factors: vec![vec![1.0; c]; c],
        });
    }
    let weight = |targeted: &[Follow<'_>], u: usize, v: usize| {
        targeted
            .iter()
            .fold(popularity[v], |w, t| w * t.factor(u, v))
    };

    for _ in 0..FIT_ROUNDS {
        let mut now: Vec<Vec<Vec<f64>>> = targeted
            .iter()
            .map(|t| vec![vec![0.0; t.factors.len()]; t.factors.len()])
            .collect();
        for u in 0..n {
            let Some(g) = groups[u] else { continue };
            if wanted[u] == 0 {
                continue;
            }
            let others = in_group[&g].iter().copied().filter(|&v| v != u);
            let z: f64 = others.clone().map(|v| weight(&targeted, u, v)).sum();
            if z <= 0.0 {
                continue;
            }
            for v in others {
                let p = to_f64(wanted[u]) * weight(&targeted, u, v) / z;
                for (a, t) in targeted.iter().enumerate() {
                    now[a][t.labels[u]][t.labels[v]] += p;
                }
            }
        }
        for (t, now) in targeted.iter_mut().zip(&now) {
            t.refit(now);
        }
    }

    let mut edges = Vec::new();
    for u in 0..n {
        let Some(g) = groups[u] else { continue };
        let mut candidates: Vec<(usize, f64)> = in_group[&g]
            .iter()
            .copied()
            .filter(|&v| v != u)
            .map(|v| (v, weight(&targeted, u, v)))
            .filter(|c| c.1 > 0.0)
            .collect();
        for _ in 0..wanted[u] {
            let Ok(pick) = WeightedIndex::new(candidates.iter().map(|c| c.1)) else {
                break;
            };
            let i = pick.sample(rng);
            edges.push((u, candidates[i].0));
            candidates.swap_remove(i);
        }
    }
    edges.sort_unstable();
    Ok(edges)
}

// ── Strengths ───────────────────────────────────────────────────────────────

/// Strength of every tie, see [`StrengthSpec`].
fn tie_strengths(
    spec: &StrengthSpec,
    edges: &[(usize, usize)],
    codes: &Codes<'_>,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<f64>, SynthGraphError> {
    let noise = Normal::new(0.0, spec.spread)
        .map_err(|e| SynthGraphError::InvalidParameter(e.to_string()))?;
    let base = (spec.mean / (1.0 - spec.mean)).ln();
    Ok(edges
        .iter()
        .map(|&(a, b)| {
            let alike: f64 = spec
                .same
                .iter()
                .filter(|(name, _)| {
                    let l = &codes[name.as_str()].1;
                    l[a] == l[b]
                })
                .map(|(_, bonus)| bonus)
                .sum();
            let x = base + noise.sample(rng) + alike;
            1.0 / (1.0 + (-x).exp())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    type Result = std::result::Result<(), Box<dyn std::error::Error>>;

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).to_string()).collect()
    }

    fn within(m: &[Vec<f64>]) -> f64 {
        (0..m.len()).map(|u| m[u][u]).sum()
    }

    #[test]
    fn within_target_keeps_category_shares() -> Result {
        let shares = [0.3, 0.3, 0.4];
        let names = labels(&["a", "b", "c"]);
        for w in [0.0, 0.2, 0.5, 0.9] {
            let m = target_matrix(
                &MixingTarget {
                    within: Some(w),
                    matrix: None,
                },
                &names,
                &shares,
            )?;
            assert!((within(&m) - w).abs() < 1e-6);
            for (u, share) in shares.iter().enumerate() {
                assert!((m[u].iter().sum::<f64>() - share).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn matrix_target_is_symmetric_and_normalised() -> Result {
        let mut row = BTreeMap::new();
        row.insert("male".to_string(), 1.0);
        let mut matrix = BTreeMap::new();
        matrix.insert("female".to_string(), row);
        let names = labels(&["female", "male"]);
        let m = target_matrix(
            &MixingTarget {
                within: None,
                matrix: Some(matrix),
            },
            &names,
            &[0.5, 0.5],
        )?;
        assert!((m[0][1] - 0.5).abs() < 1e-12 && (m[1][0] - 0.5).abs() < 1e-12);
        assert!(within(&m).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn marriage_is_a_matching_across_sexes() -> Result {
        // Ten households of six: three women and three men each.
        let n = 60;
        let request = MixingRequest {
            nodes: n,
            seed: Some(1),
            attributes: BTreeMap::from([(
                "sex".to_string(),
                (0..n)
                    .map(|i| if i % 2 == 0 { "female" } else { "male" }.to_string())
                    .collect(),
            )]),
            groups: Some((0..n).map(|i| Some(format!("h{}", i / 6))).collect()),
            degrees: Some(vec![1; n]),
            max_degree: Some(1),
            mixing: BTreeMap::from([(
                "sex".to_string(),
                MixingTarget {
                    within: Some(0.0),
                    matrix: None,
                },
            )]),
            ..MixingRequest::default()
        };
        let graph = generate_mixing_graph(&request)?;
        assert_eq!(graph.edges.len(), n / 2);
        let mut seen = HashSet::new();
        for &(a, b) in &graph.edges {
            assert!(seen.insert(a) && seen.insert(b), "one partner each");
            assert_ne!(a % 2, b % 2, "a woman and a man");
            assert_eq!(a / 6, b / 6, "same household");
        }
        assert_eq!(graph.mixing["sex"].r, Some(-1.0));
        Ok(())
    }

    #[test]
    fn filling_completes_a_matching_under_a_clashing_target() -> Result {
        // Households of four: an older couple and a younger one, but the men are
        // all "old" and the women all "young", so `age within 0.9` cannot be met.
        let n = 80;
        let request = MixingRequest {
            nodes: n,
            seed: Some(4),
            attributes: BTreeMap::from([
                (
                    "sex".to_string(),
                    (0..n)
                        .map(|i| ["male", "female"][i % 2].to_string())
                        .collect(),
                ),
                (
                    "age".to_string(),
                    (0..n)
                        .map(|i| ["old", "young"][i % 2].to_string())
                        .collect(),
                ),
            ]),
            groups: Some((0..n).map(|i| Some(format!("h{}", i / 4))).collect()),
            degrees: Some(vec![1; n]),
            max_degree: Some(1),
            mixing: BTreeMap::from([
                (
                    "sex".to_string(),
                    MixingTarget {
                        within: Some(0.0),
                        matrix: None,
                    },
                ),
                (
                    "age".to_string(),
                    MixingTarget {
                        within: Some(0.9),
                        matrix: None,
                    },
                ),
            ]),
            ..MixingRequest::default()
        };
        let graph = generate_mixing_graph(&request)?;
        assert_eq!(graph.edges.len(), n / 2, "everyone married");
        assert_eq!(
            graph.mixing["sex"].within,
            Some(0.0),
            "never two of the same sex"
        );
        Ok(())
    }

    #[test]
    fn several_targets_met_at_once() -> Result {
        let n = 1000;
        let pick =
            |values: &[&str], i: usize, step: usize| values[(i / step) % values.len()].to_string();
        let request = MixingRequest {
            nodes: n,
            seed: Some(7),
            attributes: BTreeMap::from([
                (
                    "milieu".to_string(),
                    (0..n).map(|i| pick(&["a", "b", "c"], i, 1)).collect(),
                ),
                (
                    "age".to_string(),
                    (0..n)
                        .map(|i| pick(&["young", "middle", "old"], i, 3))
                        .collect(),
                ),
                (
                    "sex".to_string(),
                    (0..n).map(|i| pick(&["female", "male"], i, 9)).collect(),
                ),
            ]),
            mean_degree: Some(6.0),
            degree_spread: 0.5,
            mixing: BTreeMap::from([
                (
                    "milieu".to_string(),
                    MixingTarget {
                        within: Some(0.7),
                        matrix: None,
                    },
                ),
                (
                    "age".to_string(),
                    MixingTarget {
                        within: Some(0.5),
                        matrix: None,
                    },
                ),
                (
                    "sex".to_string(),
                    MixingTarget {
                        within: Some(0.2),
                        matrix: None,
                    },
                ),
            ]),
            ..MixingRequest::default()
        };
        let graph = generate_mixing_graph(&request)?;
        for (name, report) in &graph.mixing {
            let (target, got) = (
                report.target.ok_or("target")?,
                report.within.ok_or("within")?,
            );
            assert!(
                (target - got).abs() < 0.03,
                "{name}: target {target}, got {got}"
            );
        }
        assert!(
            graph.mixing["sex"].within < graph.mixing["sex"].random,
            "heterophilous on sex"
        );
        let (mean, wanted) = (
            graph.mean_degree.ok_or("mean_degree")?,
            graph.wanted_degree.ok_or("wanted_degree")?,
        );
        assert!((mean - wanted).abs() < 0.5);
        assert_eq!(graph.metrics_skipped, [] as [SkippedMetric; 0]);
        Ok(())
    }

    #[test]
    fn no_node_to_average_over_means_no_mean_degree() -> Result {
        // No node at all, then nodes that all want no tie (a degree drawn around
        // `mean_degree` is at least 1, so that takes explicit zero degrees): the
        // mean degrees are over nothing, so they are absent with their reason,
        // never a made-up 0.
        for request in [
            MixingRequest {
                nodes: 0,
                mean_degree: Some(3.0),
                ..MixingRequest::default()
            },
            MixingRequest {
                nodes: 5,
                degrees: Some(vec![0; 5]),
                ..MixingRequest::default()
            },
        ] {
            let graph = generate_mixing_graph(&request)?;
            assert_eq!(graph.edges, [] as [(usize, usize); 0]);
            assert_eq!(graph.wanted_degree, None);
            assert_eq!(graph.mean_degree, None);
            let skipped: Vec<&str> = graph
                .metrics_skipped
                .iter()
                .map(|skipped| skipped.metric.as_str())
                .collect();
            assert_eq!(skipped, ["wanted_degree", "mean_degree"]);
        }
        Ok(())
    }

    #[test]
    fn ties_stay_inside_groups() -> Result {
        let n = 40;
        let request = MixingRequest {
            nodes: n,
            seed: Some(3),
            groups: Some(
                (0..n)
                    .map(|i| (i % 5 != 0).then(|| format!("g{}", i % 2)))
                    .collect(),
            ),
            mean_degree: Some(4.0),
            ..MixingRequest::default()
        };
        let graph = generate_mixing_graph(&request)?;
        assert_ne!(graph.edges, [] as [(usize, usize); 0]);
        for &(a, b) in &graph.edges {
            assert_eq!(a % 2, b % 2);
            assert!(a % 5 != 0 && b % 5 != 0, "no group, no tie");
        }
        Ok(())
    }

    #[test]
    fn same_seed_same_graph() -> Result {
        let request = MixingRequest {
            nodes: 100,
            seed: Some(5),
            mean_degree: Some(3.0),
            ..MixingRequest::default()
        };
        assert_eq!(
            generate_mixing_graph(&request)?.edges,
            generate_mixing_graph(&request)?.edges
        );
        Ok(())
    }

    fn milieu(n: usize) -> BTreeMap<String, Vec<String>> {
        BTreeMap::from([(
            "milieu".to_string(),
            (0..n).map(|i| ["a", "b", "c"][i % 3].to_string()).collect(),
        )])
    }

    fn within_target(w: f64) -> MixingTarget {
        MixingTarget {
            within: Some(w),
            matrix: None,
        }
    }

    #[test]
    fn clustering_closes_triangles_keeping_degrees_and_mixing() -> Result {
        let n = 400;
        let plain = MixingRequest {
            nodes: n,
            seed: Some(2),
            attributes: milieu(n),
            mean_degree: Some(6.0),
            degree_spread: 0.4,
            mixing: BTreeMap::from([("milieu".to_string(), within_target(0.6))]),
            ..MixingRequest::default()
        };
        let clustered = MixingRequest {
            clustering: Some(0.3),
            ..plain.clone()
        };
        let (a, b) = (
            generate_mixing_graph(&plain)?,
            generate_mixing_graph(&clustered)?,
        );
        let degrees = |edges: &[(usize, usize)]| {
            let mut d = vec![0; n];
            for &(u, v) in edges {
                d[u] += 1;
                d[v] += 1;
            }
            d
        };
        assert_eq!(degrees(&a.edges), degrees(&b.edges), "same degrees");
        let (before, after) = (
            a.clustering.transitivity.ok_or("before")?,
            b.clustering.transitivity.ok_or("after")?,
        );
        assert!(before < 0.1, "random ties close few triangles: {before}");
        assert!((after - 0.3).abs() < 0.03, "target 0.3, got {after}");
        let (wa, wb) = (
            a.mixing["milieu"].within.ok_or("within")?,
            b.mixing["milieu"].within.ok_or("within")?,
        );
        assert!((wa - wb).abs() <= MIX_SLACK, "mixing kept: {wa} then {wb}");
        Ok(())
    }

    #[test]
    fn directed_ties_follow_popularity_and_mixing() -> Result {
        let n = 1000;
        let request = MixingRequest {
            nodes: n,
            seed: Some(3),
            attributes: milieu(n),
            mean_degree: Some(6.0),
            directed: true,
            popularity_spread: 1.5,
            mixing: BTreeMap::from([("milieu".to_string(), within_target(0.7))]),
            ..MixingRequest::default()
        };
        let graph = generate_mixing_graph(&request)?;
        let mut seen = HashSet::new();
        let mut out_degree = vec![0usize; n];
        let mut in_degree = vec![0usize; n];
        for &(from, to) in &graph.edges {
            assert_ne!(from, to);
            assert!(seen.insert((from, to)), "no tie twice");
            out_degree[from] += 1;
            in_degree[to] += 1;
        }
        assert!(out_degree.iter().all(|&d| d == 6), "everyone follows six");
        let top = in_degree.iter().max().copied().unwrap_or(0);
        assert!(top > 8 * 6, "a heavy tail of followers: top {top}");
        let within = graph.mixing["milieu"].within.ok_or("within")?;
        assert!((within - 0.7).abs() < 0.03, "target 0.7, got {within}");
        assert!(graph.directed && graph.clustering.transitivity.is_none());
        Ok(())
    }

    #[test]
    fn strengths_are_higher_between_alike() -> Result {
        let n = 300;
        let request = MixingRequest {
            nodes: n,
            seed: Some(4),
            attributes: milieu(n),
            mean_degree: Some(5.0),
            strength: Some(StrengthSpec {
                mean: 0.4,
                spread: 0.5,
                same: BTreeMap::from([("milieu".to_string(), 1.5)]),
            }),
            ..MixingRequest::default()
        };
        let graph = generate_mixing_graph(&request)?;
        let strengths = graph.strengths.ok_or("strengths")?;
        assert_eq!(strengths.len(), graph.edges.len());
        assert!(strengths.iter().all(|&s| s > 0.0 && s < 1.0));
        let mean = |alike: bool| {
            let xs: Vec<f64> = graph
                .edges
                .iter()
                .zip(&strengths)
                .filter(|((a, b), _)| (a % 3 == b % 3) == alike)
                .map(|(_, &s)| s)
                .collect();
            xs.iter().sum::<f64>() / to_f64(xs.len())
        };
        let (alike, apart) = (mean(true), mean(false));
        assert!(
            (apart - 0.4).abs() < 0.05,
            "typical strength 0.4, got {apart}"
        );
        assert!(alike > apart + 0.2, "alike {alike}, apart {apart}");
        Ok(())
    }

    #[test]
    fn bad_requests_are_refused() {
        let unknown = MixingRequest {
            nodes: 2,
            mean_degree: Some(1.0),
            mixing: BTreeMap::from([(
                "sex".to_string(),
                MixingTarget {
                    within: Some(0.5),
                    matrix: None,
                },
            )]),
            ..MixingRequest::default()
        };
        assert!(generate_mixing_graph(&unknown).is_err());
        let no_degree = MixingRequest {
            nodes: 2,
            ..MixingRequest::default()
        };
        assert!(generate_mixing_graph(&no_degree).is_err());
        let clustered_directed = MixingRequest {
            nodes: 2,
            mean_degree: Some(1.0),
            directed: true,
            clustering: Some(0.2),
            ..MixingRequest::default()
        };
        assert!(generate_mixing_graph(&clustered_directed).is_err());
        let full_strength = MixingRequest {
            nodes: 2,
            mean_degree: Some(1.0),
            strength: Some(StrengthSpec {
                mean: 1.0,
                ..StrengthSpec::default()
            }),
            ..MixingRequest::default()
        };
        assert!(generate_mixing_graph(&full_strength).is_err());
    }
}
