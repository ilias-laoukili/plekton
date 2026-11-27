// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Entities and relations → a measured graph, plus an honest account of what happened.
//!
//! Nothing in this file reads [`Entity::kind`] or [`Relation::kind`]. Kinds are indexed and
//! counted, never compared against a literal — so the same code measures users, items,
//! accounts, devices, or any domain-specific entity types identically.
//! processes, or two kinds invented in a test, and a test proves exactly that.

use std::collections::BTreeMap;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::model::{AdapterError, Attrs, Entity, Extraction, Limits, Relation};

/// How many hubs the summary names by default.
pub const DEFAULT_TOP_HUBS: usize = 10;

/// A built graph: the resolved entities and relations, what was measured, and what was
/// lost on the way in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredGraph {
    /// Entities in index order — index `i` is the node `i` every metric refers to.
    pub entities: Vec<Entity>,
    /// Relations whose endpoints both resolved.
    pub relations: Vec<Relation>,
    /// Structural measurements.
    pub metrics: Metrics,
    /// What the data did not say, and what was dropped.
    pub diagnostics: Diagnostics,
}

/// Structural measurements, all from `graph-metrics` on a plain edge list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Entities after de-duplication.
    pub n_entities: usize,
    /// Relations with both endpoints resolved.
    pub n_relations: usize,
    /// Connected components — the most honest kind of cluster: parts of the graph with no
    /// path between them at all. On a capture this separates unrelated activity without
    /// anything having to be inferred.
    pub connected_components: usize,
    /// Largest component's size, in entities. A graph of 40 000 nodes in one component and
    /// one of 40 000 isolated pairs report the same component *count*; this tells them apart.
    pub largest_component: usize,
    /// Edges over possible edges; `None` below two entities, where no pair can be
    /// related.
    pub density: Option<f64>,
    /// Global clustering coefficient, when defined.
    pub clustering_coefficient: Option<f64>,
    /// Triangle-based transitivity, or `None` when the graph is too large to count
    /// triangles on, or has no open triplet to close.
    ///
    /// The threshold used to be re-applied here, because `graph-metrics` answered `0.0`
    /// above it and that reads as "no triangles" when the truth is "not computed".
    /// It now answers `None` itself, so this is a straight pass-through and the reason
    /// lives in `metrics_skipped`.
    pub transitivity: Option<f64>,
    /// Degree assortativity (Newman's r): do hubs attach to hubs? `None` when there is no
    /// variance to correlate — a regular graph, where every edge joins the same degree pair
    /// — which is not the same statement as an r of zero.
    pub degree_assortativity: Option<f64>,
    /// Fraction of relations joining two entities **of the same kind**, or `None` when
    /// the question has no content.
    ///
    /// Named for what it measures. This is *not* community structure: the partition is the
    /// entity kind the source declared, not one discovered in the graph. It answers "do
    /// hosts talk to hosts, or to domains?" — a real question, and a different one from
    /// "which hosts form a cluster", which nothing here claims to answer.
    ///
    /// `None` when fewer than two distinct kinds are present. Every edge in a single-kind
    /// graph is trivially intra-kind, so 1.0 would read as "strongly assortative" when
    /// there was never an alternative — a degenerate measurement dressed as a finding.
    /// Reporting nothing is the accurate answer.
    pub type_homophily: Option<f64>,
    /// Highest-degree entities, most connected first. Hubs are where a reader should look.
    pub top_hubs: Vec<Hub>,
    /// Entities with no relation at all — present in the data, absent from the structure.
    pub isolated_entities: usize,
    /// Metrics that are `null` because the graph exceeded their computability threshold,
    /// each with the reason.
    ///
    /// Without this a reader cannot tell "not applicable" from "too big to compute", and
    /// both look like an absent field. Triangle counting is quadratic in a node's degree,
    /// so on a large graph it is skipped rather than run for minutes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics_skipped: Vec<SkippedMetric>,
}

/// A measurement that was not taken, and why.
///
/// Re-exported from `graph-metrics`, where it lives now: three crates were writing the same
/// two strings into their own copy of this struct, and the whole point of the type is that
/// one `null` should not be able to mean several different things.
pub use plekton_metrics::skipped::SkippedMetric;

/// One high-degree entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hub {
    /// Entity id.
    pub id: String,
    /// Its kind, echoed so a reader need not cross-reference.
    pub kind: String,
    /// Number of incident relations.
    pub degree: usize,
}

/// What the graph does not show on its own: parser coverage, skipped records,
/// and entity-id conflicts. A graph metric without its ingestion coverage is
/// not complete evidence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Records the adapter understood.
    pub records_parsed: usize,
    /// Records it did not, by reason.
    pub records_skipped: BTreeMap<String, usize>,
    /// Total of the above.
    pub records_skipped_total: usize,
    /// Entities per kind, exact.
    pub entity_counts_by_kind: BTreeMap<String, usize>,
    /// Relations per kind, exact — counting only those that survived resolution.
    pub relation_counts_by_kind: BTreeMap<String, usize>,
    /// Entity ids that appeared more than once. The first definition wins; this says how
    /// often that choice was made.
    pub duplicate_entity_ids: usize,
    /// Duplicate ids whose attributes actually disagreed — the subset of the above worth
    /// worrying about, because something upstream is calling two different things the same.
    pub conflicting_entity_ids: Vec<String>,
    /// Relations naming an entity that was never declared. **Dropped**, and counted here:
    /// inventing the missing endpoint would fabricate structure that no record supports.
    pub unresolved_references: usize,
    /// Up to a few examples of the ids that went unresolved, to make the count actionable.
    pub unresolved_examples: Vec<String>,
}

/// How many conflicting/unresolved examples to keep. Enough to act on, not a second dataset.
const MAX_EXAMPLES: usize = 10;

/// Build a measured graph from an adapter's output, under [`Limits`].
///
/// De-duplicates entities by id (first definition wins), resolves relations against them,
/// and measures the result. Relations naming an unknown entity are dropped and counted.
///
/// # Errors
/// [`AdapterError::TooLarge`] when the de-duplicated entity count exceeds
/// `limits.max_entities`. This is a *separate* bound from the record limit: one record can
/// yield several entities, and it is the graph rather than the file that has to be resident.
/// Nothing is measured above it — see [`Limits`] for why a prefix would mislead.
pub fn build_bounded(
    extraction: Extraction,
    top_hubs: usize,
    limits: Limits,
) -> Result<StructuredGraph, AdapterError> {
    let Extraction {
        entities: raw_entities,
        relations: raw_relations,
        records_parsed,
        records_skipped,
    } = extraction;

    let mut diagnostics = Diagnostics {
        records_parsed,
        records_skipped_total: records_skipped.values().sum(),
        records_skipped,
        ..Diagnostics::default()
    };

    // ── Entities: de-duplicate, keeping the first definition ──────────────────
    let mut index_of: HashMap<String, usize> = HashMap::new();
    let mut entities: Vec<Entity> = Vec::with_capacity(raw_entities.len());
    for entity in raw_entities {
        let Some(&existing) = index_of.get(&entity.id) else {
            if entities.len() >= limits.max_entities {
                return Err(AdapterError::TooLarge {
                    seen: entities.len() + 1,
                    limit: limits.max_entities,
                    unit: "entities",
                    knob: "max_entities",
                });
            }
            index_of.insert(entity.id.clone(), entities.len());
            entities.push(entity);
            continue;
        };
        diagnostics.duplicate_entity_ids += 1;
        // A repeat is ordinary — the same host appears on every line. A repeat that
        // *disagrees* is not, so only that is surfaced.
        let disagrees = entities
            .get(existing)
            .is_some_and(|kept: &Entity| attrs_conflict(&kept.attrs, &entity.attrs));
        if disagrees && diagnostics.conflicting_entity_ids.len() < MAX_EXAMPLES {
            diagnostics.conflicting_entity_ids.push(entity.id);
        }
    }

    // ── Relations: resolve endpoints, drop and report what does not resolve ───
    let mut relations: Vec<Relation> = Vec::with_capacity(raw_relations.len());
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(raw_relations.len());
    for relation in raw_relations {
        let (Some(&src), Some(&dst)) = (index_of.get(&relation.src), index_of.get(&relation.dst))
        else {
            diagnostics.unresolved_references += 1;
            if diagnostics.unresolved_examples.len() < MAX_EXAMPLES {
                let missing = if index_of.contains_key(&relation.src) {
                    relation.dst
                } else {
                    relation.src
                };
                diagnostics.unresolved_examples.push(missing);
            }
            continue;
        };
        edges.push((src, dst));
        relations.push(relation);
    }

    for entity in &entities {
        *diagnostics
            .entity_counts_by_kind
            .entry(entity.kind.clone())
            .or_insert(0) += 1;
    }
    for relation in &relations {
        *diagnostics
            .relation_counts_by_kind
            .entry(relation.kind.clone())
            .or_insert(0) += 1;
    }

    let metrics = measure(&entities, &edges, top_hubs);
    Ok(StructuredGraph {
        entities,
        relations,
        metrics,
        diagnostics,
    })
}

/// [`build_bounded`] with the default limits, for callers that hold small inputs.
///
/// # Panics
/// Never in practice — but it discards the limit error, so prefer [`build_bounded`] on any
/// input whose size is not already known.
#[must_use]
pub fn build(extraction: Extraction, top_hubs: usize) -> StructuredGraph {
    let unbounded = Limits {
        max_records: usize::MAX,
        max_entities: usize::MAX,
    };
    // Unreachable with usize::MAX limits, but kept total rather than unwrapped: this is a
    // convenience wrapper, and a panic here would be a worse failure than an empty graph.
    build_bounded(extraction, top_hubs, unbounded).unwrap_or_else(|_| StructuredGraph {
        entities: Vec::new(),
        relations: Vec::new(),
        metrics: measure(&[], &[], top_hubs),
        diagnostics: Diagnostics::default(),
    })
}

/// Do two attribute sets state different values for the same key?
///
/// Absence never conflicts with presence: one record carrying a field another omits is
/// normal. Only a disagreement on a shared key means two sources are describing the same id
/// differently.
fn attrs_conflict(a: &Attrs, b: &Attrs) -> bool {
    a.iter()
        .any(|(key, value)| b.get(key).is_some_and(|other| other != value))
}

/// The metrics that the size of the graph alone leaves undefined or too costly, each
/// with its reason: what `n` entities, `n_relations` relations and `n_kinds` kinds
/// rule out before anything is measured.
fn skipped_by_size(n: usize, n_relations: usize, n_kinds: usize) -> Vec<SkippedMetric> {
    let mut skipped = Vec::new();
    if n < 2 {
        skipped.push(SkippedMetric::undefined(
            "density",
            "fewer than two entities, so no pair of them can be related",
        ));
    }
    if n == 0 {
        skipped.push(SkippedMetric::undefined(
            "clustering_coefficient",
            "there is no entity, so there is no mean to take",
        ));
    }
    if n > plekton_metrics::clustering::CLUSTERING_THRESHOLD {
        skipped.push(SkippedMetric::too_large(
            "clustering_coefficient",
            n,
            plekton_metrics::clustering::CLUSTERING_THRESHOLD,
        ));
    }
    if n > plekton_metrics::transitivity::TRANSITIVITY_THRESHOLD {
        skipped.push(SkippedMetric::too_large(
            "transitivity",
            n,
            plekton_metrics::transitivity::TRANSITIVITY_THRESHOLD,
        ));
    }
    if n_kinds < 2 {
        skipped.push(SkippedMetric::undefined(
            "type_homophily",
            "only one entity kind is present, so every relation is trivially intra-kind",
        ));
    } else if n_relations == 0 {
        skipped.push(SkippedMetric::undefined(
            "type_homophily",
            "there are no relations to measure",
        ));
    }
    skipped
}

/// Measure the graph. Every number here comes from `graph-metrics` or is a count.
fn measure(entities: &[Entity], edges: &[(usize, usize)], top_hubs: usize) -> Metrics {
    let n = entities.len();

    // The kind partition, passed *in* to homophily exactly as `graph-metrics` expects an
    // assignment. Keeping it a parameter is what will let a real community detector replace
    // it later as a change of data rather than of architecture.
    let mut kind_index: HashMap<&str, usize> = HashMap::new();
    let assignment: Vec<usize> = entities
        .iter()
        .map(|entity| {
            let next = kind_index.len();
            *kind_index.entry(entity.kind.as_str()).or_insert(next)
        })
        .collect();

    let mut degree = vec![0usize; n];
    for &(src, dst) in edges {
        if let Some(slot) = degree.get_mut(src) {
            *slot += 1;
        }
        if src != dst {
            if let Some(slot) = degree.get_mut(dst) {
                *slot += 1;
            }
        }
    }

    // Stable order: degree descending, then id ascending. The tie-break is what makes the
    // reported hubs identical across runs on equal-degree nodes.
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| {
        let (da, db) = (degree.get(a).copied(), degree.get(b).copied());
        db.cmp(&da).then_with(|| {
            let left = entities.get(a).map(|e| e.id.as_str());
            let right = entities.get(b).map(|e| e.id.as_str());
            left.cmp(&right)
        })
    });
    let top: Vec<Hub> = ranked
        .iter()
        .take(top_hubs)
        .filter_map(|&i| {
            let entity = entities.get(i)?;
            Some(Hub {
                id: entity.id.clone(),
                kind: entity.kind.clone(),
                degree: degree.get(i).copied().unwrap_or(0),
            })
        })
        .collect();

    // Recorded so a reader can tell "not applicable" from "too big to compute" — both of
    // which otherwise arrive as a bare null.
    let mut metrics_skipped = skipped_by_size(n, edges.len(), kind_index.len());

    let transitivity = plekton_metrics::transitivity::transitivity(n, edges);
    if transitivity.is_none() && n <= plekton_metrics::transitivity::TRANSITIVITY_THRESHOLD {
        metrics_skipped.push(SkippedMetric::undefined(
            "transitivity",
            "no node has two neighbours, so there is no open triplet to be a fraction of",
        ));
    }
    let degree_assortativity = plekton_metrics::assortativity::degree_assortativity(n, edges);
    if degree_assortativity.is_none() {
        metrics_skipped.push(SkippedMetric::undefined(
            "degree_assortativity",
            "every relation joins the same degree pair, so there is no variance to correlate",
        ));
    }

    Metrics {
        n_entities: n,
        n_relations: edges.len(),
        connected_components: plekton_metrics::components::count_components(n, edges),
        largest_component: largest_component(n, edges),
        density: plekton_metrics::density::density(n, edges.len()),
        clustering_coefficient: plekton_metrics::clustering::compute_clustering_coefficient(
            n, edges,
        ),
        transitivity,
        degree_assortativity,
        // Only meaningful with something to be homophilous *against*. `compute_homophily`
        // measures any labelling, a single kind included, where every relation is
        // trivially intra-kind and it says 1.0 — a degenerate measurement dressed as a
        // finding. This guard is what keeps that case out.
        type_homophily: (kind_index.len() >= 2)
            .then(|| plekton_metrics::homophily::compute_homophily(edges, &assignment))
            .flatten(),
        top_hubs: top,
        isolated_entities: degree.iter().filter(|&&d| d == 0).count(),
        metrics_skipped,
    }
}

/// Union-find root, with path halving so repeated lookups stay near-flat.
fn find(parent: &mut [usize], mut node: usize) -> usize {
    while parent.get(node).copied().unwrap_or(node) != node {
        let grandparent = parent
            .get(parent.get(node).copied().unwrap_or(node))
            .copied()
            .unwrap_or(node);
        if let Some(slot) = parent.get_mut(node) {
            *slot = grandparent;
        }
        node = grandparent;
    }
    node
}

/// Size of the biggest connected component, via union-find.
///
/// `graph-metrics` counts components but does not size them, and the difference between one
/// giant component and many small ones is most of what "is this graph connected?" means.
fn largest_component(n: usize, edges: &[(usize, usize)]) -> usize {
    if n == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..n).collect();

    for &(a, b) in edges {
        if a >= n || b >= n {
            continue;
        }
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            if let Some(slot) = parent.get_mut(ra) {
                *slot = rb;
            }
        }
    }

    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for node in 0..n {
        let root = find(&mut parent, node);
        *sizes.entry(root).or_insert(0) += 1;
    }
    sizes.values().copied().max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two entities joined by one relation, expressed in whatever kinds the caller likes.
    fn pair(kind_a: &str, kind_b: &str, relation: &str) -> Extraction {
        Extraction {
            entities: vec![Entity::new("a", kind_a), Entity::new("b", kind_b)],
            relations: vec![Relation::new("a", "b", relation)],
            records_parsed: 1,
            records_skipped: BTreeMap::new(),
        }
    }

    /// **The genericity proof.** The same shape measured under two completely different kind vocabularies
    /// invented ones must give identical metrics. If a per-type branch ever appears in this
    /// file, these two stop agreeing.
    #[test]
    fn the_core_measures_invented_kinds_exactly_as_it_measures_real_ones() {
        let realistic = build(pair("host", "domain", "queried"), DEFAULT_TOP_HUBS);
        let invented = build(pair("widget", "sprocket", "flanged"), DEFAULT_TOP_HUBS);

        // Compared as serialised documents rather than field by field, so EVERY measurement
        // is covered — including any added later — and exactly, not within a tolerance.
        //
        // `top_hubs` is excluded because it deliberately echoes each hub's kind back to the
        // reader; those strings differing is the data, not the behaviour. Everything the
        // core *computed* must be bit-identical, and the hubs are compared by id and degree
        // just below.
        let strip = |metrics: &Metrics| {
            let Ok(mut value) = serde_json::to_value(metrics) else {
                panic!("metrics must serialise");
            };
            if let Some(object) = value.as_object_mut() {
                object.remove("top_hubs");
            }
            value.to_string()
        };
        assert_eq!(
            strip(&realistic.metrics),
            strip(&invented.metrics),
            "the core computed something different for two kinds"
        );

        let shape = |metrics: &Metrics| -> Vec<(String, usize)> {
            metrics
                .top_hubs
                .iter()
                .map(|hub| (hub.id.clone(), hub.degree))
                .collect()
        };
        assert_eq!(
            shape(&realistic.metrics),
            shape(&invented.metrics),
            "hub ranking must not depend on the kinds"
        );
    }

    #[test]
    fn a_relation_naming_an_undeclared_entity_is_dropped_and_reported() {
        // Silently keeping it would require inventing the missing endpoint, which would
        // fabricate a node no record supports.
        let extraction = Extraction {
            entities: vec![Entity::new("a", "host")],
            relations: vec![Relation::new("a", "ghost", "connected_to")],
            records_parsed: 1,
            records_skipped: BTreeMap::new(),
        };
        let graph = build(extraction, DEFAULT_TOP_HUBS);

        assert_eq!(graph.metrics.n_relations, 0);
        assert_eq!(graph.diagnostics.unresolved_references, 1);
        assert_eq!(graph.diagnostics.unresolved_examples, vec!["ghost"]);
    }

    #[test]
    fn a_repeated_id_is_deduplicated_and_only_a_disagreement_is_flagged() {
        let extraction = Extraction {
            entities: vec![
                Entity::new("h1", "host").with("os", "linux"),
                // Same value: an ordinary repeat, not a conflict.
                Entity::new("h1", "host").with("os", "linux"),
                // Different value for a shared key: something upstream disagrees.
                Entity::new("h1", "host").with("os", "windows"),
                // A key the first record lacked: absence never conflicts with presence.
                Entity::new("h1", "host").with("role", "server"),
            ],
            relations: Vec::new(),
            records_parsed: 4,
            records_skipped: BTreeMap::new(),
        };
        let graph = build(extraction, DEFAULT_TOP_HUBS);

        assert_eq!(graph.metrics.n_entities, 1);
        assert_eq!(graph.diagnostics.duplicate_entity_ids, 3);
        assert_eq!(graph.diagnostics.conflicting_entity_ids, vec!["h1"]);
    }

    #[test]
    fn hubs_are_ranked_by_degree_and_tie_broken_by_id() {
        let extraction = Extraction {
            entities: vec![
                Entity::new("hub", "host"),
                Entity::new("b", "host"),
                Entity::new("a", "host"),
            ],
            relations: vec![
                Relation::new("hub", "a", "connected_to"),
                Relation::new("hub", "b", "connected_to"),
            ],
            records_parsed: 2,
            records_skipped: BTreeMap::new(),
        };
        let graph = build(extraction, DEFAULT_TOP_HUBS);

        let ranked: Vec<&str> = graph
            .metrics
            .top_hubs
            .iter()
            .map(|h| h.id.as_str())
            .collect();
        // hub has degree 2; a and b both have 1, so id order decides — deterministically.
        assert_eq!(ranked, vec!["hub", "a", "b"]);
        assert_eq!(graph.metrics.top_hubs[0].degree, 2);
    }

    #[test]
    fn islands_are_visible_as_components_and_isolated_entities() {
        let extraction = Extraction {
            entities: vec![
                Entity::new("a", "host"),
                Entity::new("b", "host"),
                Entity::new("lonely", "host"),
            ],
            relations: vec![Relation::new("a", "b", "connected_to")],
            records_parsed: 1,
            records_skipped: BTreeMap::new(),
        };
        let graph = build(extraction, DEFAULT_TOP_HUBS);

        assert_eq!(graph.metrics.connected_components, 2);
        assert_eq!(graph.metrics.largest_component, 2);
        assert_eq!(graph.metrics.isolated_entities, 1);
    }

    #[test]
    fn homophily_is_reported_only_when_there_are_kinds_to_compare() {
        // One kind: every edge is trivially intra-kind, so any number would overstate.
        let single = build(pair("host", "host", "connected_to"), DEFAULT_TOP_HUBS);
        assert_eq!(single.metrics.type_homophily, None);

        // Two kinds, one crossing edge: a real 0.0.
        let cross = build(pair("host", "domain", "queried"), DEFAULT_TOP_HUBS);
        let Some(homophily) = cross.metrics.type_homophily else {
            panic!("two kinds means the measure has content");
        };
        assert!(homophily.abs() < f64::EPSILON, "{homophily}");
    }

    #[test]
    fn homophily_counts_intra_kind_edges_when_kinds_differ() {
        let extraction = Extraction {
            entities: vec![
                Entity::new("h1", "host"),
                Entity::new("h2", "host"),
                Entity::new("d1", "domain"),
            ],
            relations: vec![
                Relation::new("h1", "h2", "connected_to"), // intra-kind
                Relation::new("h1", "d1", "queried"),      // cross-kind
            ],
            records_parsed: 2,
            records_skipped: BTreeMap::new(),
        };
        let graph = build(extraction, DEFAULT_TOP_HUBS);
        let Some(homophily) = graph.metrics.type_homophily else {
            panic!("two kinds means the measure has content");
        };
        assert!((homophily - 0.5).abs() < f64::EPSILON, "{homophily}");
    }

    #[test]
    fn an_empty_extraction_measures_without_panicking() {
        let graph = build(Extraction::default(), DEFAULT_TOP_HUBS);
        assert_eq!(graph.metrics.n_entities, 0);
        assert_eq!(graph.metrics.connected_components, 0);
        assert_eq!(graph.metrics.largest_component, 0);
        // Nothing to measure: no density and no mean clustering, each with its reason,
        // rather than a made-up 0.0.
        assert_eq!(graph.metrics.density, None);
        assert_eq!(graph.metrics.clustering_coefficient, None);
        let skipped: Vec<&str> = graph
            .metrics
            .metrics_skipped
            .iter()
            .map(|skipped| skipped.metric.as_str())
            .collect();
        assert!(skipped.contains(&"density"), "{skipped:?}");
        assert!(skipped.contains(&"clustering_coefficient"), "{skipped:?}");
    }

    #[test]
    fn two_kinds_without_relations_say_why_there_is_no_type_homophily() {
        let extraction = Extraction {
            entities: vec![Entity::new("h1", "host"), Entity::new("d1", "domain")],
            ..Extraction::default()
        };
        let graph = build(extraction, DEFAULT_TOP_HUBS);
        assert_eq!(graph.metrics.type_homophily, None);
        assert!(graph
            .metrics
            .metrics_skipped
            .iter()
            .any(|skipped| skipped.metric == "type_homophily"));
        // Two entities: a density exists, and it is 0.
        assert_eq!(graph.metrics.density, Some(0.0));
    }

    #[test]
    fn skipped_records_survive_into_the_diagnostics() {
        let mut extraction = Extraction::default();
        extraction.skip("malformed json");
        extraction.skip("malformed json");
        let graph = build(extraction, DEFAULT_TOP_HUBS);

        assert_eq!(graph.diagnostics.records_skipped_total, 2);
        assert_eq!(graph.diagnostics.records_skipped["malformed json"], 2);
    }
}
