// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! A `StructuredGraph` as the tensors a heterogeneous GNN takes.
//!
//! # Why this exists and what it is not
//!
//! The core is heterogeneous by construction — `kind` is an open string.
//! `plekton.graph/v1` is a homogeneous benchmark format with a homophily and a
//! community label per node; this module exports a separate heterogeneous
//! schema (`plekton.hetero-graph/v1`) matching `PyG`'s `HeteroData` layout.
//!
//! # The canonical triple, which is `PyG`'s shape and not an invention
//!
//! `HeteroData` keys edges by `(source type, relation, destination type)`. Two `co_occurs_with`
//! edges between different type pairs are **different edge types**, and collapsing them is how
//! a heterogeneous graph quietly becomes a homogeneous one with extra labels.
//!
//! # What this refuses to do
//!
//! It does not invent features. A `HeteroData` needs an `x` per node type and there is nothing
//! in an entity graph to put there — the attributes are an open string map, not a feature
//! vector, and turning one into the other is a modelling decision with consequences for what a
//! score means. So this emits **structure only**, and says so in the artifact, because a
//! consumer that found no `x` and silently used ones or degrees would be scoring a model on a
//! representation nobody chose.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::graph::StructuredGraph;

/// The schema this writes.
pub const SCHEMA: &str = "plekton.hetero-graph/v1";

/// One edge type, keyed the way `HeteroData` keys them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EdgeType {
    /// The source node type.
    pub src_type: String,
    /// The relation.
    pub relation: String,
    /// The destination node type.
    pub dst_type: String,
    /// `[2, E]` in `PyG`'s layout: sources then destinations, indexed **within their own node
    /// type**, which is what `HeteroData` expects and what makes this not a homogeneous graph
    /// with labels.
    pub edge_index: [Vec<usize>; 2],
}

/// A graph as heterogeneous tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HeteroGraph {
    /// Format id.
    pub schema: &'static str,
    /// Node ids per type, in the order the indices refer to.
    pub nodes: BTreeMap<String, Vec<String>>,
    /// Edge types, canonically keyed.
    pub edges: Vec<EdgeType>,
    /// Stated in the artifact rather than left for a consumer to discover.
    pub features: FeatureStance,
}

/// What this export says about node features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "stance", rename_all = "snake_case")]
pub enum FeatureStance {
    /// **None, and none should be invented.**
    None {
        /// Why.
        why: &'static str,
    },
}

/// Convert a structured graph into heterogeneous tensors.
///
/// Node types are entity kinds; edge types are the `(kind, relation, kind)` triples that
/// actually occur. Both are sorted, so two runs over one graph produce identical bytes.
#[must_use]
pub fn to_hetero(graph: &StructuredGraph) -> HeteroGraph {
    let mut nodes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entity in &graph.entities {
        nodes
            .entry(entity.kind.clone())
            .or_default()
            .push(entity.id.clone());
    }
    for ids in nodes.values_mut() {
        ids.sort_unstable();
        ids.dedup();
    }

    // id -> (type, index within that type)
    let mut position: BTreeMap<&str, (&str, usize)> = BTreeMap::new();
    for (kind, ids) in &nodes {
        for (index, id) in ids.iter().enumerate() {
            position.insert(id.as_str(), (kind.as_str(), index));
        }
    }

    let mut buckets: BTreeMap<(String, String, String), [Vec<usize>; 2]> = BTreeMap::new();
    for relation in &graph.relations {
        let (Some(&(src_type, src_index)), Some(&(dst_type, dst_index))) = (
            position.get(relation.src.as_str()),
            position.get(relation.dst.as_str()),
        ) else {
            // An edge to an id no entity declared. The core already counts these as unresolved
            // references; a tensor cannot carry one, and inventing a node for it would turn a
            // dangling reference into a thing that exists.
            continue;
        };
        let key = (
            src_type.to_owned(),
            relation.kind.clone(),
            dst_type.to_owned(),
        );
        let entry = buckets
            .entry(key)
            .or_insert_with(|| [Vec::new(), Vec::new()]);
        entry[0].push(src_index);
        entry[1].push(dst_index);
    }

    let edges = buckets
        .into_iter()
        .map(|((src_type, relation, dst_type), edge_index)| EdgeType {
            src_type,
            relation,
            dst_type,
            edge_index,
        })
        .collect();

    HeteroGraph {
        schema: SCHEMA,
        nodes,
        edges,
        features: FeatureStance::None {
            why: "an entity graph's attributes are an open string map, not a feature vector. \
                  Turning one into the other is a modelling decision with consequences for what \
                  a score means, so this emits structure only — a consumer that found no `x` \
                  and silently used ones or degrees would be scoring a model on a \
                  representation nobody chose.",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{to_hetero, SCHEMA};
    use crate::model::{Entity, Extraction, Relation};

    fn graph() -> crate::graph::StructuredGraph {
        let mut extraction = Extraction::default();
        extraction
            .entities
            .push(Entity::new("ip:host:192.0.2.1", "host"));
        extraction
            .entities
            .push(Entity::new("ip:host:192.0.2.2", "host"));
        extraction
            .entities
            .push(Entity::new("lorawan:dev_eui:0102", "dev_eui"));
        extraction
            .entities
            .push(Entity::new("can:arbitration_id:0x0C9", "arbitration_id"));
        extraction.relations.push(Relation::new(
            "ip:host:192.0.2.1",
            "lorawan:dev_eui:0102",
            "co_occurs_with",
        ));
        extraction.relations.push(Relation::new(
            "ip:host:192.0.2.2",
            "can:arbitration_id:0x0C9",
            "co_occurs_with",
        ));
        crate::build(extraction, 10)
    }

    /// The same relation between different type pairs is two edge types, not one.
    #[test]
    fn one_relation_over_two_type_pairs_is_two_canonical_triples() {
        let hetero = to_hetero(&graph());
        assert_eq!(hetero.schema, SCHEMA);
        let triples: Vec<(&str, &str, &str)> = hetero
            .edges
            .iter()
            .map(|edge| {
                (
                    edge.src_type.as_str(),
                    edge.relation.as_str(),
                    edge.dst_type.as_str(),
                )
            })
            .collect();
        assert_eq!(
            triples,
            vec![
                ("host", "co_occurs_with", "arbitration_id"),
                ("host", "co_occurs_with", "dev_eui"),
            ],
            "collapsing these is how a heterogeneous graph quietly becomes a homogeneous one",
        );
    }

    /// Indices are within their own node type, which is what makes it hetero.
    #[test]
    fn indices_are_local_to_their_node_type() {
        let hetero = to_hetero(&graph());
        assert_eq!(hetero.nodes["host"].len(), 2);
        assert_eq!(hetero.nodes["dev_eui"].len(), 1);
        for edge in &hetero.edges {
            for &index in &edge.edge_index[1] {
                assert!(
                    index < hetero.nodes[&edge.dst_type].len(),
                    "a destination index must address its own type's list",
                );
            }
        }
    }

    /// Two runs over one graph produce identical bytes.
    #[test]
    fn the_export_is_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let once = serde_json::to_string(&to_hetero(&graph()))?;
        let twice = serde_json::to_string(&to_hetero(&graph()))?;
        assert_eq!(once, twice);
        Ok(())
    }

    /// The absence of features is stated, not left for a consumer to fill in.
    #[test]
    fn the_artifact_says_it_carries_no_features() -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&to_hetero(&graph()))?;
        assert!(json.contains("\"stance\":\"none\""));
        assert!(
            json.contains("scoring a model on a representation nobody chose"),
            "a consumer that invents `x` must have been told not to: {json}",
        );
        Ok(())
    }
}
