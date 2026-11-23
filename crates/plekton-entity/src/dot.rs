// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! A measured graph → `GraphViz` DOT.
//!
//! # What this is for, and the trap it documents
//!
//! `dot` — the layout engine people reach for because it shares the format's name — draws
//! *hierarchies*. Handed a few hundred interconnected nodes it produces the picture everyone
//! has seen once and never wants again: a solid bar of overlapping labels with lines swept
//! across it. The graphs this crate builds are not hierarchies, so the header written into
//! every file names `sfdp` instead, which is a force-directed engine and puts communities
//! where a reader expects to find them.
//!
//! # Honesty survives the rendering
//!
//! A drawing is the most persuasive form a result takes and the easiest one to over-read, so
//! the DOT carries its own caveats as comments at the top of the file: how many entities and
//! relations there are, how many records went unread, and — the one that matters most —
//! whether relations were **dropped** because they named an entity nothing declared. A picture
//! of a graph that silently lost a tenth of its edges is a picture of a different graph.

use std::fmt::Write as _;

use crate::graph::StructuredGraph;

/// Colours assigned to entity kinds, in first-seen order.
///
/// Deliberately few and deliberately dull: a palette that runs out is better than one that
/// invites reading meaning into hue. Beyond these the kinds cycle, and the legend — which is
/// written into the graph — is what actually distinguishes them.
const PALETTE: [&str; 8] = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f",
];

/// Escape a string for a DOT quoted identifier.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The part of an id a reader reads, for the label.
///
/// A qualified id is `<domain>:<kind>:<value>` and only the value distinguishes one node from its
/// neighbour — the domain and the kind are constant across a whole colour, and the legend already
/// names them. A bare id has no such structure and is returned whole rather than truncated at
/// some length, because cutting an address in half makes two nodes look like one.
fn short_label(id: &str) -> &str {
    crate::namespace::parse(id).map_or(id, |qualified| qualified.value)
}

/// How many relations a node touches, when one node touches enough of them to flatten the layout.
///
/// The threshold is half. Not tuned: at half the edges, removing that one node disconnects more
/// of the drawing than it leaves, which is the point at which "graph" stops describing what a
/// reader will see. Below it, saying nothing is right.
fn dominating_hub(graph: &StructuredGraph) -> Option<(&str, usize)> {
    let hub = graph.metrics.top_hubs.first()?;
    (graph.metrics.n_relations > 0 && hub.degree * 2 > graph.metrics.n_relations)
        .then_some((hub.id.as_str(), hub.degree))
}

/// Render `graph` as an undirected DOT document.
///
/// Undirected because the core stores no direction it could honour: [`crate::Relation`] has a
/// `src` and a `dst`, but whether that ordering means anything is the *adapter's* business and
/// is not recorded. Drawing arrowheads would assert a direction the data does not carry.
#[must_use]
pub fn to_dot(graph: &StructuredGraph) -> String {
    let mut out = String::new();
    let diagnostics = &graph.diagnostics;

    // ── The caveats, before the picture ──────────────────────────────────────
    out.push_str("// Rendered by plekton. Lay this out with a FORCE-DIRECTED engine:\n");
    out.push_str("//     sfdp -Tsvg this.dot -o graph.svg     (or: neato, fdp)\n");
    out.push_str("// `dot` itself draws hierarchies and will render this as an unreadable bar.\n");
    out.push_str("//\n");
    let _ = writeln!(
        out,
        "// {} entities, {} relations, {} connected component(s).",
        graph.metrics.n_entities, graph.metrics.n_relations, graph.metrics.connected_components
    );
    if diagnostics.records_skipped_total > 0 {
        let _ = writeln!(
            out,
            "// INCOMPLETE: {} record(s) went unread. This draws what was parsed, not the log.",
            diagnostics.records_skipped_total
        );
    }
    if diagnostics.unresolved_references > 0 {
        let _ = writeln!(
            out,
            "// EDGES DROPPED: {} relation(s) named an entity nothing declared, and were \
             discarded rather than given an invented endpoint.",
            diagnostics.unresolved_references
        );
    }
    // A star flattens whatever engine is pointed at it, and `structure --source probe` produces
    // one by construction: the probe is adjacent to every spoke. Said HERE, beside the layout
    // advice, because the point is to know before spending a render rather than after seeing the
    // result. Only when a node actually dominates — a standing warning is one nobody reads.
    if let Some(hub) = dominating_hub(graph) {
        let _ = writeln!(
            out,
            "// ONE HUB: `{}` touches {} of the {} relations here, so this will lay out as a \
             star whatever engine draws it. Drop it to see the rest.",
            hub.0, hub.1, graph.metrics.n_relations
        );
    }

    out.push_str("graph plekton {\n");
    out.push_str("  graph [overlap=false, splines=true, bgcolor=\"transparent\"];\n");
    out.push_str("  node  [shape=ellipse, style=filled, fontname=\"Helvetica\", fontsize=10];\n");
    out.push_str("  edge  [color=\"#88888899\", fontname=\"Helvetica\", fontsize=8];\n\n");

    // Colour by kind, assigned in the order kinds first appear so the same input always
    // yields the same file — a diff between two renderings has to mean the graph changed.
    let mut kinds: Vec<&str> = Vec::new();
    for entity in &graph.entities {
        if !kinds.contains(&entity.kind.as_str()) {
            kinds.push(entity.kind.as_str());
        }
    }
    let colour_of = |kind: &str| -> &'static str {
        kinds
            .iter()
            .position(|known| *known == kind)
            .map_or(PALETTE[0], |index| PALETTE[index % PALETTE.len()])
    };

    for (index, entity) in graph.entities.iter().enumerate() {
        // The LABEL is what a reader reads and the TOOLTIP is where the identity goes. An id is
        // the widest string in the graph — a BLE address, an IPv4 host, and after namespacing a
        // domain and a kind glued to the front of it — and every node wearing one is what made
        // the drawing unreadable on real data. The kind moves into the tooltip beside it, where
        // it says more than it did alone: the colour and the legend already carry the kind.
        let _ = writeln!(
            out,
            "  n{index} [label=\"{}\", fillcolor=\"{}\", tooltip=\"{} ({})\"];",
            escape(short_label(&entity.id)),
            colour_of(&entity.kind),
            escape(&entity.id),
            escape(&entity.kind),
        );
    }
    out.push('\n');

    // Relations reference ids; the drawing needs indices.
    let index_of: std::collections::HashMap<&str, usize> = graph
        .entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity.id.as_str(), index))
        .collect();

    for relation in &graph.relations {
        let (Some(src), Some(dst)) = (
            index_of.get(relation.src.as_str()),
            index_of.get(relation.dst.as_str()),
        ) else {
            // Unreachable for a built graph — `build` drops unresolved relations and counts
            // them — but skipping beats emitting an edge to a node that is not in the file.
            continue;
        };
        let _ = writeln!(
            out,
            "  n{src} -- n{dst} [label=\"{}\"];",
            escape(&relation.kind)
        );
    }

    // The legend is a subgraph rather than a comment: the colours mean nothing without it,
    // and a legend that lives outside the image is a legend nobody has when they need it.
    if kinds.len() > 1 {
        out.push_str("\n  subgraph cluster_legend {\n");
        out.push_str("    label=\"entity kinds\"; fontsize=9; color=\"#cccccc\";\n");
        for (index, kind) in kinds.iter().enumerate() {
            let _ = writeln!(
                out,
                "    legend{index} [label=\"{}\", fillcolor=\"{}\", shape=box];",
                escape(kind),
                PALETTE[index % PALETTE.len()]
            );
        }
        out.push_str("  }\n");
    }

    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build, Entity, Extraction, Relation, DEFAULT_TOP_HUBS};

    fn graph_with(unresolved: bool) -> StructuredGraph {
        let mut relations = vec![Relation::new("10.0.0.1", "evil.example", "queried")];
        if unresolved {
            relations.push(Relation::new("10.0.0.1", "never-declared", "queried"));
        }
        build(
            Extraction {
                entities: vec![
                    Entity::new("10.0.0.1", "host"),
                    Entity::new("evil.example", "domain"),
                ],
                relations,
                records_parsed: 1,
                ..Extraction::default()
            },
            DEFAULT_TOP_HUBS,
        )
    }

    /// A star: one probe adjacent to everything, which is what `structure --source probe`
    /// produces and what flattens a force-directed layout into a line.
    fn star(spokes: usize) -> StructuredGraph {
        let mut entities = vec![Entity::new("radio:probe:radio", "probe")];
        let mut relations = Vec::new();
        for index in 0..spokes {
            let id = format!("radio:ble_address:00:00:5E:00:53:{index:02X}");
            entities.push(Entity::new(&id, "ble_address"));
            relations.push(Relation::new("radio:probe:radio", &id, "observed"));
        }
        build(
            Extraction {
                entities,
                relations,
                records_parsed: spokes,
                ..Extraction::default()
            },
            DEFAULT_TOP_HUBS,
        )
    }

    /// Entry 67: on real data every label was a full address, and after namespacing it is a full
    /// address with a domain and a kind glued to the front of it.
    #[test]
    fn a_label_is_the_part_a_reader_reads_and_the_tooltip_keeps_the_rest() {
        let dot = to_dot(&star(2));
        assert!(
            dot.contains(r#"label="00:00:5E:00:53:00""#),
            "a node's label must be the value half of its id, not the whole namespaced id:\n{dot}"
        );
        assert!(
            !dot.contains(r#"label="radio:ble_address:00:00:5E:00:53:00""#),
            "the namespaced id is in every label, which is the widest possible string:\n{dot}"
        );
        assert!(
            dot.contains("radio:ble_address:00:00:5E:00:53:00"),
            "and the whole id must still be somewhere — the tooltip — because it is the identity"
        );
    }

    /// The other half of entry 67: a star renders as a bar however good the engine is, and the
    /// file said nothing about it.
    #[test]
    fn a_dominating_hub_is_named_before_anyone_spends_a_render_on_it() {
        let dot = to_dot(&star(12));
        assert!(
            dot.contains("radio:probe:radio"),
            "the hub must be named:\n{dot}"
        );
        let header: String = dot
            .lines()
            .take_while(|line| line.starts_with("//"))
            .collect();
        assert!(
            header.contains("radio:probe:radio"),
            "and named in the HEADER, beside the layout advice, because the point is to know \
             before rendering rather than after:\n{header}"
        );
    }

    /// And not on a graph where no node dominates — a standing warning is one nobody reads.
    #[test]
    fn a_graph_with_no_dominating_hub_says_nothing_about_one() {
        let dot = to_dot(&graph_with(false));
        let header: String = dot
            .lines()
            .take_while(|line| line.starts_with("//"))
            .collect();
        assert!(
            !header.contains("of the edges"),
            "two nodes and one edge is not a star:\n{header}"
        );
    }

    #[test]
    fn the_document_is_undirected_and_well_formed() {
        let dot = to_dot(&graph_with(false));
        assert!(dot.starts_with("//"), "the caveats come before the picture");
        assert!(dot.contains("graph plekton {"));
        assert!(
            dot.contains(" -- "),
            "undirected: the data carries no direction"
        );
        assert!(
            !dot.contains(" -> "),
            "an arrowhead would assert a direction"
        );
        assert_eq!(
            dot.matches('{').count(),
            dot.matches('}').count(),
            "unbalanced braces would not parse"
        );
    }

    /// **The property that matters.** A picture of a graph that lost edges must say so, on
    /// the picture — a drawing is the most persuasive form a result takes.
    #[test]
    fn dropped_edges_are_declared_in_the_file() {
        let dot = to_dot(&graph_with(true));
        assert!(
            dot.contains("EDGES DROPPED"),
            "a rendering that silently lost an edge draws a different graph:\n{dot}"
        );
    }

    /// The layout trap is documented where somebody will actually meet it.
    #[test]
    fn the_file_names_the_engine_that_will_not_embarrass_the_reader() {
        let dot = to_dot(&graph_with(false));
        assert!(
            dot.contains("sfdp"),
            "the force-directed engine must be named"
        );
    }

    #[test]
    fn rendering_is_deterministic() {
        assert_eq!(to_dot(&graph_with(false)), to_dot(&graph_with(false)));
    }

    /// A label carrying a quote must not break out of its own attribute.
    #[test]
    fn identifiers_are_escaped() {
        let graph = build(
            Extraction {
                entities: vec![Entity::new("say \"hi\"", "host")],
                records_parsed: 1,
                ..Extraction::default()
            },
            DEFAULT_TOP_HUBS,
        );
        let dot = to_dot(&graph);
        assert!(
            dot.contains(r#"label="say \"hi\"""#),
            "unescaped quote:\n{dot}"
        );
    }
}
