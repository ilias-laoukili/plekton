// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! The generic vocabulary: entities, relations, and the adapter that produces them.
//!
//! Everything here is deliberately incurious about what the data *means*. [`Entity::kind`]
//! and [`Relation::kind`] are opaque strings — the core indexes and counts them but never
//! reads them, which is what makes a per-type branch impossible to write by accident
//! rather than merely discouraged.

use std::{collections::BTreeMap, io::BufRead};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Free-form attributes carried alongside an entity or relation.
///
/// `BTreeMap` rather than a hash map because the whole pipeline must be deterministic:
/// serialising attributes has to yield the same bytes on every run.
pub type Attrs = BTreeMap<String, Value>;

/// A node: a host, a process, a file, a domain, a user, a flow — the core does not know
/// and does not ask.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    /// Identity within this graph. Two entities sharing an id are the same thing.
    pub id: String,
    /// What the source called it. Opaque to the core.
    pub kind: String,
    /// Whatever the adapter chose to carry over.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: Attrs,
}

impl Entity {
    /// An entity with no attributes — the common case in an adapter.
    pub fn new(id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            attrs: Attrs::new(),
        }
    }

    /// Attach one attribute, chainable.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }
}

/// An edge: `spawned`, `connected_to`, `resolved_to`, `accessed` — again, opaque.
///
/// `src` and `dst` are entity ids rather than indices, so an adapter can emit a relation
/// before (or without) the entity it names. What happens to such a relation is the core's
/// business, and it is reported rather than silently resolved — see
/// [`crate::Diagnostics::unresolved_references`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity id.
    pub src: String,
    /// Destination entity id.
    pub dst: String,
    /// What the source called this connection. Opaque to the core.
    pub kind: String,
    /// Whatever the adapter chose to carry over.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: Attrs,
}

impl Relation {
    /// A relation with no attributes.
    pub fn new(src: impl Into<String>, dst: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            kind: kind.into(),
            attrs: Attrs::new(),
        }
    }

    /// Attach one attribute, chainable.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.attrs.insert(key.into(), value.into());
        self
    }
}

/// What an adapter produces: the records it understood, and an account of the rest.
///
/// The account is not optional. A parser that quietly drops what it cannot read produces a
/// graph that looks complete and is not, and no downstream metric can reveal the
/// difference — so the skipped records travel with the data that survived.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extraction {
    /// Entities the adapter recognised. Duplicate ids are expected and resolved by the core.
    pub entities: Vec<Entity>,
    /// Relations the adapter recognised.
    pub relations: Vec<Relation>,
    /// Records read and understood.
    pub records_parsed: usize,
    /// Records read and *not* understood, counted by reason.
    pub records_skipped: BTreeMap<String, usize>,
}

impl Extraction {
    /// Record that one input record could not be used, and why.
    ///
    /// The reason is a short stable category (`"malformed json"`, `"unknown log type"`),
    /// not a per-record message: an agent needs to know *how much* went unread and
    /// broadly why, and a thousand distinct strings would tell it neither.
    pub fn skip(&mut self, reason: impl Into<String>) {
        *self.records_skipped.entry(reason.into()).or_insert(0) += 1;
    }

    /// Total records the adapter refused.
    #[must_use]
    pub fn skipped_total(&self) -> usize {
        self.records_skipped.values().sum()
    }
}

/// How much input to accept before refusing.
///
/// # Why refuse rather than truncate
///
/// Processing a prefix and reporting how much was dropped sounds honest, and the count is —
/// but the *shape* is not. Components, hubs and homophily computed on the first 2% of a log
/// describe a graph nobody has: the hub that mattered may be entirely in the tail. A caller
/// acting on those numbers is worse off than one told plainly that the input is too large.
/// So above the cap nothing is measured, and the refusal says how to proceed.
///
/// Both limits are raisable by a caller who knows the memory cost they are accepting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// Records read from the input before refusing.
    pub max_records: usize,
    /// Entities in the built graph before refusing.
    ///
    /// Separate from `max_records` because they bound different things: one input record can
    /// yield several entities, and it is the graph — not the file — that has to fit in memory.
    pub max_entities: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_records: DEFAULT_MAX_RECORDS,
            max_entities: DEFAULT_MAX_ENTITIES,
        }
    }
}

/// Records accepted by default: 250 000.
///
/// Chosen from empirical measurement on dirty JSONL streams (3% malformed records,
/// hot domains producing real hubs), release build, Apple Silicon:
///
/// | records | entities | relations | time | peak RSS |
/// |---|---|---|---|---|
/// | 10 000 | 18 986 | 12 621 | 0.37 s | 30 MB |
/// | 100 000 | 142 568 | 127 068 | 0.35 s | 156 MB |
/// | **250 000** | **247 275** | **317 413** | **0.67 s** | **398 MB** |
/// | 500 000 | 309 005 | 634 660 | 1.01 s | 745 MB |
///
/// 250 000 sits where the peak is still a few hundred megabytes — comfortable on any laptop
/// or modest container — while 500 000 approaches a gigabyte, which is not a figure to
/// impose on a caller who never asked. Memory tracks *relations* more than entities: entity
/// growth flattens as ids repeat and de-duplicate, while relations keep accruing.
pub const DEFAULT_MAX_RECORDS: usize = 250_000;

/// Entities accepted by default: 250 000.
///
/// A second guard for input that does *not* de-duplicate — the table above shows 250 000
/// records yielding 247 275 entities, so a source where every record names fresh ids reaches
/// this bound at about the same time as the record bound, and one with heavy repetition is
/// governed by records instead. Both are raisable together.
pub const DEFAULT_MAX_ENTITIES: usize = 250_000;

/// Turn a source's bytes into entities and relations. **That is the entire contract.**
///
/// An adapter maps fields. It does not build graphs, compute metrics, deduplicate, resolve
/// references, or decide what is interesting — all of that is the core's, and identical for
/// every source. If an adapter ever needs one of those, the core is not generic enough and
/// that is the bug to fix.
pub trait Adapter {
    /// Name reported in provenance, e.g. `"csv"` or `"jsonl"`.
    fn name(&self) -> &'static str;

    /// Map records from `input` into entities and relations, reading incrementally.
    ///
    /// A reader rather than a slice so a 10 GB file is *refused by measurement* instead of
    /// by an allocation failure: records are counted as they stream past, and the limit trips
    /// long before the whole file is resident.
    ///
    /// # Errors
    /// Only for input that cannot be processed at all, including
    /// [`AdapterError::TooLarge`] when `limits.max_records` is exceeded. A single unreadable
    /// *record* is not an error — it is counted via [`Extraction::skip`], because one bad
    /// line in a million should narrow the answer, not refuse it.
    fn extract_from(
        &self,
        input: &mut dyn BufRead,
        limits: Limits,
    ) -> Result<Extraction, AdapterError>;

    /// Convenience wrapper for callers that already hold the bytes.
    ///
    /// # Errors
    /// As [`Adapter::extract_from`].
    fn extract(&self, input: &[u8]) -> Result<Extraction, AdapterError> {
        self.extract_from(&mut std::io::BufReader::new(input), Limits::default())
    }
}

/// Why an adapter could not process its input at all.
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    /// The input was not valid UTF-8.
    #[error("input is not valid UTF-8: {0}")]
    NotUtf8(#[from] std::str::Utf8Error),
    /// The caller named a source variant this adapter does not handle.
    #[error("unknown source type `{requested}`; this adapter handles: {supported}")]
    UnknownSourceType {
        /// What the caller asked for.
        requested: String,
        /// What is on offer, comma-separated.
        supported: String,
    },
    /// The input exceeds a limit, so nothing was measured.
    ///
    /// Carries what would be needed to proceed, because "too large" alone leaves a caller
    /// with no next step.
    #[error(
        "this input has at least {seen} {unit}, above the limit of {limit}. Structure would \
         otherwise describe only a prefix, and the shape of a prefix is not the shape of the \
         log. Pre-filter or split the input, or raise `{knob}` explicitly (memory grows with it)."
    )]
    TooLarge {
        /// How many were seen before stopping.
        seen: usize,
        /// The limit that was hit.
        limit: usize,
        /// What is being counted, e.g. `"records"`.
        unit: &'static str,
        /// The argument that raises this limit.
        knob: &'static str,
    },
    /// The input was structurally unusable (not merely one bad record).
    #[error("{0}")]
    Unusable(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_records_are_counted_by_reason() {
        let mut extraction = Extraction::default();
        extraction.skip("malformed json");
        extraction.skip("malformed json");
        extraction.skip("unknown log type");

        assert_eq!(extraction.skipped_total(), 3);
        assert_eq!(extraction.records_skipped["malformed json"], 2);
        assert_eq!(extraction.records_skipped["unknown log type"], 1);
    }

    #[test]
    fn attributes_serialise_in_a_stable_order() {
        // Determinism is a product claim, and it starts here: a hash map would reorder
        // these between runs and every downstream digest would move with it.
        let entity = Entity::new("h1", "host")
            .with("zeta", 1)
            .with("alpha", 2)
            .with("mu", 3);
        let Ok(json) = serde_json::to_string(&entity) else {
            panic!("an entity must serialise");
        };
        assert!(json.contains(r#""alpha":2,"mu":3,"zeta":1"#), "{json}");
    }

    #[test]
    fn an_entity_without_attributes_omits_the_field() {
        let Ok(json) = serde_json::to_string(&Entity::new("h1", "host")) else {
            panic!("an entity must serialise");
        };
        assert_eq!(json, r#"{"id":"h1","kind":"host"}"#);
    }
}
