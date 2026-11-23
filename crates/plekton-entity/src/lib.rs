// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Records → an entity/relation graph you can navigate, with an account of what was lost.
//!
//! Raw event streams and tables — transaction logs, interaction records, relational exports —
//! show data row by row and hide the *shape*: which entities are hubs, what is connected to
//! what, which parts are islands. This crate turns any source of records into that shape.
//!
//! # The split that makes it general
//!
//! An [`Adapter`] maps one source's records into [`Entity`] and [`Relation`] values, and
//! does nothing else. The core ([`build`]) resolves and measures them, and knows nothing
//! about any source. `kind` is an opaque string on both sides — the core indexes and counts
//! kinds but never compares one to a literal, so a per-type branch cannot be written here
//! by accident. A test pins it: the same graph shape measured under `"user"`/`"item"` and
//! under `"widget"`/`"sprocket"` yields identical metrics.
//!
//! The payoff is that a new source costs a field-mapping and nothing more. If an adapter
//! ever needs graph logic, the core is not general enough and *that* is the bug.
//!
//! # Honesty travels with the data
//!
//! [`build`] returns [`Diagnostics`] alongside the graph: records skipped and why, exact
//! counts per kind, duplicate ids, and relations naming entities that were never declared.
//! Those last are **dropped**, because inventing the missing endpoint would fabricate
//! structure no record supports — and dropping them silently would leave a graph that looks
//! complete and is not. A metric without its coverage is not evidence.
//!
//! ```
//! use plekton_entity::{build, Entity, Extraction, Relation, DEFAULT_TOP_HUBS};
//!
//! let extraction = Extraction {
//!     entities: vec![
//!         Entity::new("app:user:alice", "user"),
//!         Entity::new("app:item:book1", "item"),
//!     ],
//!     relations: vec![Relation::new("app:user:alice", "app:item:book1", "bought")],
//!     records_parsed: 1,
//!     ..Extraction::default()
//! };
//!
//! let graph = build(extraction, DEFAULT_TOP_HUBS);
//! assert_eq!(graph.metrics.n_entities, 2);
//! assert_eq!(graph.metrics.connected_components, 1);
//! // A user bought an item: the relation crosses two kinds, so homophily is 0.
//! assert_eq!(graph.metrics.type_homophily, Some(0.0));
//! ```

#![warn(missing_docs)]

pub mod adapters;
pub mod dot;
pub mod graph;
pub mod hetero;
pub mod join;
pub mod live;
pub mod model;
pub mod namespace;
pub mod source;

pub use dot::to_dot;
pub use graph::{
    build, build_bounded, Diagnostics, Hub, Metrics, SkippedMetric, StructuredGraph,
    DEFAULT_TOP_HUBS,
};
pub use model::{
    Adapter, AdapterError, Attrs, Entity, Extraction, Limits, Relation, DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_RECORDS,
};
