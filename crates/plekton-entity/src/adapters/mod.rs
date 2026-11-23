// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Source adapters: one module per generic input format, each a field-mapping and nothing more.
//!
//! Every adapter here implements [`crate::Adapter`] and stops there. None of them
//! deduplicates, resolves references, builds a graph or computes a metric — that is the
//! core's work and it is identical for all of them.

pub mod jsonl;
pub mod lines;
pub mod tabular;
