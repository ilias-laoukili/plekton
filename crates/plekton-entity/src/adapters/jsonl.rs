// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Newline-delimited JSON objects (JSONL / NDJSON) → entities and relations,
//! driven by an explicit field mapping ([`TabularConfig`]).
//!
//! Reuses [`crate::adapters::lines::for_each_line`] for bounded reading and
//! lossy-UTF-8 resilience so malformed or truncated lines are skipped and
//! counted rather than aborting a batch.

use std::io::BufRead;

use serde_json::Value;

use crate::{
    adapters::{
        lines::for_each_line,
        tabular::{TabularAdapter, TabularConfig},
    },
    model::{Adapter, AdapterError, Entity, Extraction, Limits, Relation},
};

/// Reads newline-delimited JSON objects under a caller-supplied field mapping.
#[derive(Debug, Clone)]
pub struct JsonlAdapter {
    config: TabularConfig,
}

impl JsonlAdapter {
    /// Construct a JSONL adapter from the same field-mapping config as [`TabularAdapter`].
    ///
    /// # Errors
    /// Returns [`AdapterError::Unusable`] if the mapping is invalid (missing domain,
    /// no entity declarations, or empty column names).
    pub fn new(config: TabularConfig) -> Result<Self, AdapterError> {
        // Reuse TabularAdapter's validation rules so CSV and JSONL enforce identical invariants.
        let _ = TabularAdapter::new(config.clone())?;
        Ok(Self { config })
    }

    fn kind_of(&self, field: &str) -> Option<&str> {
        self.config
            .entities
            .iter()
            .find(|spec| spec.column == field)
            .map(|spec| spec.kind.as_str())
    }

    fn qualify(&self, kind: &str, raw: &str) -> String {
        crate::namespace::qualify(&self.config.domain, kind, raw).unwrap_or_else(|_| raw.to_owned())
    }

    fn field_as_string(obj: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
        let val = obj.get(key)?;
        let s = match val {
            Value::String(text) => text.trim().to_owned(),
            Value::Number(num) => num.to_string(),
            Value::Bool(flag) => flag.to_string(),
            Value::Null | Value::Array(_) | Value::Object(_) => return None,
        };
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    fn map_line(&self, line: &str, out: &mut Extraction) {
        let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(line) else {
            out.skip("invalid_json_object");
            return;
        };

        let mut emitted = false;
        for spec in &self.config.entities {
            let Some(raw) = Self::field_as_string(&obj, &spec.column) else {
                continue;
            };
            let mut entity = Entity::new(self.qualify(&spec.kind, &raw), spec.kind.clone());
            for attr in &spec.attrs {
                if let Some(val) = obj.get(attr) {
                    if !val.is_null() {
                        entity.attrs.insert(attr.clone(), val.clone());
                    }
                }
            }
            out.entities.push(entity);
            emitted = true;
        }

        for spec in &self.config.relations {
            let (Some(src_raw), Some(dst_raw)) = (
                Self::field_as_string(&obj, &spec.from),
                Self::field_as_string(&obj, &spec.to),
            ) else {
                continue;
            };
            let src_id = self
                .kind_of(&spec.from)
                .map_or_else(|| src_raw.clone(), |k| self.qualify(k, &src_raw));
            let dst_id = self
                .kind_of(&spec.to)
                .map_or_else(|| dst_raw.clone(), |k| self.qualify(k, &dst_raw));
            let mut rel = Relation::new(src_id, dst_id, spec.kind.clone());
            for attr in &spec.attrs {
                if let Some(val) = obj.get(attr) {
                    if !val.is_null() {
                        rel.attrs.insert(attr.clone(), val.clone());
                    }
                }
            }
            out.relations.push(rel);
            emitted = true;
        }

        if emitted {
            out.records_parsed += 1;
        } else {
            out.skip("no_mapped_fields");
        }
    }
}

impl Adapter for JsonlAdapter {
    fn name(&self) -> &'static str {
        "jsonl"
    }

    fn extract_from(
        &self,
        input: &mut dyn BufRead,
        limits: Limits,
    ) -> Result<Extraction, AdapterError> {
        for_each_line(input, limits, |line, out| self.map_line(line, out))
    }
}
