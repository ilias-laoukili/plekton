// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Supported record sources and their shared adapter dispatch.

use crate::{Adapter as _, Extraction, Limits};

/// Why a source could not be read.
#[derive(Debug)]
pub enum SourceError {
    /// The caller named an unavailable source or omitted required configuration.
    Unsupported(String),
    /// The source data could not be processed.
    Unreadable(String),
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(message) | Self::Unreadable(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SourceError {}

/// Every source this build can turn into an entity graph.
pub const SOURCES: &[(&str, &str)] = &[
    (
        "csv",
        "A delimited table (CSV/TSV) with an explicit entity and relation column mapping",
    ),
    (
        "jsonl",
        "Newline-delimited JSON objects (JSONL/NDJSON) with an explicit field mapping",
    ),
];

/// Return the accepted source names as a stable comma-separated string.
#[must_use]
pub fn supported() -> String {
    SOURCES
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Extract entities and relations from one supported buffered source.
///
/// # Errors
/// Returns [`SourceError::Unsupported`] for an unknown or incompletely configured source and
/// [`SourceError::Unreadable`] when the selected adapter rejects the input.
pub fn extract(
    source: &str,
    config: Option<&serde_json::Value>,
    input: &mut dyn std::io::BufRead,
    limits: Limits,
) -> Result<Extraction, SourceError> {
    let parse_mapping = || -> Result<crate::adapters::tabular::TabularConfig, SourceError> {
        let Some(cfg) = config else {
            return Err(SourceError::Unsupported(format!(
                "source `{source}` needs a config mapping fields to entities and relations, \
                 for example {{\"domain\":\"app\",\"entities\":[{{\"column\":\"user\",\"kind\":\"user\"}}],\
                 \"relations\":[{{\"from\":\"user\",\"to\":\"item\",\"kind\":\"bought\"}}]}}"
            )));
        };
        serde_json::from_value(cfg.clone()).map_err(|error| {
            SourceError::Unsupported(format!("the config mapping is not usable: {error}"))
        })
    };

    let extraction = if source == "csv" || source == "tabular" {
        let parsed = parse_mapping()?;
        let adapter = crate::adapters::tabular::TabularAdapter::new(parsed)
            .map_err(|error| SourceError::Unsupported(error.to_string()))?;
        adapter.extract_from(input, limits)
    } else if source == "jsonl" || source == "ndjson" {
        let parsed = parse_mapping()?;
        let adapter = crate::adapters::jsonl::JsonlAdapter::new(parsed)
            .map_err(|error| SourceError::Unsupported(error.to_string()))?;
        adapter.extract_from(input, limits)
    } else {
        return Err(SourceError::Unsupported(format!(
            "unknown source `{source}`; this build reads: {}",
            supported()
        )));
    };
    extraction.map_err(|error| SourceError::Unreadable(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_source_lists_every_supported_adapter() {
        let Err(SourceError::Unsupported(message)) = extract(
            "smoke-signals",
            None,
            &mut std::io::BufReader::new(&b""[..]),
            Limits::default(),
        ) else {
            panic!("an unknown source must be refused");
        };
        for (name, _) in SOURCES {
            assert!(message.contains(name), "the refusal must list `{name}`");
        }
    }

    #[test]
    fn csv_requires_an_explicit_mapping() {
        let Err(SourceError::Unsupported(message)) = extract(
            "csv",
            None,
            &mut std::io::BufReader::new(&b""[..]),
            Limits::default(),
        ) else {
            panic!("CSV without a mapping must be refused");
        };
        assert!(message.contains("entities"));
    }

    #[test]
    fn jsonl_extracts_mapped_records() {
        let config = serde_json::json!({
            "domain": "shop",
            "entities": [
                {"column": "user", "kind": "user"},
                {"column": "item", "kind": "item"}
            ],
            "relations": [
                {"from": "user", "to": "item", "kind": "bought"}
            ]
        });
        let data = b"{\"user\":\"alice\",\"item\":\"book\"}\n";
        let Ok(out) = extract(
            "jsonl",
            Some(&config),
            &mut std::io::BufReader::new(&data[..]),
            Limits::default(),
        ) else {
            panic!("valid jsonl must extract");
        };
        assert_eq!(out.records_parsed, 1);
        assert_eq!(out.entities.len(), 2);
        assert_eq!(out.relations.len(), 1);
    }
}
