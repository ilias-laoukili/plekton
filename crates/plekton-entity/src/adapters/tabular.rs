// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! CSV and other delimited tables → entities and relations, driven by a column mapping.
//!
//! The other adapters know their source's field names. This one cannot: a CSV has whatever
//! columns someone chose. So the *caller* supplies the mapping — which column holds an
//! entity id, what kind it is, which pair of columns forms a relation — and this file
//! applies it.
//!
//! That makes it generic by construction rather than by discipline: there is no list of
//! known schemas here to grow, and no format-sniffing to guess wrong. A table this adapter
//! has never seen works if the caller can describe it, and fails loudly if they cannot.
//!
//! Uses the `csv` crate rather than a naive line splitter so quoted fields with
//! embedded commas and newlines never corrupt data silently.

use std::io::BufRead;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::{Adapter, AdapterError, Entity, Extraction, Limits, Relation};

/// One column that becomes an entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityColumn {
    /// Column holding the entity's id.
    pub column: String,
    /// Kind to give it. Opaque here as everywhere else.
    pub kind: String,
    /// Columns to carry onto the entity as attributes.
    #[serde(default)]
    pub attrs: Vec<String>,
}

/// A relation between two columns of the same row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationColumn {
    /// Column holding the source entity's id.
    pub from: String,
    /// Column holding the destination entity's id.
    pub to: String,
    /// Kind to give the relation.
    pub kind: String,
    /// Columns to carry onto the relation as attributes.
    #[serde(default)]
    pub attrs: Vec<String>,
}

/// How to read a table: which columns are entities, and which pairs are relations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TabularConfig {
    /// Which domain this table's ids belong to.
    ///
    /// **Required, and deliberately not defaulted.** Every other adapter knows its own domain
    /// because it knows what it reads; a table is whatever the caller points it at, so only the
    /// caller can say. A default here would put every CSV anyone ever loads into one namespace,
    /// which is the merge [`crate::namespace`] exists to stop — and it would do it silently,
    /// which is worse than the bare ids this replaces. Follow-ups entry 37.
    pub domain: String,
    /// Columns that become entities.
    #[serde(default)]
    pub entities: Vec<EntityColumn>,
    /// Column pairs that become relations.
    #[serde(default)]
    pub relations: Vec<RelationColumn>,
    /// Field separator. Defaults to `,`; set `"\t"` for TSV.
    #[serde(default)]
    pub delimiter: Option<String>,
}

/// Reads delimited tables under a caller-supplied column mapping.
#[derive(Debug, Clone)]
pub struct TabularAdapter {
    config: TabularConfig,
}

impl TabularAdapter {
    /// The kind declared for a column, when one is.
    fn kind_of(&self, column: &str) -> Option<&str> {
        self.config
            .entities
            .iter()
            .find(|spec| spec.column == column)
            .map(|spec| spec.kind.as_str())
    }

    /// `<domain>:<kind>:<value>`, with the domain the caller declared.
    ///
    /// Not [`crate::namespace::qualify_owned`]: that takes `&'static str`, and here both halves
    /// come from a config file at run time. The separator is stripped from them rather than
    /// refused, because a caller who wrote `kind: "ip:host"` meant one kind and refusing the
    /// whole run over a punctuation choice helps nobody — while letting it through would make
    /// the id parse into a different domain than the one declared.
    fn qualify(&self, kind: &str, value: &str) -> String {
        let domain = self.config.domain.replace(crate::namespace::SEPARATOR, "_");
        let kind = kind.replace(crate::namespace::SEPARATOR, "_");
        format!(
            "{domain}{}{kind}{}{value}",
            crate::namespace::SEPARATOR,
            crate::namespace::SEPARATOR
        )
    }

    /// Build an adapter for `config`.
    ///
    /// # Errors
    /// [`AdapterError::Unusable`] if the mapping could not produce anything — no entity
    /// columns at all, or a delimiter that is not a single byte. A mapping that yields
    /// nothing is a mistake worth reporting up front rather than an empty graph later.
    pub fn new(config: TabularConfig) -> Result<Self, AdapterError> {
        if config.domain.trim().is_empty() {
            return Err(AdapterError::Unusable(
                "the mapping declares no `domain`, and a table has none of its own. Every id \
                 this adapter mints is `<domain>:<kind>:<value>`, and the domain is the one part \
                 only the caller knows — add `\"domain\": \"…\"` naming what these rows are \
                 about (`ip`, `dns`, `windows`, `rf`, or your own). Defaulting it would put \
                 every table anyone loads into one namespace, silently."
                    .to_owned(),
            ));
        }
        if config.entities.is_empty() {
            return Err(AdapterError::Unusable(
                "the mapping declares no entity columns, so no graph could be built; \
                 give at least one { column, kind }"
                    .to_owned(),
            ));
        }
        if let Some(delimiter) = &config.delimiter {
            if delimiter.len() != 1 {
                return Err(AdapterError::Unusable(format!(
                    "delimiter must be exactly one byte, got {delimiter:?}"
                )));
            }
        }
        Ok(Self { config })
    }

    /// The configured delimiter, defaulting to a comma.
    fn delimiter(&self) -> u8 {
        self.config
            .delimiter
            .as_ref()
            .and_then(|d| d.as_bytes().first().copied())
            .unwrap_or(b',')
    }
}

/// Drops a leading UTF-8 BOM, passing everything else through untouched.
///
/// Excel, `PowerShell` and several SIEM exporters prefix their CSV with `EF BB BF`. The `csv`
/// crate treats it as part of the first header name, so a mapping naming that column fails
/// to match and the caller is told their column does not exist — about the least helpful
/// true statement available.
struct BomStripper<'a> {
    inner: &'a mut dyn BufRead,
    checked: bool,
}

impl<'a> BomStripper<'a> {
    fn new(inner: &'a mut dyn BufRead) -> Self {
        Self {
            inner,
            checked: false,
        }
    }
}

impl std::io::Read for BomStripper<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.checked {
            self.checked = true;
            let head = self.inner.fill_buf()?;
            if head.starts_with(&[0xEF, 0xBB, 0xBF]) {
                self.inner.consume(3);
            }
        }
        self.inner.read(buf)
    }
}

/// Value of `column` in this row, or `None` when absent or empty.
fn cell<'a>(
    row: &'a csv::StringRecord,
    headers: &csv::StringRecord,
    column: &str,
) -> Option<&'a str> {
    let position = headers.iter().position(|header| header == column)?;
    row.get(position).filter(|value| !value.is_empty())
}

/// Copy the named columns into an attribute map, skipping absent ones.
fn gather(
    row: &csv::StringRecord,
    headers: &csv::StringRecord,
    columns: &[String],
) -> crate::Attrs {
    let mut attrs = crate::Attrs::new();
    for column in columns {
        if let Some(value) = cell(row, headers, column) {
            attrs.insert(column.clone(), Value::String(value.to_owned()));
        }
    }
    attrs
}

impl Adapter for TabularAdapter {
    fn name(&self) -> &'static str {
        "tabular"
    }

    fn extract_from(
        &self,
        input: &mut dyn BufRead,
        limits: Limits,
    ) -> Result<Extraction, AdapterError> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(self.delimiter())
            // Ragged rows are normal in exported data and must not abort the read.
            .flexible(true)
            // Excel and PowerShell both write a UTF-8 BOM. Without this the first header
            // becomes "\u{feff}user", every mapping that names `user` fails to match, and
            // the error blames the caller for a file that looks perfectly correct.
            .trim(csv::Trim::Headers)
            .from_reader(BomStripper::new(input));

        let headers = reader
            .headers()
            .map_err(|e| AdapterError::Unusable(format!("could not read the header row: {e}")))?
            .clone();

        // A mapping naming a column the table does not have would otherwise produce an
        // empty graph with no explanation. Refusing up front says which column is wrong.
        let mut missing: Vec<&str> = Vec::new();
        for entity in &self.config.entities {
            if !headers.iter().any(|header| header == entity.column) {
                missing.push(&entity.column);
            }
        }
        for relation in &self.config.relations {
            for column in [&relation.from, &relation.to] {
                if !headers.iter().any(|header| header == column) {
                    missing.push(column);
                }
            }
        }
        if !missing.is_empty() {
            let available: Vec<&str> = headers.iter().collect();
            return Err(AdapterError::Unusable(format!(
                "the mapping names column(s) {missing:?} that the table does not have; \
                 available: {available:?}"
            )));
        }

        let mut out = Extraction::default();
        let mut seen = 0usize;
        for record in reader.records() {
            seen += 1;
            if seen > limits.max_records {
                // Refused rather than truncated: metrics on a prefix of a table describe a
                // graph nobody has. See `Limits`.
                return Err(AdapterError::TooLarge {
                    seen,
                    limit: limits.max_records,
                    unit: "rows",
                    knob: "max_records",
                });
            }
            let Ok(row) = record else {
                out.skip("unreadable row");
                continue;
            };

            let mut used = false;
            for spec in &self.config.entities {
                let Some(id) = cell(&row, &headers, &spec.column) else {
                    continue;
                };
                out.entities.push(Entity {
                    id: self.qualify(&spec.kind, id),
                    kind: spec.kind.clone(),
                    attrs: gather(&row, &headers, &spec.attrs),
                });
                used = true;
            }

            for spec in &self.config.relations {
                let (Some(from), Some(to)) = (
                    cell(&row, &headers, &spec.from),
                    cell(&row, &headers, &spec.to),
                ) else {
                    continue;
                };
                // An endpoint's id has to be minted exactly as its entity was, so the KIND comes
                // from the entity column of the same name rather than from the relation. A
                // relation naming a column no entity column declares was already a dangling
                // reference; it is refused by name now instead of producing one.
                let (Some(from_kind), Some(to_kind)) =
                    (self.kind_of(&spec.from), self.kind_of(&spec.to))
                else {
                    return Err(AdapterError::Unusable(format!(
                        "the relation `{}` joins `{}` to `{}`, and at least one of those is not \
                         declared as an entity column. An endpoint whose kind nothing states \
                         cannot be namespaced the same way its entity is, so the edge would name \
                         an entity this adapter never emits.",
                        spec.kind, spec.from, spec.to
                    )));
                };
                out.relations.push(Relation {
                    src: self.qualify(from_kind, from),
                    dst: self.qualify(to_kind, to),
                    kind: spec.kind.clone(),
                    attrs: gather(&row, &headers, &spec.attrs),
                });
                used = true;
            }

            if used {
                out.records_parsed += 1;
            } else {
                // The row parsed fine; it just had nothing in the mapped columns. Counted,
                // because a table where most rows are empty in the mapped columns is a
                // mapping problem the caller should see.
                out.skip("row had no value in any mapped column");
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TabularConfig {
        TabularConfig {
            domain: "ip".to_owned(),
            entities: vec![
                EntityColumn {
                    column: "user".to_owned(),
                    kind: "user".to_owned(),
                    attrs: vec!["dept".to_owned()],
                },
                EntityColumn {
                    column: "server".to_owned(),
                    kind: "server".to_owned(),
                    attrs: Vec::new(),
                },
            ],
            relations: vec![RelationColumn {
                from: "user".to_owned(),
                to: "server".to_owned(),
                kind: "logged_into".to_owned(),
                attrs: vec!["result".to_owned()],
            }],
            delimiter: None,
        }
    }

    fn extract(config: TabularConfig, input: &str) -> Extraction {
        let Ok(adapter) = TabularAdapter::new(config) else {
            panic!("the mapping should be usable");
        };
        let Ok(out) = adapter.extract(input.as_bytes()) else {
            panic!("the table should be readable");
        };
        out
    }

    #[test]
    fn a_mapped_table_becomes_entities_and_relations() {
        let out = extract(
            config(),
            "user,server,dept,result\nalice,db01,finance,success\nbob,db01,ops,failure\n",
        );

        assert_eq!(out.records_parsed, 2);
        // Two users and two (duplicate) servers; the core de-duplicates later.
        assert_eq!(out.entities.len(), 4);
        assert_eq!(out.relations.len(), 2);
        assert_eq!(out.relations[0].kind, "logged_into");
        assert_eq!(out.relations[0].attrs["result"], "success");
        assert_eq!(out.entities[0].attrs["dept"], "finance");
    }

    /// The point of a configurable adapter: a table it has never seen, described by the
    /// caller, works — with kinds this code has no knowledge of.
    #[test]
    fn a_completely_unfamiliar_table_works_from_its_mapping_alone() {
        let config = TabularConfig {
            domain: "ip".to_owned(),
            entities: vec![
                EntityColumn {
                    column: "part_no".to_owned(),
                    kind: "part".to_owned(),
                    attrs: Vec::new(),
                },
                EntityColumn {
                    column: "assembly".to_owned(),
                    kind: "assembly".to_owned(),
                    attrs: Vec::new(),
                },
            ],
            relations: vec![RelationColumn {
                from: "part_no".to_owned(),
                to: "assembly".to_owned(),
                kind: "fits_into".to_owned(),
                attrs: Vec::new(),
            }],
            delimiter: None,
        };
        let out = extract(config, "part_no,assembly\nP-1,A-9\nP-2,A-9\n");

        let graph = crate::build(out, crate::DEFAULT_TOP_HUBS);
        assert_eq!(graph.metrics.n_entities, 3);
        assert_eq!(graph.metrics.connected_components, 1);
        // A-9 is the hub: two parts fit into it.
        assert_eq!(graph.metrics.top_hubs[0].id, "ip:assembly:A-9");
    }

    #[test]
    fn a_tab_separated_table_is_read_when_the_delimiter_says_so() {
        let mut config = config();
        config.delimiter = Some("\t".to_owned());
        let out = extract(config, "user\tserver\tdept\tresult\nalice\tdb01\tfin\tok\n");
        assert_eq!(out.records_parsed, 1);
        assert_eq!(out.relations.len(), 1);
    }

    /// Quoted fields containing the delimiter are exactly where a hand-rolled splitter
    /// corrupts data without saying so — which is why this uses a real CSV reader.
    #[test]
    fn a_quoted_field_containing_a_comma_stays_one_value() {
        let out = extract(
            config(),
            "user,server,dept,result\n\"Doe, Alice\",db01,finance,success\n",
        );
        assert_eq!(out.entities[0].id, "ip:user:Doe, Alice");
    }

    // ── What it refuses, and how clearly ──────────────────────────────────────

    #[test]
    fn a_mapping_with_no_entity_columns_is_refused_up_front() {
        let config = TabularConfig {
            domain: "ip".to_owned(),
            ..TabularConfig::default()
        };
        let Err(error) = TabularAdapter::new(config) else {
            panic!("a mapping that can build nothing is a mistake");
        };
        assert!(error.to_string().contains("no entity columns"));
    }

    #[test]
    fn a_mapping_naming_a_missing_column_says_which_and_what_was_available() {
        let Ok(adapter) = TabularAdapter::new(config()) else {
            panic!("the mapping is well-formed");
        };
        let Err(error) = adapter.extract(b"account,host\nalice,db01\n") else {
            panic!("a mapping that matches no column must be refused");
        };
        let message = error.to_string();
        // Both halves matter: which column is wrong, and what it could have been.
        assert!(message.contains("user"), "{message}");
        assert!(message.contains("account"), "{message}");
    }

    #[test]
    fn a_row_with_nothing_in_the_mapped_columns_is_counted_not_ignored() {
        let out = extract(
            config(),
            "user,server,dept,result\nalice,db01,fin,ok\n,,,\n",
        );
        assert_eq!(out.records_parsed, 1);
        assert_eq!(
            out.records_skipped["row had no value in any mapped column"],
            1
        );
    }

    #[test]
    fn a_multi_byte_delimiter_is_refused() {
        let mut config = config();
        config.delimiter = Some("||".to_owned());
        let Err(error) = TabularAdapter::new(config) else {
            panic!("a two-byte delimiter is not a delimiter");
        };
        assert!(error.to_string().contains("one byte"));
    }

    // ── What real exported tables actually look like ──────────────────────────

    /// Excel and `PowerShell` prefix CSV with a UTF-8 BOM. Without handling it the first
    /// column name carries an invisible character and every mapping naming it "does not
    /// exist" — a true statement that helps nobody.
    #[test]
    fn a_utf8_bom_does_not_hide_the_first_column() {
        let with_bom = "\u{feff}user,server,dept,result\nalice,db01,fin,ok\n";
        let out = extract(config(), with_bom);
        assert_eq!(out.records_parsed, 1);
        assert_eq!(out.entities[0].id, "ip:user:alice");
    }

    /// Windows-exported files use CRLF; a stray \r would otherwise ride along inside the
    /// last value of every row and split one entity into two.
    #[test]
    fn crlf_line_endings_do_not_contaminate_the_last_column() {
        let out = extract(
            config(),
            "user,server,dept,result\r\nalice,db01,fin,ok\r\nbob,db01,ops,no\r\n",
        );
        assert_eq!(out.records_parsed, 2);
        // `result` is last, so a surviving \r would show up here.
        assert_eq!(out.relations[0].attrs["result"], "ok");
    }

    /// A quoted field may contain the line ending itself.
    #[test]
    fn a_quoted_field_may_span_lines() {
        let out = extract(
            config(),
            "user,server,dept,result\n\"multi\nline\",db01,fin,ok\n",
        );
        assert_eq!(out.records_parsed, 1);
        assert_eq!(out.entities[0].id, "ip:user:multi\nline");
    }

    /// Rows with more or fewer columns than the header are normal in exported data.
    #[test]
    fn ragged_rows_are_read_rather_than_aborting_the_table() {
        let out = extract(
            config(),
            "user,server,dept,result\nalice,db01\nbob,db02,ops,no,extra\n",
        );
        assert_eq!(
            out.records_parsed, 2,
            "both rows still name a user and a server"
        );
    }

    /// Bytes that are not UTF-8 cost their row, never the table.
    #[test]
    fn invalid_utf8_costs_one_row_not_the_file() {
        let mut bytes = b"user,server,dept,result\n".to_vec();
        bytes.extend(b"alice,db01,fin,ok\n");
        bytes.extend(vec![0xff, 0xfe]);
        bytes.extend(b",db02,ops,no\n");
        bytes.extend(b"carol,db03,mkt,ok\n");

        let Ok(adapter) = TabularAdapter::new(config()) else {
            panic!("the mapping is usable");
        };
        let Ok(out) = adapter.extract(&bytes) else {
            panic!("bad bytes in one row must not fail the table");
        };
        assert!(out.records_parsed >= 2, "the good rows survive");
    }

    /// A header-only file is not an error; it simply has no rows.
    #[test]
    fn a_header_without_rows_yields_an_empty_extraction() {
        let out = extract(config(), "user,server,dept,result\n");
        assert_eq!(out.records_parsed, 0);
        assert_eq!(out.entities.len(), 0);
    }

    #[test]
    fn extraction_is_deterministic() {
        let table = "user,server,dept,result\nalice,db01,fin,ok\nbob,db02,ops,no\n";
        let Ok(first) = serde_json::to_string(&extract(config(), table)) else {
            panic!("serialisable");
        };
        let Ok(second) = serde_json::to_string(&extract(config(), table)) else {
            panic!("serialisable");
        };
        assert_eq!(first, second);
    }
}
