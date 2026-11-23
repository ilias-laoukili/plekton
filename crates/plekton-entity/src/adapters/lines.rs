// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Shared line-streaming for newline-delimited record adapters (such as JSONL).
//!
//! Counting, limit enforcement, and lossy-UTF-8 handling live here once so every
//! line-oriented adapter applies identical record semantics.

use std::io::BufRead;

use crate::model::{AdapterError, Extraction, Limits};

/// Feed each non-empty line to `map`, counting records and stopping at the limit.
///
/// Lines are decoded lossily rather than strictly: a single invalid byte sequence in the
/// middle of a large log should cost that one record, not the whole run. Whatever survives
/// decoding still has to parse as a record, so a mangled line is skipped and counted like
/// any other unreadable one.
///
/// # Errors
/// [`AdapterError::TooLarge`] once `limits.max_records` lines have been read.
pub fn for_each_line<F>(
    input: &mut dyn BufRead,
    limits: Limits,
    mut map: F,
) -> Result<Extraction, AdapterError>
where
    F: FnMut(&str, &mut Extraction),
{
    let mut out = Extraction::default();
    let mut seen = 0usize;
    let mut raw: Vec<u8> = Vec::new();

    loop {
        raw.clear();
        // read_until rather than lines(): it never fails on invalid UTF-8, so the decision
        // about mangled bytes stays here instead of aborting the read.
        match input.read_until(b'\n', &mut raw) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                return Err(AdapterError::Unusable(format!(
                    "could not read input: {error}"
                )))
            }
        }

        let line = String::from_utf8_lossy(&raw);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        seen += 1;
        if seen > limits.max_records {
            // Counted, not measured: the graph is discarded rather than reported on a
            // prefix. See `Limits` for why truncating would be worse than refusing.
            return Err(AdapterError::TooLarge {
                seen,
                limit: limits.max_records,
                unit: "records",
                knob: "max_records",
            });
        }

        map(line, &mut out);
    }

    Ok(out)
}
