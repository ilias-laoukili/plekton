// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Robustness tests: large, malformed, or hostile inputs must never panic,
//! never grow without bound, and never silently drop records.

use std::{fmt::Write as _, io::Cursor};

use plekton_entity::{
    adapters::{jsonl::JsonlAdapter, tabular::TabularConfig},
    build_bounded, Adapter, Limits, DEFAULT_TOP_HUBS,
};

fn make_jsonl_adapter() -> JsonlAdapter {
    let config: TabularConfig = serde_json::from_value(serde_json::json!({
        "domain": "app",
        "entities": [
            { "column": "src", "kind": "node" },
            { "column": "dst", "kind": "node" }
        ],
        "relations": [
            { "from": "src", "to": "dst", "kind": "links_to" }
        ]
    }))
    .unwrap_or_else(|e| panic!("valid test config: {e}"));
    JsonlAdapter::new(config).unwrap_or_else(|e| panic!("valid adapter: {e}"))
}

/// Dirty JSONL stream: mostly valid records mixed with malformed lines.
fn dirty_jsonl(records: usize) -> String {
    let mut out = String::with_capacity(records * 56);
    for i in 0..records {
        match i % 20 {
            3 => out.push_str("{not json at all\n"),
            7 => out.push_str("{\"unrelated\":1}\n"),
            11 => out.push_str("{\"src\":\"\",\"dst\":\"\"}\n"),
            15 => out.push('\n'),
            _ => {
                let src = i % 4096;
                let dst = (i * 7) % 512;
                let _ = writeln!(out, "{{\"src\":\"u{src}\",\"dst\":\"v{dst}\"}}");
            }
        }
    }
    out
}

#[test]
fn a_hundred_thousand_dirty_records_are_processed_without_losing_the_count() {
    let log = dirty_jsonl(100_000);
    let mut reader = Cursor::new(log.as_bytes());
    let limits = Limits {
        max_records: 200_000,
        max_entities: 200_000,
    };

    let Ok(extraction) = make_jsonl_adapter().extract_from(&mut reader, limits) else {
        panic!("100k records is within the raised limit");
    };

    let blank = 100_000 / 20;
    assert_eq!(
        extraction.records_parsed + extraction.skipped_total(),
        100_000 - blank,
        "a record must be either parsed or counted as skipped"
    );
    assert!(extraction.records_parsed > 70_000);

    let Ok(graph) = build_bounded(extraction, DEFAULT_TOP_HUBS, limits) else {
        panic!("the graph is within limits");
    };
    assert!(graph.metrics.n_entities > 1_000);
}

#[test]
fn an_oversized_input_is_refused_with_an_actionable_message() {
    let log = dirty_jsonl(5_000);
    let mut reader = Cursor::new(log.as_bytes());
    let limits = Limits {
        max_records: 100,
        max_entities: 100,
    };

    let Err(error) = make_jsonl_adapter().extract_from(&mut reader, limits) else {
        panic!("5000 records must not pass a 100-record limit");
    };
    let message = error.to_string();
    assert!(message.contains("101"), "{message}");
    assert!(message.contains("100"), "{message}");
    assert!(message.contains("max_records"), "{message}");
    assert!(
        message.contains("split") || message.contains("filter"),
        "the refusal must say what to do next: {message}"
    );
    assert!(message.contains("prefix"), "{message}");
}

#[test]
fn the_entity_limit_refuses_independently_of_the_record_limit() {
    let mut log = String::new();
    for i in 0..1_000 {
        let _ = writeln!(log, "{{\"src\":\"a{i}\",\"dst\":\"b{i}\"}}");
    }
    let limits = Limits {
        max_records: 10_000,
        max_entities: 100,
    };

    let Ok(extraction) =
        make_jsonl_adapter().extract_from(&mut Cursor::new(log.as_bytes()), limits)
    else {
        panic!("the record limit is not reached");
    };
    let Err(error) = build_bounded(extraction, DEFAULT_TOP_HUBS, limits) else {
        panic!("2000 entities must not pass a 100-entity limit");
    };
    assert!(error.to_string().contains("max_entities"), "{error}");
}

#[test]
fn adversarial_input_is_skipped_and_counted_rather_than_fatal() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("only newlines", b"\n\n\n\n".to_vec()),
        ("nul bytes", b"\0\0\0\n{\"src\":\"a\"}\n".to_vec()),
        ("invalid utf-8", vec![0xff, 0xfe, 0xfd, b'\n']),
        ("unterminated json", b"{\"src\":\"a\"".to_vec()),
        ("json array not object", b"[1,2,3]\n".to_vec()),
        ("json scalar", b"42\n\"text\"\ntrue\n".to_vec()),
        ("deeply nested", {
            let mut v = b"{\"src\":\"a\",\"x\":".to_vec();
            v.extend(std::iter::repeat_n(b'[', 2_000));
            v.extend(std::iter::repeat_n(b']', 2_000));
            v.extend(b"}\n");
            v
        }),
        ("one enormous line", {
            let mut v = b"{\"src\":\"".to_vec();
            v.extend(std::iter::repeat_n(b'A', 2_000_000));
            v.extend(b"\",\"dst\":\"b\"}\n");
            v
        }),
        (
            "no trailing newline",
            b"{\"src\":\"a\",\"dst\":\"b\"}".to_vec(),
        ),
    ];

    for (name, bytes) in cases {
        let mut reader = Cursor::new(bytes.as_slice());
        let Ok(extraction) = make_jsonl_adapter().extract_from(&mut reader, Limits::default())
        else {
            panic!("`{name}` should not fail the batch");
        };
        let total = extraction.records_parsed + extraction.skipped_total();
        assert!(total <= 3_000, "`{name}` invented records: {total}");
        let Ok(graph) = build_bounded(extraction, DEFAULT_TOP_HUBS, Limits::default()) else {
            panic!("`{name}` should build");
        };
        assert_eq!(
            graph.metrics.n_entities,
            graph.entities.len(),
            "`{name}` disagreed with itself about entity count"
        );
    }
}

#[test]
fn an_enormous_line_is_one_record_not_several() {
    let mut bytes = b"{\"src\":\"a\",\"dst\":\"".to_vec();
    bytes.extend(std::iter::repeat_n(b'B', 5_000_000));
    bytes.extend(b"\"}\n{\"src\":\"c\",\"dst\":\"d\"}\n");

    let Ok(extraction) =
        make_jsonl_adapter().extract_from(&mut Cursor::new(bytes.as_slice()), Limits::default())
    else {
        panic!("two records is within any limit");
    };
    assert_eq!(extraction.records_parsed + extraction.skipped_total(), 2);
}

#[test]
fn invalid_utf8_in_one_record_does_not_cost_the_others() {
    let mut bytes = b"{\"src\":\"a\",\"dst\":\"b\"}\n".to_vec();
    bytes.extend(vec![0xff, 0xfe, b'\n']);
    bytes.extend(b"{\"src\":\"c\",\"dst\":\"d\"}\n");

    let Ok(extraction) =
        make_jsonl_adapter().extract_from(&mut Cursor::new(bytes.as_slice()), Limits::default())
    else {
        panic!("bad bytes in one line must not fail the batch");
    };
    assert_eq!(extraction.records_parsed, 2, "both good records survive");
    assert_eq!(extraction.skipped_total(), 1, "the bad line is counted");
}

#[test]
fn a_large_run_is_reproducible() {
    let log = dirty_jsonl(20_000);
    let limits = Limits::default();

    let build_once = || {
        let Ok(extraction) =
            make_jsonl_adapter().extract_from(&mut Cursor::new(log.as_bytes()), limits)
        else {
            panic!("within limits");
        };
        let Ok(graph) = build_bounded(extraction, DEFAULT_TOP_HUBS, limits) else {
            panic!("within limits");
        };
        let Ok(json) = serde_json::to_string(&graph.metrics) else {
            panic!("metrics serialise");
        };
        json
    };

    assert_eq!(build_once(), build_once());
}
