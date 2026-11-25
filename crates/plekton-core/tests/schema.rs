// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Contract tests for the JSON Schemas (`schemas/*.schema.json`).
//!
//! A real generated graph must validate against `graph.schema.json`, and a real
//! heterogeneous export from `plekton-entity` must validate against
//! `hetero_graph.schema.json`.

use std::path::{Path, PathBuf};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn repo_root() -> PathBuf {
    // crates/plekton-core → repo root.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_json(rel: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let path = repo_root().join(rel);
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

fn validator_for(schema_rel: &str) -> Result<jsonschema::Validator, Box<dyn std::error::Error>> {
    let schema = load_json(schema_rel)?;
    jsonschema::validator_for(&schema).map_err(|e| format!("{schema_rel}: {e}").into())
}

/// Validate `instance`, flattening all violations into one readable error.
fn check(
    validator: &jsonschema::Validator,
    instance: &serde_json::Value,
    what: &str,
) -> TestResult {
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| format!("  - {error}"))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("{what} violates the schema:\n{}", errors.join("\n")).into())
    }
}

/// A real generated graph validates against `graph.schema.json` and carries `schema`.
#[test]
fn generated_graph_validates_against_schema() -> TestResult {
    let validator = validator_for("schemas/graph.schema.json")?;
    let result = plekton::generate_sbm_core(20, 4, 0.5, 0.1, None, Some(7))?;
    let graph = plekton::sbm_result_to_graph_output(&result);
    let value = serde_json::to_value(&graph)?;
    assert_eq!(
        value["metadata"]["schema"],
        serde_json::json!(plekton::GRAPH_SCHEMA_ID)
    );
    check(&validator, &value, "generated SBM graph")?;

    // A graph missing the format tag must fail.
    let mut bad = value;
    bad["metadata"]
        .as_object_mut()
        .ok_or("metadata is not an object")?
        .remove("schema");
    assert!(
        !validator.is_valid(&bad),
        "a graph JSON without the schema tag must fail validation"
    );
    Ok(())
}

/// An undefined homophily is written as `null` (with its reason), and that
/// document both validates and is read back by the Rust reader (`json_to_pyg`'s
/// path) as `None` — the producer `plekton.nx.networkx_to_json` relies on both.
#[test]
fn null_homophily_validates_and_reads_back_as_none() -> TestResult {
    let validator = validator_for("schemas/graph.schema.json")?;
    let result = plekton::generate_sbm_core(20, 4, 0.5, 0.1, None, Some(7))?;
    let mut value = serde_json::to_value(plekton::sbm_result_to_graph_output(&result))?;
    assert!(
        value["metadata"]["homophily"].is_number(),
        "a generated graph with edges must carry a measured homophily"
    );

    let metadata = value["metadata"]
        .as_object_mut()
        .ok_or("metadata is not an object")?;
    metadata.insert("homophily".into(), serde_json::Value::Null);
    metadata.insert(
        "metrics_skipped".into(),
        serde_json::json!([{
            "metric": "homophily",
            "reason": "undefined here: no node is labelled (every community is 0)"
        }]),
    );
    check(&validator, &value, "graph with null homophily")?;

    let read: plekton::GraphOutput = serde_json::from_value(value.clone())?;
    assert_eq!(read.metadata.homophily, None);

    // Nullable is not untyped: the field is still required, and still a number
    // whenever it is present.
    let mut bad = value.clone();
    bad["metadata"]["homophily"] = serde_json::json!("0.5");
    assert!(!validator.is_valid(&bad), "a string homophily must fail");
    let mut missing = value;
    missing["metadata"]
        .as_object_mut()
        .ok_or("metadata is not an object")?
        .remove("homophily");
    assert!(
        !validator.is_valid(&missing),
        "homophily must stay required"
    );
    Ok(())
}

/// A real heterogeneous export from `plekton-entity` validates against `hetero_graph.schema.json`.
#[test]
fn hetero_graph_validates_against_schema() -> TestResult {
    let validator = validator_for("schemas/hetero_graph.schema.json")?;
    let extraction = plekton_entity::Extraction {
        entities: vec![
            plekton_entity::Entity::new("app:user:alice", "user"),
            plekton_entity::Entity::new("app:item:book1", "item"),
        ],
        relations: vec![plekton_entity::Relation::new(
            "app:user:alice",
            "app:item:book1",
            "bought",
        )],
        records_parsed: 1,
        ..plekton_entity::Extraction::default()
    };
    let structured = plekton_entity::build(extraction, plekton_entity::DEFAULT_TOP_HUBS);
    let hetero = plekton_entity::hetero::to_hetero(&structured);
    let value = serde_json::to_value(&hetero)?;
    assert_eq!(
        value["schema"],
        serde_json::json!(plekton_entity::hetero::SCHEMA)
    );
    check(&validator, &value, "hetero graph")?;
    Ok(())
}
