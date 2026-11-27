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
/// document both validates and is read back by the Rust reader (`json_to_graph`'s
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
            "reason": "undefined here: no node carries a community attribute"
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

/// Graphs whose metrics are undefined — no node, one node, no edge, no labels —
/// still validate against the schema as it is: an undefined `homophily` is the
/// `null` it already allows, `avg_degree_actual` and `density` are omitted (both
/// optional), and none of them is ever written as a `null` the schema refuses.
#[test]
fn degenerate_graphs_validate_against_the_unchanged_schema() -> TestResult {
    let validator = validator_for("schemas/graph.schema.json")?;
    let edgeless = plekton::generate_sbm_core(5, 2, 0.0, 0.0, None, Some(1))?;
    let cases = [
        (
            "no node",
            plekton::arrays_to_graph_output(0, &[], Some(&[]), None)?,
        ),
        (
            "one node",
            plekton::arrays_to_graph_output(1, &[], Some(&[0]), None)?,
        ),
        ("no edge", plekton::sbm_result_to_graph_output(&edgeless)),
        (
            "no labels",
            plekton::arrays_to_graph_output(3, &[(0, 1), (1, 2)], None, None)?,
        ),
    ];
    for (what, graph) in cases {
        let value = serde_json::to_value(&graph)?;
        check(&validator, &value, what)?;
        let metadata = value["metadata"]
            .as_object()
            .ok_or("metadata is not an object")?;
        assert!(
            metadata["homophily"].is_null(),
            "{what}: homophily should be null"
        );
        for field in ["avg_degree_actual", "density", "n_components"] {
            assert!(
                metadata.get(field).is_none_or(|v| !v.is_null()),
                "{what}: {field} must be a number or absent, never null"
            );
        }
        let skipped: Vec<&str> = metadata["metrics_skipped"]
            .as_array()
            .ok_or("metrics_skipped is missing")?
            .iter()
            .filter_map(|entry| entry["metric"].as_str())
            .collect();
        assert!(skipped.contains(&"homophily"), "{what}: {skipped:?}");
    }
    Ok(())
}

/// `original_id`, which `plekton.nx.networkx_to_json` writes on every node when it
/// renumbers them, is optional, holds an integer or a string, and is ignored by
/// the Rust reader.
#[test]
fn original_id_is_optional_and_an_integer_or_a_string() -> TestResult {
    let validator = validator_for("schemas/graph.schema.json")?;
    let result = plekton::generate_sbm_core(20, 4, 0.5, 0.1, None, Some(7))?;
    let plain = serde_json::to_value(plekton::sbm_result_to_graph_output(&result))?;
    check(&validator, &plain, "graph without original_id")?;

    // Integers and strings mixed, as for a NetworkX graph with both kinds of id.
    let mut renumbered = plain;
    let nodes = renumbered["nodes"]
        .as_array_mut()
        .ok_or("nodes is not an array")?;
    for (index, node) in nodes.iter_mut().enumerate() {
        let original = if index % 2 == 0 {
            serde_json::json!(index * 10)
        } else {
            serde_json::json!(format!("n{index}"))
        };
        node.as_object_mut()
            .ok_or("node is not an object")?
            .insert("original_id".into(), original);
    }
    check(&validator, &renumbered, "graph with original_id")?;

    let read: plekton::GraphOutput = serde_json::from_value(renumbered.clone())?;
    assert_eq!(read.nodes.len(), 20);

    for wrong in [
        serde_json::json!(1.5),
        serde_json::Value::Null,
        serde_json::json!(true),
        serde_json::json!([1]),
    ] {
        let mut bad = renumbered.clone();
        bad["nodes"][0]["original_id"] = wrong.clone();
        assert!(!validator.is_valid(&bad), "original_id {wrong} must fail");
    }
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
