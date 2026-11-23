// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Every adapter emits namespaced ids (`<domain>:<kind>:<value>`), preventing
//! silent cross-domain identifier collisions.

use plekton_entity::{
    adapters::{
        jsonl::JsonlAdapter,
        tabular::{TabularAdapter, TabularConfig},
    },
    model::{Adapter as _, Extraction},
    namespace::is_qualified,
};

type Fallible<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn assert_all_qualified(extraction: &Extraction, adapter: &str) {
    for entity in &extraction.entities {
        assert!(
            is_qualified(&entity.id),
            "{adapter} emitted `{}`, which is not `<domain>:<kind>:<value>`",
            entity.id,
        );
    }
    for relation in &extraction.relations {
        for id in [&relation.src, &relation.dst] {
            assert!(
                is_qualified(id),
                "{adapter} drew an edge to `{id}`, which is not qualified",
            );
        }
    }
    assert!(
        !extraction.entities.is_empty(),
        "{adapter} produced nothing, so this guard checked nothing"
    );
}

#[test]
fn tabular_emits_only_qualified_ids() -> Fallible {
    let config: TabularConfig = serde_json::from_value(serde_json::json!({
        "domain": "shop",
        "entities": [
            { "column": "user", "kind": "user" },
            { "column": "item", "kind": "item" }
        ],
        "relations": [
            { "from": "user", "to": "item", "kind": "bought" }
        ]
    }))?;
    let adapter = TabularAdapter::new(config)?;
    let out = adapter.extract(b"user,item\nalice,book-1\n")?;
    assert_all_qualified(&out, "tabular");
    Ok(())
}

#[test]
fn jsonl_emits_only_qualified_ids() -> Fallible {
    let config: TabularConfig = serde_json::from_value(serde_json::json!({
        "domain": "shop",
        "entities": [
            { "column": "user", "kind": "user" },
            { "column": "item", "kind": "item" }
        ],
        "relations": [
            { "from": "user", "to": "item", "kind": "bought" }
        ]
    }))?;
    let adapter = JsonlAdapter::new(config)?;
    let out = adapter.extract(b"{\"user\":\"alice\",\"item\":\"book:vol:1\"}\n")?;
    assert_all_qualified(&out, "jsonl");
    assert!(
        out.entities.iter().any(|e| e.id == "shop:item:book:vol:1"),
        "colons inside values must survive intact"
    );
    Ok(())
}

#[test]
fn tabular_and_jsonl_meet_on_the_same_qualified_id() -> Fallible {
    let config: TabularConfig = serde_json::from_value(serde_json::json!({
        "domain": "org",
        "entities": [
            { "column": "person", "kind": "person" },
            { "column": "team", "kind": "team" }
        ],
        "relations": [
            { "from": "person", "to": "team", "kind": "member_of" }
        ]
    }))?;
    let csv_out = TabularAdapter::new(config.clone())?.extract(b"person,team\nalice,core\n")?;
    let jsonl_out =
        JsonlAdapter::new(config)?.extract(b"{\"person\":\"alice\",\"team\":\"core\"}\n")?;
    assert_eq!(csv_out.entities[0].id, jsonl_out.entities[0].id);
    assert_eq!(csv_out.entities[1].id, jsonl_out.entities[1].id);
    Ok(())
}

#[test]
fn a_table_without_a_declared_domain_is_refused_and_told_why() -> Fallible {
    let config: TabularConfig = serde_json::from_value(serde_json::json!({
        "domain": "",
        "entities": [ { "column": "user", "kind": "user" } ]
    }))?;
    let Err(error) = TabularAdapter::new(config) else {
        panic!("a table with no domain cannot mint an id")
    };
    let text = error.to_string();
    assert!(text.contains("domain"), "{text}");
    assert!(
        text.contains("silently"),
        "the refusal has to say why a default would be worse than a refusal: {text}"
    );
    Ok(())
}
