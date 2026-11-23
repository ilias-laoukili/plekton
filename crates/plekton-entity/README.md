# plekton-entity

Domain-agnostic records → heterogeneous entity/relation graph builder with explicit coverage diagnostics.

- **Generic source adapters**: CSV/TSV (`adapters::tabular`) and JSONL/NDJSON (`adapters::jsonl`) driven by an explicit field-to-entity/relation mapping (`TabularConfig`) with required domain namespacing (`<domain>:<kind>:<value>`).
- **Bounded ingestion & honest diagnostics**: enforces `max_records` and `max_entities` budgets, counts skipped/malformed lines and unresolved references instead of silently dropping them.
- **Cross-domain temporal joins (`join`)**: windowed `co_occurs_with` and `same_identifier` joins with pre-allocation edge budgets (`MAX_EDGES`) and time-base compatibility checks.
- **Tumbling live windows (`live`)**, **Graphviz DOT export (`dot`)**, and **PyTorch Geometric `HeteroData` export (`hetero`, schema `plekton.hetero-graph/v1`)**.
