# AGENTS.md

Guide for coding agents and contributors working in the `plekton` repository.

## Before Starting

- Check that the starting point is sound: `cargo test --workspace --exclude plekton-python` and `pytest -q` pass.
- Check whether the local pre-commit hook is active: `git config core.hooksPath` should report `.githooks` (activate once per clone with `git config core.hooksPath .githooks`).
- Inspect `git status` and `git diff` to preserve any manual edits made by the author; never overwrite them silently.

## Project Overview

`plekton` is a pure-Rust synthetic graph generation, estimation, metric, and sparsification toolkit with zero-copy Python bindings (`PyO3` + `maturin`).

### Workspace Layout & Dependency DAG

```text
crates/
  plekton-metrics   (leaf crate — structural metrics & edge similarities over `&[(usize, usize)]`)
  plekton-fit       (leaf crate — DC-SBM estimation & SLPA overlapping communities; `rand` only)
  plekton-sparsify  (depends on: plekton-metrics — approx-ER scores, metric backbone, selection)
  plekton-entity    (depends on: plekton-metrics — generic CSV/JSONL → heterogeneous entity graph)
  plekton-core      (lib `plekton`, depends on: plekton-metrics — SBM, O(E) DC-SBM, cSBM, BA, mixing_graph)
  plekton-python    (cdylib `_plekton`, depends on core/fit/metrics/sparsify — PyO3 abi3-py311)
python/
  plekton/          (lean Python package: numpy-only base + optional `pyg`, `networkx`, `bench` extras)
  tests/             (pytest suites; torch/pyg/networkx tests skip cleanly when extras are absent)
schemas/             (JSON Schemas validated by `crates/plekton-core/tests/schema.rs`)
docs/                (model reference `models.md` and JSON contract `json-schema.md`)
benchmarks/          (reproducible throughput benchmark `bench.py`)
assets/              (SVG logos light/dark for the README + social-preview source)
```

## Build, Test & Lint Commands

### Rust (pure-Rust crates; `plekton-python` is excluded from `default-members`)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude plekton-python --all-targets -- -D warnings
cargo test --workspace --exclude plekton-python
cargo test --doc --workspace --exclude plekton-python
cargo check -p plekton-python
```

### Python (requires a venv with `maturin`, `numpy`, `pytest`, `ruff`, `mypy`)

```bash
maturin develop --release
pytest -v
ruff check python benchmarks
mypy
```

## Architectural & Engineering Invariants

1. **No panics in library code**: `unwrap_used` and `expect_used` are `deny` workspace-wide (`Cargo.toml`). Propagate typed errors (`SynthGraphError`, `FitError`, `SparsifyError`, `AdapterError`). Only tests may relax this.
2. **Strict lints everywhere**:
   - Rust: `clippy::pedantic` and `clippy::nursery` are enabled; `#![warn(missing_docs)]` is active on public crates.
   - Python: `ruff` uses `select = ["ALL"]` and `mypy` runs in `strict = true` mode.
3. **Domain-agnostic scope**: Every crate and Python module must stay universal. Do not add domain-specific schemas, dataset loaders, or security/vertical adapters to the core workspace.
4. **Zero-copy NumPy contract (`crates/plekton-python`)**:
   - Return owned C-contiguous `ndarray::Array1` / `Array2` via `IntoPyArray::into_pyarray(py)`. Never call `to_pyarray_bound` (it clones).
   - Run all non-trivial Rust work inside `py.allow_threads(|| ...)` so the GIL is released.
   - Keep `import plekton` strictly `numpy`-only; defer `torch`, `torch_geometric`, and `networkx` imports to their optional submodules.
5. **Cross-platform seed determinism**:
   - Core generators (`sbm`, `csbm`, `ba`, `mixing`) and the `plekton-fit` detectors (label propagation, SLPA) must stay deterministic for a given `seed`.
   - Seeded randomness uses `rand_chacha::ChaCha8Rng` only (`plekton-core`'s `sampling::seeded_rng`, or `ChaCha8Rng::seed_from_u64`). Never `SmallRng`, `StdRng` or `thread_rng`: rand does not promise them to be reproducible across platforms or versions.
   - `crates/plekton-core/tests/determinism_golden.rs` pins exact edges, labels, and features; `crates/plekton-fit/tests/determinism_golden.rs` pins label propagation and SLPA. Never alter RNG consumption order or sampling loops accidentally; intentional algorithmic revisions must explicitly re-freeze golden constants and document why.
   - `plekton-sparsify` pins its two seeded outputs in unit tests: `approx_er::tests::scores_are_unchanged_by_the_sampling_stream` (approximate effective-resistance scores, bit for bit) and `ops::sampled::tests::selection_golden` (a `Strategy::Sampled` selection). The projection reads stream 0 of `ChaCha8Rng::seed_from_u64(seed)` and the sampling stream 1, so one seed given to both never shares draws. Same rule as above: never change these values accidentally; re-freeze them only deliberately, saying why.
   - Not covered by the goldens: 32-bit targets (rand 0.8 samples `usize` ranges at pointer width) and minor-version upgrades of `rand` / `rand_distr`, which may change values and then require a deliberate re-freeze.
6. **Honest metrics & pre-allocation budgets**:
   - Undefined or threshold-skipped metrics return `Option::None` (paired with `SkippedMetric`), never a fabricated `0.0`.
   - Quadratic structures (`plekton-fit`'s `K×K` matrix, `plekton-entity`'s windowed joins) must project memory before allocation and refuse above budget.
7. **Schema stability**:
   - `schemas/graph.schema.json` (`plekton.graph/v1`) and `schemas/hetero_graph.schema.json` (`plekton.hetero-graph/v1`) are enforced by `crates/plekton-core/tests/schema.rs`. Any schema change must update both the schema file and its producer/tests in the same commit.
8. **Python bindings for `plekton-python`**:
   - Never link `plekton-python` with plain `cargo test` on macOS (cdylib expects Python symbols at runtime via `maturin`). Use `cargo check -p plekton-python` for Rust compilation checks and `maturin develop && pytest` for runtime testing.

## Agent & Repository Discipline

- **No agent attribution**: An agent is never a contributor. Never add `Co-Authored-By` trailers or attribute authorship/contributions to an agent in commit messages, pull requests, documentation, or source files.
- **Git & file hygiene**:
  - Do not commit without being explicitly asked.
  - Do not create new Markdown (`.md`) documentation files without being asked.
  - Never rewrite published/recorded git history (`amend`, `rebase`, `force push`) and never bypass hooks (`--no-verify`).
- **Deterministic generated artifacts**: Any generated export (JSON, DOT, deterministic CSV) must be byte-identical when its inputs have not changed — never embed build timestamps or random IDs in deterministic outputs.
- **Scientific references**: Cite published journal/proceedings versions with verified DOIs (or official proceedings URLs), not preprints or unverified metadata.
- **Before handing back**: Run the relevant Rust and Python checks and state clearly what was verified and what was **not** verified (for example, `torch`/`pyg` tests skipped when optional extras are absent).
