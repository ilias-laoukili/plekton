# plekton

[![CI](https://github.com/ilias-laoukili/plekton/actions/workflows/ci.yml/badge.svg)](https://github.com/ilias-laoukili/plekton/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

*From Ancient Greek **πλεκτόν** (plektón — woven, meshed structure).*

High-performance synthetic graph generation, blockmodel estimation, structural metrics, spectral sparsification, and leakage-free splitters — written in **pure Rust** with **zero-copy Python bindings** (`PyO3` + `maturin`, `numpy`-only base package) for **network science, simulation, and graph ML**.

## Highlights

- **Multi-attribute mixing graphs (`mixing_graph`)** — generate graphs meeting simultaneous homophily / heterophily targets (or full mixing matrices) across multiple node attributes via Iterative Proportional Fitting (IPF), with group confinement (households, teams, organizations), degree caps / matchings, triangle-closing clustering targets, tie strengths, and directed heavy-tail popularity layers.
- **Fast blockmodel & scale-free generators (`SBM`, $O(N + E)$ `DC-SBM`, `cSBM`, `BA`)** — geometric skip sampling for sparse SBM/cSBM, exact dyadic weight-bucketed $O(N + E)$ DC-SBM (no $O(N^2)$ collapse under heavy Pareto tails), correlated node features via Cholesky decomposition, and zero-copy NumPy output usable directly with NumPy, NetworkX, or PyTorch Geometric.
- **Fit → Generate → Measure closed loop** — estimate block sizes, $K \times K$ mixing matrices, and per-node degree corrections $\theta$ (`plekton.fit`), detect overlapping communities via SLPA, regenerate from fits (`generate_sbm_from_fit`), and compare structural metrics (`plekton.metrics`) without fabricating `0.0` for undefined quantities.
- **Spectral & backbone sparsification (`plekton.sparsify`)** — Spielman–Srivastava approximate effective resistance (Johnson–Lindenstrauss + dependency-free conjugate gradient, parallelized with Rayon), shortest-path metric backbone, degree-aware & weighted selection, and connectivity-preserving edge selection.
- **Leakage-free ML splitters (`plekton.splits`)** — group-disjoint stratified bin-packing, chronological splits with interval embargo, and strict degeneracy diagnostics.
- **Domain-agnostic entity graphs (`crates/plekton-entity`)** — turn CSV/TSV and JSONL event records into heterogeneous graphs with coverage diagnostics, temporal co-occurrence joins, tumbling windows, Graphviz DOT, and `plekton.hetero-graph/v1` export.
- **Deterministic & lean** — fixed seeds produce identical graphs across platforms; `import plekton` requires only `numpy` (`pyg`, `networkx`, and `bench` are optional extras).

## Quick Start (Python)

### 1. Multi-attribute mixing graphs

```python
import plekton as gg

# Marriages: one partner each, inside the household, heterophilous on sex, homophilous on age.
marriage = gg.mixing_graph(
    len(people),
    attributes={"sex": sexes, "age": age_bands},
    groups=households,
    degrees=[1 if married else 0 for married in status],
    max_degree=1,
    mixing={"sex": {"within": 0.0}, "age": {"within": 0.8}},
    seed=42,
)
edges = marriage["edges"]          # np.ndarray[int64] of shape (E, 2)
report = marriage["mixing"]["sex"] # {"target": 0.0, "within": 0.0, "random": ..., "r": -1.0}
```

### 2. Attributed & heavy-tailed graphs (cSBM, $O(N + E)$ DC-SBM)

```python
import plekton as gg

# Contextual SBM: zero-copy NumPy arrays (usable directly with NumPy, NetworkX, or PyG)
edge_index, x, labels, effective_h = gg.generate_csbm(
    n_nodes=10_000,
    n_communities=5,
    homophily=0.75,
    avg_degree=10.0,
    feat_dim=64,
    mu=1.5,
    theta_exponent=2.5,  # exact O(N + E) degree-corrected heavy tail
    seed=42,
)
```

### 3. Fit → Regenerate → Sparsify → Measure

```python
import numpy as np
import plekton as gg

edge_index, labels, _ = gg.generate_sbm(500, 4, p_in=0.15, p_out=0.01, seed=0)
edges = np.ascontiguousarray(edge_index[:, edge_index[0] < edge_index[1]].T)

# Estimate blockmodel & regenerate
fitted = gg.fit.fit_blockmodel(edges, n_nodes=500, labels=labels)
ei2, labels2, _ = gg.generate_sbm_from_fit(
    500, fitted["block_sizes"], fitted["b_matrix"], fitted["theta"], seed=1
)

# Spectral sparsification (Spielman–Srivastava approximate effective resistance)
resistance = gg.sparsify.approx_effective_resistance(edges, n_nodes=500, seed=42)
keep = gg.sparsify.select_mask(edges, 500, resistance, "sampled", retention_ratio=0.5)
sparse_edges = edges[keep]

# Structural metrics (None rather than fabricated 0.0 when undefined)
print(gg.metrics.homophily(sparse_edges, labels))
print(gg.metrics.degree_summary(500, sparse_edges))
```

### 4. Overlapping communities (SLPA) & leakage-free splits

```python
import plekton as gg

communities = gg.overlapping_communities(n_nodes, edges.tolist(), seed=0, rounds=100, threshold=0.2)
train_mask, val_mask, test_mask = gg.splits.grouped_split(group_keys, fracs=(0.64, 0.16, 0.20), seed=0)
```

## Benchmarks & Trade-offs

Measured via `python benchmarks/bench.py --quick` (isolated subprocess per task for exact `ru_maxrss`, 3 timed repetitions after warm-up, on Apple A18 Pro · Darwin arm64 · Python 3.11.17):

| Task ($K=5$, $\bar{d}=10$) | Engine | Nodes | Edges | Median (ms) | Peak RAM (MiB) |
|---|---|---:|---:|---:|---:|
| SBM (sparse) | `plekton` | 10,000 | 50,095 | **2.52** | **37.7** |
| SBM (sparse) | `networkx` | 10,000 | 49,658 | 504.53 | 65.2 |
| DC-SBM ($\gamma=2.5$) | `plekton` | 10,000 | 48,295 | **4.61** | **37.7** |
| cSBM ($F=32$) | `plekton` | 10,000 | 50,095 | **3.45** | **39.7** |
| BA ($m=5$) | `plekton` | 10,000 | 49,985 | **1.21** | **39.0** |
| `mixing_graph` (2 attrs, $d=6$) | `plekton` | 2,000 | 6,000 | **18.23** | **38.9** |

**Explicit trade-offs:**
- SBM, DC-SBM, cSBM, and BA produce **simple undirected graphs** (no self-loops or parallel edges); directed graphs are supported by `mixing_graph(..., directed=True)` and `plekton-entity`.
- Global transitivity in JSON metadata is skipped above 20,000 nodes (`None` + `SkippedMetric`), rather than slowing down generation or fabricating `0.0`.
- Generators return zero-copy NumPy arrays directly; JSON serialization is opt-in via `plekton.to_json(edge_index, labels, x=None, path=None)`.

## Workspace Structure

| Path | Description |
|---|---|
| `crates/plekton-core` (`plekton`) | Pure-Rust generation engine: SBM, $O(N+E)$ DC-SBM, cSBM, BA, `mixing_graph`, JSON output |
| `crates/plekton-metrics` | Structural metrics & edge similarities over plain `&[(usize, usize)]` edge lists |
| `crates/plekton-fit` | Method-of-moments DC-SBM estimator, seeded label propagation, SLPA overlapping communities |
| `crates/plekton-sparsify` | Spectral approx-ER (Spielman–Srivastava), metric backbone, threshold, degree-aware & sampled sparsification |
| `crates/plekton-entity` | Generic CSV/JSONL → heterogeneous entity/relation graph builder, temporal joins, tumbling windows, DOT & PyG export |
| `crates/plekton-python` | PyO3 (`abi3-py311`) zero-copy NumPy bindings (`plekton._plekton`) |
| `python/plekton` | Python package (`mixing`, `fit`, `metrics`, `sparsify`, `splits`, `distributions`, `presets`, `sweep`, optional `pyg`/`hetero`/`nx`) |
| `schemas/` | Versioned JSON Schemas (`plekton.graph/v1`, `plekton.hetero-graph/v1`) |
| `docs/` | Model documentation and JSON contract reference |
| `benchmarks/` | Reproducible throughput benchmark (`benchmarks/bench.py`) |

## Build and Test

### Rust

```bash
git config core.hooksPath .githooks   # once per clone: activates local pre-commit checks
cargo build
cargo test
cargo clippy --workspace --exclude plekton-python --all-targets -- -D warnings
```

### Python

`plekton-python` is a `cdylib` built by `maturin` (so it is excluded from Cargo's `default-members`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install maturin pytest numpy ruff mypy
maturin develop --release
pytest -v
```

Optional extras (`torch` / `torch_geometric` / `networkx`) can be installed with `pip install -e '.[all]'`; tests requiring absent extras skip cleanly.

## Documentation

- [Generative models & scaling](docs/models.md)
- [JSON Schema reference](docs/json-schema.md)
- [AGENTS.md](AGENTS.md) — architecture & engineering invariants for contributors and coding agents

## License

Apache-2.0 — see [LICENSE](LICENSE).
