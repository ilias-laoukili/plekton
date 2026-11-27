<div align="center">
  <p>
    <img src="assets/logo.svg#gh-light-mode-only" alt="plekton logo" width="96">
    <img src="assets/logo-dark.svg#gh-dark-mode-only" alt="plekton logo" width="96">
  </p>

  <h1>plekton</h1>

  <p>Fast synthetic graph generation and sparsification in Rust, with Python bindings.</p>

  <p>
    <a href="https://github.com/ilias-laoukili/plekton/actions/workflows/ci.yml"><img src="https://github.com/ilias-laoukili/plekton/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  </p>
</div>

*From Greek πλεκτόν ("woven").* Like crochet — where every stitch depends on local topology and resists naïve mechanization — `plekton` automates the synthesis and sparsification of structured graphs.

## Quick start

Requires Python ≥ 3.11 and a Rust toolchain: the extension is compiled during installation.

```bash
pip install git+https://github.com/ilias-laoukili/plekton
```

```python
import plekton
from plekton import sparsify

graph = plekton.generate_sbm(1_000, 4, p_in=0.05, p_out=0.002, seed=42)
scores = sparsify.approx_effective_resistance(graph, seed=42)
keep = sparsify.select_mask(graph, scores, "sampled", retention_ratio=0.5, seed=42)
sparse = graph.edge_subgraph(keep)
print(f"{graph.n_edges} edges, homophily {graph.metadata['homophily']:.2f}, {sparse.n_edges} kept")
```

Every generator returns a `plekton.Graph`: `n_nodes` nodes and a read-only `(E, 2)` int64 `edges` array that lists each undirected edge once, as `u < v`, plus optional `labels`, `x` and per-edge `edge_attrs`. `graph.edge_index` gives the PyG `(2, 2E)` layout. Every metric, fit and sparsify function takes the `Graph` first (or an `(E, 2)` array with `n_nodes=`), and refuses a `(2, E)` array, an edge listed in both directions, an id outside `[0, n_nodes)` or a non-integer dtype instead of silently miscounting.

## Benchmarks

Measured with `python benchmarks/bench.py --quick` on an Apple A18 Pro (Darwin arm64, Python 3.11.17). Each task runs in a fresh subprocess; times are medians of 3 timed runs after one warm-up call. SBM-family rows use K = 5 blocks, mean degree 10 and homophily 0.75.

| Task | Engine | Nodes | Edges | Median (ms) | Peak RAM (MiB) |
|---|---|---:|---:|---:|---:|
| SBM | `plekton` | 10,000 | 50,095 | 2.52 | 37.7 |
| SBM | `networkx` | 10,000 | 49,658 | 504.53 | 65.2 |
| DC-SBM (γ = 2.5) | `plekton` | 10,000 | 48,295 | 4.61 | 37.7 |
| cSBM (F = 32) | `plekton` | 10,000 | 50,095 | 3.45 | 39.7 |
| BA (m = 5) | `plekton` | 10,000 | 49,985 | 1.21 | 39.0 |
| `mixing_graph` (2 attributes, degree 6) | `plekton` | 2,000 | 6,000 | 18.23 | 38.9 |

- Peak RAM is the peak resident set size of the whole process (`ru_maxrss`), Python interpreter and imports included.
- The two engines draw from different random streams, so edge counts differ slightly for identical parameters.
- NetworkX (`nx.stochastic_block_model`) is the only external engine in the script; the other rows time `plekton` alone.

To reproduce (release build required; options in [benchmarks/README.md](benchmarks/README.md)):

```bash
maturin develop --release
python benchmarks/bench.py --quick
```

## Features

### Generation

- **SBM** — geometric skip sampling inside each block pair: O(E) on sparse graphs instead of testing all N² pairs. Arbitrary block sizes and K × K matrices through `generate_sbm_from_fit`.
- **DC-SBM** — exact degree-corrected sampling in O(N + E + K² log² N) expected time. Dyadic weight buckets keep the acceptance rate of every proposal above 25 %, so heavy Pareto tails never collapse to O(N²). Node weights θ come from Pareto(γ − 1) with γ > 2, or from a fit.
- **cSBM** — topology plus correlated F-dimensional node features: class centroids scaled by μ; Gaussian, uniform or Laplacian noise; Cholesky-factorised covariance; feature label noise; triangle closing; optional component stitching.
- **Barabási–Albert** — preferential attachment from an initial `m0`-clique, `m` edges per new node.
- **`mixing_graph`** — simultaneous homophily or heterophily targets (or full category-pair mixing matrices) on several categorical attributes, solved by iterative proportional fitting (IPF). Group confinement, degree caps and exact matchings, clustering targets reached by degree- and mixing-preserving rewiring, tie strengths, and directed graphs with log-normal popularity.

### Estimation and metrics

- **`plekton.fit`** — method-of-moments DC-SBM estimation (block sizes, K × K mixing matrix, per-node θ), seeded label propagation, and SLPA overlapping communities.
- **`plekton.metrics`** — edge homophily, attribute mixing (Newman's r), mean local clustering, global transitivity, degree assortativity, components, density, degree summary, and two-sample KS distance between degree distributions.

### Sparsification

- **Spectral** — Spielman–Srivastava approximate effective resistance: Johnson–Lindenstrauss projection and a dependency-free conjugate-gradient Laplacian solver, parallelised with Rayon.
- **Metric backbone** — keeps only shortest-path-preserving edges; the retained fraction follows from the graph, not from a ratio.
- **Score-based selection** — threshold, degree-aware (per-node minimum) or weighted sampling over Jaccard, Adamic–Adar, degree-product or feature-cosine scores. A guard (`guard="isolates"`, the default) gives every node that had an edge its highest-scoring one back, so the selection leaves no node isolated; it does not keep the graph connected.

### Machine-learning utilities

- **`plekton.splits`** — leakage-free splits: group-disjoint stratified bin-packing, chronological splits with interval embargo, and strict degeneracy diagnostics.
- **Interoperability** — PyTorch Geometric (`plekton.pyg`, `plekton.hetero`), NetworkX (`plekton.nx`), canonical `plekton.graph/v1` JSON (`plekton.to_json`), benchmark presets (`plekton.presets`) and parameter sweeps (`plekton.sweep`).
- **Entity graphs** (`crates/plekton-entity`, Rust) — CSV/TSV and JSONL records turned into heterogeneous entity graphs with coverage diagnostics, temporal co-occurrence joins, tumbling windows, Graphviz DOT and `plekton.hetero-graph/v1` export.

## Examples

### Multi-attribute mixing

```python
import plekton

# 600 people in households of 6. Marry inside the household:
# heterophilous on sex, homophilous on age band.
n = 600
sex = ["female", "male"] * (n // 2)
age = [("18-34", "35-54", "55+")[(i // 2) % 3] for i in range(n)]
households = [f"h{i // 6}" for i in range(n)]

marriage = plekton.mixing_graph(
    n,
    attributes={"sex": sex, "age": age},
    groups=households,
    degrees=[1] * n,
    max_degree=1,
    mixing={"sex": {"within": 0.0}, "age": {"within": 0.8}},
    seed=42,
)
marriage.edges                      # read-only int64 array of shape (300, 2)
marriage.metadata["mixing"]["sex"]  # {'target': 0.0, 'within': 0.0, 'random': 0.5, 'r': -1.0}
marriage.metadata["mixing"]["age"]  # within ≈ 0.81 for a target of 0.8 (random: 0.33)
```

### Attributed, heavy-tailed graphs (cSBM with degree correction)

```python
import plekton

graph = plekton.generate_csbm(
    n_nodes=10_000,
    n_communities=5,
    homophily=0.75,
    avg_degree=10.0,
    feat_dim=64,
    mu=1.5,
    theta_exponent=2.5,  # exact degree-corrected heavy tail
    seed=42,
)
# graph.edges: int64 (E, 2) · graph.x: float32 (10000, 64) · graph.labels: int64 (10000,)
# graph.edge_index: int64 (2, 2E), every edge in both directions, for PyG
```

### Fit → regenerate → sparsify → measure

```python
import plekton

graph = plekton.generate_sbm(500, 4, p_in=0.15, p_out=0.01, seed=0)

# Estimate the blockmodel on the graph's own labels, then regenerate from the fit.
fitted = plekton.fit.fit_blockmodel(graph)
regenerated = plekton.generate_sbm_from_fit(
    500, fitted["block_sizes"], fitted["b_matrix"], theta=fitted["theta"], seed=1
)

# Keep half of the edges, sampled by approximate effective resistance.
resistance = plekton.sparsify.approx_effective_resistance(graph, seed=42)
keep = plekton.sparsify.select_mask(graph, resistance, "sampled", retention_ratio=0.5, seed=42)
sparse = graph.edge_subgraph(keep)  # same nodes and labels, half the edges

# Metrics return None, never a fabricated 0.0, when a quantity is undefined.
print(plekton.metrics.homophily(sparse))
print(plekton.metrics.degree_summary(sparse))
```

### Overlapping communities and leakage-free splits

```python
import plekton

# Two 8-cliques joined through node 16, which SLPA places in both communities.
circle_a = [(a, b) for a in range(8) for b in range(a + 1, 8)]
circle_b = [(a, b) for a in range(8, 16) for b in range(a + 1, 16)]
bridge = [(v, 16) for v in (0, 1, 2, 3, 8, 9, 10, 11)]
graph = plekton.Graph(circle_a + circle_b + bridge, n_nodes=17)
communities = plekton.fit.overlapping_communities(graph, seed=1)

# Group-disjoint split: no household straddles train, validation and test.
households = [f"h{i // 4}" for i in range(400)]
train, val, test = plekton.splits.grouped_split(households, fracs=(0.64, 0.16, 0.20), seed=0)
```

Like every graph function, `plekton.fit.overlapping_communities` takes a `plekton.Graph`, or an `(E, 2)` edge array with `n_nodes=`.

## Guarantees and trade-offs

- **Deterministic** — a fixed `seed` reproduces the same graph or detection: generators and detectors draw from `ChaCha8Rng` (`rand_chacha`). Golden tests pin the exact edges, labels and features of SBM, DC-SBM, BA and cSBM (`crates/plekton-core/tests/determinism_golden.rs`) and the output of label propagation and SLPA (`crates/plekton-fit/tests/determinism_golden.rs`); CI runs them on Linux, macOS and Windows. What they do not cover is listed in [docs/models.md](docs/models.md#determinism).
- **Lean** — `import plekton` needs only NumPy; `torch`, `torch_geometric` and `networkx` come with the optional extras `pyg`, `networkx` and `all`.
- **Zero-copy** — generators hand the canonical edges, labels and features over from Rust without copying, so a generated `Graph` holds 16 bytes per edge; the PyG `edge_index` (32 more) is built only when asked for. JSON serialisation is opt-in through `plekton.to_json(graph, path=None)`.
- **No fabricated values** — undefined or skipped metrics are `None`, never `0.0`, and JSON metadata records the reason in `metrics_skipped` (global transitivity, for instance, is skipped above 20,000 nodes).
- **Simple graphs** — SBM, DC-SBM, cSBM and BA produce simple undirected graphs (no self-loops, no parallel edges); directed graphs come from `mixing_graph(..., directed=True)` and `plekton-entity`. `plekton.Graph` enforces it: a self-loop, a repeated edge or an undirected edge listed in both directions is refused.
- **One seed convention** — every randomised function defaults to `seed=None`, which draws a seed; the generators record the seed that ran in `graph.metadata["seed"]`, so an unseeded graph can still be regenerated.

## Workspace

| Path | Contents |
|---|---|
| `crates/plekton-core` (`plekton`) | Pure-Rust generation engine: SBM, O(N + E) DC-SBM, cSBM, BA, `mixing_graph`, JSON output |
| `crates/plekton-metrics` | Structural metrics and edge similarities over plain `&[(usize, usize)]` edge lists |
| `crates/plekton-fit` | Method-of-moments DC-SBM estimator, seeded label propagation, SLPA overlapping communities |
| `crates/plekton-sparsify` | Spectral approx-ER (Spielman–Srivastava), metric backbone, threshold, degree-aware and sampled selection |
| `crates/plekton-entity` | Generic CSV/JSONL → heterogeneous entity/relation graphs, temporal joins, tumbling windows, DOT and PyG export |
| `crates/plekton-python` | PyO3 (`abi3-py311`) zero-copy NumPy bindings (`plekton._plekton`) |
| `python/plekton` | Python package (`mixing`, `fit`, `metrics`, `sparsify`, `splits`, `distributions`, `presets`, `sweep`; optional `pyg`, `hetero`, `nx`) |
| `schemas/` | Versioned JSON Schemas (`plekton.graph/v1`, `plekton.hetero-graph/v1`) |
| `docs/` | Model documentation and JSON contract reference |
| `benchmarks/` | Reproducible throughput benchmark (`benchmarks/bench.py`) |

## Development

### Rust

```bash
git config core.hooksPath .githooks   # once per clone: activates local pre-commit checks
cargo build
cargo test
cargo clippy --workspace --exclude plekton-python --all-targets -- -D warnings
```

### Python

`plekton-python` is a `cdylib` built by `maturin`, so it is excluded from Cargo's `default-members`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install maturin pytest numpy ruff mypy
maturin develop --release
pytest -v
```

Optional extras (`torch`, `torch_geometric`, `networkx`) install with `pip install -e '.[all]'`; tests that need an absent extra are skipped.

## Documentation

- [Generative models and scaling](docs/models.md) — definitions, complexity and references
- [JSON Schema reference](docs/json-schema.md) — `generate_from_config` input and `plekton.graph/v1` output
- [Benchmarks](benchmarks/README.md) — methodology and options
- [AGENTS.md](AGENTS.md) — architecture and engineering invariants for contributors and coding agents

## License

Apache-2.0 — see [LICENSE](LICENSE).
