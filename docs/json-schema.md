# JSON Schema Reference

Machine-readable schemas live in [`schemas/graph.schema.json`](../schemas/graph.schema.json) (`plekton.graph/v1`) and [`schemas/hetero_graph.schema.json`](../schemas/hetero_graph.schema.json) (`plekton.hetero-graph/v1`), and are contract-tested by `crates/plekton-core/tests/schema.rs`.

## Config input (`generate_from_config`)

```json
{
  "model_type": "csbm",
  "seed": 42,
  "parameters": {
    "n_nodes": 1000,
    "n_communities": 5,
    "homophily": 0.8,
    "avg_degree": 10.0,
    "features_dim": 16,
    "mu": 1.0,
    "feature_dist": "gaussian",
    "theta_exponent": null,
    "feat_noise_ratio": 0.0,
    "ensure_connected": false,
    "m": null,
    "m0": null
  }
}
```

### `model_type`

| Value | Model |
|---|---|
| `"sbm_classique"` or `"sbm"` | Stochastic Block Model / DC-SBM |
| `"csbm"` | Contextual SBM (topology + node features) |
| `"ba"` | Barabási–Albert preferential attachment |

### `parameters`

| Field | Type | Applies to | Description |
|---|---|---|---|
| `n_nodes` | int | all | Total number of nodes |
| `n_communities` | int | all | Number of community blocks |
| `homophily` | float `[0,1]` | SBM/cSBM | Target fraction of intra-community edges |
| `avg_degree` | float | SBM/cSBM | Target average node degree |
| `features_dim` | int | cSBM | Feature vector dimensionality (default `16`) |
| `mu` | float | cSBM | Signal-to-noise ratio (centroid separation, default `1.0`) |
| `feature_dist` | string | cSBM | `"gaussian"`, `"uniform"`, or `"laplacian"` |
| `theta_exponent` | float\|null | DC-SBM/cSBM | Pareto exponent $\gamma > 2$ for degree heterogeneity |
| `b_matrix` | float[]\|null | SBM | Optional symmetric $K \times K$ mixing matrix (row-major) |
| `block_sizes` | int[]\|null | SBM | Optional block sizes summing to `n_nodes` |
| `theta` | float[]\|null | SBM | Optional supplied per-node degree weights (mutually exclusive with `theta_exponent`) |
| `feat_noise_ratio` | float `[0,1]` | cSBM | Fraction of nodes with noisy class labels for features |
| `ensure_connected` | bool | cSBM | Stitch disconnected components with random bridges |
| `m` | int\|null | BA | Edges added per new node (default `2`) |
| `m0` | int\|null | BA | Initial clique size (default `m`) |

---

## Graph output (`plekton.graph/v1`)

```json
{
  "metadata": {
    "schema": "plekton.graph/v1",
    "n_nodes": 1000,
    "n_edges": 4985,
    "homophily": 0.802,
    "avg_degree_actual": 9.970,
    "n_components": 1,
    "transitivity_global": 0.041,
    "clustering_mean_local": null,
    "density": 0.00998,
    "assortativity": -0.012,
    "metrics_skipped": [
      {
        "metric": "clustering_mean_local",
        "reason": "this producer computes global transitivity only"
      }
    ]
  },
  "nodes": [
    { "id": 0, "community": 0, "features": [0.5, -1.2, 0.3] }
  ],
  "edges": [
    { "source": 0, "target": 12 }
  ]
}
```

- Every edge satisfies `source < target` (canonical undirected pair, listed once).
- `features` is present on nodes only when using `csbm`.
- `original_id` (optional, integer or string) is written only by `plekton.nx.networkx_to_json`, and only when the NetworkX ids are not already exactly `0..n-1`: every node is then renumbered (integer ids in ascending order, then string ids) and keeps its NetworkX id there. It is on every node or on none; without it, `id` is the original id. The Rust reader ignores it.
- `transitivity_global` is `null` above 20 000 nodes or when no open triplet exists; `metrics_skipped` explains every `null` metric.
- `homophily` is always present but may be `null` when it is undefined: `plekton.nx.networkx_to_json` writes `null` for a graph with no edges or no labelled node (every `community` is `0`), the same rule as `plekton.metrics.homophily`, and records the reason in `metrics_skipped`.
- `assortativity` is `null` for edgeless or regular graphs where degree variance is zero.
