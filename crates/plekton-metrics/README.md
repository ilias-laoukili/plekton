# plekton-metrics

Domain-agnostic structural graph-metric primitives operating directly on plain `&[(usize, usize)]` edge lists — no graph object or wrapper type required.

## Modules

- `homophily` — edge homophily (`Option<f64>`)
- `mixing` — categorical attribute mixing (`within`, `random`, Newman's `r`)
- `clustering` — mean local clustering coefficient (`Option<f64>`)
- `transitivity` — global transitivity `3 * triangles / open_triplets` (`Option<f64>`)
- `assortativity` — Newman's degree assortativity `r` (`Option<f64>`)
- `components` — connected-component count
- `density` — undirected edge density
- `degree_distribution` — degree sequence, summary (`mean`, `max`, `p50`, `p90`, `p99`, `gini`), and two-sample KS distance
- `similarity` — per-edge Jaccard, Adamic–Adar, degree product, and feature cosine scores
- `skipped` — `SkippedMetric` explaining why any `Option<f64>` metric returned `None`

## Example

```rust
use plekton_metrics::{
    clustering::compute_clustering_coefficient,
    density::density,
    homophily::compute_homophily,
};

let edges = [(0, 1), (1, 2), (2, 0)];
let communities = [0, 0, 1];
assert!(compute_homophily(&edges, &communities).is_some());
assert_eq!(compute_clustering_coefficient(3, &edges), Some(1.0));
assert_eq!(density(3, edges.len()), 1.0);
```
