# plekton (`crates/plekton-core`)

Pure-Rust synthetic graph generation engine:

- **SBM** — $O(E)$ geometric skip sampling.
- **DC-SBM** — exact $O(N + E)$ degree-corrected SBM via dyadic weight bucketing (guaranteed acceptance rate $> 25\%$ even under heavy Pareto tails).
- **cSBM** — contextual SBM with correlated Gaussian/Uniform/Laplacian node features (Cholesky covariance), label noise, triangle closing, and optional component stitching.
- **Barabási–Albert (BA)** — preferential attachment scale-free backbone.
- **Multi-attribute mixing graphs (`generate_mixing_graph`)** — simultaneous per-attribute homophily/heterophily targets via IPF, group confinement, degree caps, triangle-closing clustering, tie strengths, and directed popularity layers.

Zero Python or NumPy dependencies. Python bindings live in [`../plekton-python`](../plekton-python).

## Quick Example

```rust
use plekton::generate_sbm_core;

let graph = generate_sbm_core(1_000, 4, 0.08, 0.005, Some(2.5), Some(42))?;
println!("{} nodes, {} edges, h = {:.3}", graph.n_nodes, graph.canonical_edges.len(), graph.effective_h);
```

See [docs/models.md](../../docs/models.md) and [docs/json-schema.md](../../docs/json-schema.md) for full parameter and JSON contract details.
