# plekton-sparsify

Principled edge-sparsification strategies for undirected graphs in pure Rust (parallelized with Rayon):

- **Spectral sparsification (`approx_er`, `sparsify_spectral`)** — approximate effective resistance via Spielman–Srivastava Johnson–Lindenstrauss projection and a dependency-free conjugate-gradient Laplacian solver.
- **Metric backbone (`backbone`, `sparsify_metric_backbone`)** — keeps only shortest-path-preserving edges.
- **Score-driven selection (`Strategy::Threshold`, `Strategy::DegreeAware`, `Strategy::Sampled`)** — paired with `plekton-metrics` similarity scores (`Jaccard`, `AdamicAdar`, `DegreeProduct`, `FeatureCosine`) and optional `preserve_connectivity` so no non-isolated node is left stranded for GNN message passing.
