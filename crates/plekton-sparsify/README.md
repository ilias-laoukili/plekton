# plekton-sparsify

Principled edge-sparsification strategies for undirected graphs in pure Rust (parallelized with Rayon):

- **Effective-resistance sampling (`approx_er`, `sparsify_resistance_sampled`)** — approximate effective resistance via the Spielman–Srivastava Johnson–Lindenstrauss projection and a dependency-free conjugate-gradient Laplacian solver, then weighted sampling without replacement. The kept edges are not reweighted, so there is no spectral guarantee.
- **Metric backbone (`backbone`, `sparsify_metric_backbone`)** — keeps only shortest-path-preserving edges.
- **Score-driven selection (`Strategy::Threshold`, `Strategy::DegreeAware`, `Strategy::Sampled`)** — paired with `plekton-metrics` similarity scores (`Jaccard`, `AdamicAdar`, `DegreeProduct`, `FeatureCosine`) and a `Guard`: `Guard::Isolates` gives every node that had an edge its highest-scoring one back, so no node is left stranded for GNN message passing. It does not keep the graph connected.
