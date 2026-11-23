# Generative Graph Models

`plekton` (`crates/plekton-core`) implements five families of synthetic graph models in pure Rust, paired with structural estimation (`plekton-fit`), metrics (`plekton-metrics`), and sparsification (`plekton-sparsify`).

## 1. Stochastic Block Model (SBM)

Nodes are partitioned into $K$ communities; each pair of nodes in blocks $(a, b)$ connects independently with probability $B_{ab}$ (either uniform intra/inter probabilities `p_in` / `p_out`, or an arbitrary symmetric $K \times K$ matrix `B` and arbitrary block sizes via `generate_sbm_from_fit`) [1].

Instead of testing all $\binom{N}{2}$ node pairs in $O(N^2)$ time, `plekton` draws geometric skip distances between consecutive edges inside each block pair, running in $O(E)$ time for sparse graphs.

## 2. Degree-Corrected SBM (DC-SBM) — Exact $O(N + E)$

A standard SBM gives every node in a block the same expected degree, which misses the hubs seen in real networks. DC-SBM assigns each node a non-negative activity weight $\theta_i$ (normalized to unit mean within its block), either drawn from a heavy-tailed $\text{Pareto}(\gamma - 1)$ distribution ($\gamma > 2$) or estimated from a real graph via `plekton-fit` [2].

Each edge $(u, v)$ between blocks $(a, b)$ is present independently with exact probability $p_{uv} = \min(1,\, B_{ab}\,\theta_u\,\theta_v)$. Partitioning each block's positive-weight nodes into **dyadic weight buckets** $[2^r, 2^{r+1})$ bounds the rejection ceiling within every bucket pair to at most $4\times$ the true probability (acceptance rate $> 25\%$ on every proposal), giving exact sampling in $O(N + E + K^2 \log^2 N)$ expected time without collapsing to $O(N^2)$ under heavy tails.

## 3. Contextual SBM (cSBM)

Extends SBM / DC-SBM by pairing topology with correlated $F$-dimensional node features so graph neural networks can be stressed across both structural homophily and feature signal-to-noise ratio [3]. Each class $c$ has a centroid vector scaled by $\mu$, and each node receives class-conditional noise (`gaussian`, `uniform`, or `laplacian`), optional correlated covariance via Cholesky factorization, per-node feature label noise (`feat_noise_ratio`), optional triangle closing (`p_triangle`), and optional component stitching (`ensure_connected`).

## 4. Barabási–Albert (BA)

Builds a scale-free graph by preferential attachment: starting from an initial clique of `m0` nodes, each arriving node connects to `m` distinct existing nodes chosen with probability proportional to their current degree [4].

## 5. Multi-Attribute Mixing Graphs (`mixing_graph`)

Generates graphs that satisfy simultaneous homophily, heterophily, or full category-pair mixing targets across **multiple categorical node attributes at once** (e.g., heterophilous on sex and homophilous on age at the same time):

- **Undirected**: forms joint attribute cells, solves per-block expected ties with Iterative Proportional Fitting (IPF) so every attribute target and degree budget is met simultaneously, respects optional `groups` confinement and `max_degree` caps (including exact matchings when `max_degree = 1`), and optionally rewires edge pairs to match a target global transitivity (`clustering`) without breaking degrees or mixing.
- **Directed**: draws outgoing edges weighted by log-normal node popularity (`popularity_spread`) multiplied by IPF-fitted attribute factors.
- **Tie strengths**: optional `(0, 1)` weights drawn on the logit scale with per-attribute homophily bonuses (`strength`).

## Companion Analysis Primitives

`plekton` pairs these generators with structural evaluation and reduction primitives: degree assortativity and honest structural metrics (`plekton-metrics`) [5], Spielman–Srivastava approximate effective-resistance sparsification (`plekton-sparsify`) [6], and Speaker–Listener Label Propagation (SLPA) for overlapping community detection (`plekton-fit`) [7].

## Ecosystem Positioning

| Capability | `plekton` | `NetworkX` | `python-igraph` | `graph-tool` |
|---|:---:|:---:|:---:|:---:|
| Sparse SBM in $O(E)$ | ✓ | ✗ ($O(N^2)$) | ✓ | ✓ |
| Exact $O(N+E)$ DC-SBM | ✓ | ✗ | ✗ | ✓ |
| cSBM (correlated features + PyG tensors) | ✓ | ✗ | ✗ | ✗ |
| Multi-attribute IPF `mixing_graph` | ✓ | ✗ | ✗ | ✗ |
| Spectral approx-ER & metric backbone | ✓ | ✗ | ✗ | ✗ |
| Lightweight install (`numpy`-only wheel) | ✓ | ✓ | ✓ | ✗ (C++/Boost) |

See [benchmarks/README.md](../benchmarks/README.md) and [`benchmarks/bench.py`](../benchmarks/bench.py) for reproducible throughput and peak-memory benchmarks.

## References

[1] P. W. Holland, K. B. Laskey, and S. Leinhardt. “Stochastic blockmodels: First steps.” *Social Networks*, 5(2), 109–137, 1983. [DOI](https://doi.org/10.1016/0378-8733(83)90021-7).

[2] B. Karrer and M. E. J. Newman. “Stochastic blockmodels and community structure in networks.” *Physical Review E*, 83(1), 016107, 2011. [DOI](https://doi.org/10.1103/PhysRevE.83.016107).

[3] Y. Deshpande, S. Sen, A. Montanari, and E. Mossel. “Contextual Stochastic Block Models.” *Advances in Neural Information Processing Systems*, volume 31, 2018. [URL](https://proceedings.neurips.cc/paper/2018/hash/08fc80de8121419136e443a70489c123-Abstract.html).

[4] A.-L. Barabási and R. Albert. “Emergence of Scaling in Random Networks.” *Science*, 286(5439), 509–512, 1999. [DOI](https://doi.org/10.1126/science.286.5439.509).

[5] M. E. J. Newman. “Assortative Mixing in Networks.” *Physical Review Letters*, 89(20), 208701, 2002. [DOI](https://doi.org/10.1103/PhysRevLett.89.208701).

[6] D. A. Spielman and N. Srivastava. “Graph Sparsification by Effective Resistances.” *SIAM Journal on Computing*, 40(6), 1913–1926, 2011. [DOI](https://doi.org/10.1137/080734029).

[7] J. Xie, B. K. Szymanski, and X. Liu. “SLPA: Uncovering Overlapping Communities in Social Networks via a Speaker-Listener Interaction Dynamic Process.” *Proceedings of the 11th IEEE International Conference on Data Mining Workshops (ICDMW)*, 344–349, 2011. [DOI](https://doi.org/10.1109/ICDMW.2011.154).
