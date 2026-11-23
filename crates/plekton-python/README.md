# plekton-python

The **PyO3 / maturin** binding crate compiling to `plekton._plekton` (`abi3-py311` stable ABI — one wheel per `(os, arch)` supports Python 3.11+).

Wraps:
- `plekton` (`plekton-core`) — SBM, $O(N+E)$ DC-SBM, cSBM, BA, `generate_sbm_from_fit`, native zero-copy `mixing_graph`, JSON contracts
- `plekton-fit` — `plekton.fit` (`fit_blockmodel`, `label_propagation`, `overlapping_communities`)
- `plekton-metrics` — `plekton.metrics` (structural metrics & degree distribution summaries/KS)
- `plekton-sparsify` — `plekton.sparsify` (similarity scores, approx effective resistance, selection masks, metric backbone)

Built via `maturin develop --release` (excluded from Cargo `default-members`). Never published to crates.io (`publish = false`).
