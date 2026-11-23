# plekton-fit

Blockmodel estimation (inverse of DC-SBM generation) and community detection from plain `&[(usize, usize)]` edge lists. Depends only on `rand`.

- `fit(&edges, n_nodes, partition)` — method-of-moments estimation of block sizes, $K \times K$ mixing matrix `b_matrix`, and per-node degree correction `theta` (unit mean per block, in generator node order). Projects $K^2$ against `MAX_MIXING_ENTRIES` before allocating.
- `detect_label_propagation(&edges, n_nodes, seed, max_rounds)` — deterministic seeded asynchronous label propagation returning `Partition::Detected`.
- `detect_overlapping(&edges, n_nodes, seed, rounds, threshold)` — deterministic SLPA overlapping community detection (Xie, Szymanski & Liu, 2011).

## Example

```rust
use plekton_fit::{detect_label_propagation, fit, Partition};

let edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
let declared = Partition::Declared {
    attribute: "group".to_owned(),
    labels: vec![0, 0, 0, 1, 1, 1],
};
let fitted = fit(&edges, 6, declared)?;
assert_eq!(fitted.block_sizes, vec![3, 3]);

let detected = detect_label_propagation(&edges, 6, 42, 20);
assert_eq!(detected.provenance(), "detected");
```
