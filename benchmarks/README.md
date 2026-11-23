# Benchmarks

A single self-contained script benchmarks the release wheel across SBM, exact $O(N + E)$ DC-SBM, cSBM, Barabási–Albert, multi-attribute `mixing_graph`, and spectral effective-resistance sparsification (with optional NetworkX comparison when installed).

Each `(task, engine, n_nodes)` runs in its **own fresh Python subprocess** so that peak resident memory (`ru_maxrss`, reported in MiB) belongs solely to that engine and task, alongside full host hardware and package metadata.

## Running

```bash
maturin develop --release
python benchmarks/bench.py                     # full suite (up to 50,000 nodes)
python benchmarks/bench.py --quick             # fast check (up to 10,000 nodes)
python benchmarks/bench.py --csv results.csv   # export CSV table
python benchmarks/bench.py --json results.json # export JSON with machine metadata
```

Pure-Rust SBM / DC-SBM scaling can also be timed directly without Python:

```bash
cargo run --release -p plekton --bin sbm_bench -- --n 50000 --k 5 --d 10 --h 0.75 --theta 2.5
```
