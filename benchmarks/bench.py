# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Reproducible throughput & peak-memory benchmark for ``plekton``.

Each task runs inside its own fresh Python subprocess so that peak resident
memory (``ru_maxrss``) belongs solely to that engine and task — ``plekton``
processes never pay ``networkx`` import overhead, and vice versa.

Run from the repository root after building the release wheel::

    python benchmarks/bench.py                   # full suite (up to 50,000 nodes)
    python benchmarks/bench.py --quick           # fast check (up to 10,000 nodes)
    python benchmarks/bench.py --csv out.csv
    python benchmarks/bench.py --json out.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_K = 5
_AVG_DEGREE = 10.0
_HOMOPHILY = 0.75
_SEED = 42


@dataclass(frozen=True)
class BenchRow:
    """Single measured benchmark entry."""

    task: str
    engine: str
    n_nodes: int
    n_edges: int
    median_ms: float
    peak_mib: float


def _peak_mib() -> float:
    """Peak resident set size of the current process in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = float(2**20) if sys.platform == "darwin" else float(2**10)
    return peak / divisor


def _time_median_ms(fn: Callable[[], object], reps: int) -> float:
    """Warm up once, then return the median wall-clock duration in milliseconds."""
    fn()
    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1_000.0)
    finally:
        if was_enabled:
            gc.enable()
    return statistics.median(samples)


def _sbm_probs(n_nodes: int) -> tuple[float, float]:
    """Derive ``(p_in, p_out)`` for balanced blocks of size ``n_nodes / _K``."""
    block = n_nodes / _K
    p_in = min(1.0, _HOMOPHILY * _AVG_DEGREE / max(block - 1.0, 1.0))
    p_out = min(1.0, (1.0 - _HOMOPHILY) * _AVG_DEGREE / max((_K - 1) * block, 1.0))
    return p_in, p_out


def _sysctl(name: str) -> str | None:
    """Read a macOS sysctl string if available."""
    sysctl_bin = Path("/usr/sbin/sysctl")
    if not sysctl_bin.exists():
        return None
    try:
        completed = subprocess.run(  # noqa: S603
            [str(sysctl_bin), "-n", name],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _pkg_version(name: str) -> str | None:
    """Return an installed distribution's version, or ``None`` if absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _machine_info() -> dict[str, object]:
    """Collect host CPU, RAM, OS, and package version metadata."""
    mem_raw = _sysctl("hw.memsize")
    mem_gib = round(int(mem_raw) / float(2**30), 1) if mem_raw and mem_raw.isdigit() else None
    cpu = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    return {
        "processor": cpu,
        "cores": os.cpu_count(),
        "memory_gib": mem_gib,
        "system": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": _pkg_version("numpy"),
        "networkx": _pkg_version("networkx"),
        "plekton": _pkg_version("plekton"),
    }


# ── Isolated worker tasks (executed inside fresh subprocesses) ────────────────


def _worker_sbm(n_nodes: int, reps: int) -> BenchRow:
    import plekton as gg  # noqa: PLC0415

    p_in, p_out = _sbm_probs(n_nodes)
    ei, _, _ = gg.generate_sbm(n_nodes, _K, p_in, p_out, seed=_SEED)
    ms = _time_median_ms(lambda: gg.generate_sbm(n_nodes, _K, p_in, p_out, seed=_SEED), reps)
    return BenchRow("SBM (sparse)", "plekton", n_nodes, int(ei.shape[1] // 2), ms, _peak_mib())


def _worker_dcsbm(n_nodes: int, reps: int) -> BenchRow:
    import plekton as gg  # noqa: PLC0415

    p_in, p_out = _sbm_probs(n_nodes)
    ei, _, _ = gg.generate_sbm(n_nodes, _K, p_in, p_out, theta_exponent=2.5, seed=_SEED)
    ms = _time_median_ms(
        lambda: gg.generate_sbm(n_nodes, _K, p_in, p_out, theta_exponent=2.5, seed=_SEED),
        reps,
    )
    return BenchRow(
        "DC-SBM (gamma=2.5)",
        "plekton",
        n_nodes,
        int(ei.shape[1] // 2),
        ms,
        _peak_mib(),
    )


def _worker_csbm(n_nodes: int, reps: int) -> BenchRow:
    import plekton as gg  # noqa: PLC0415

    ei, _, _, _ = gg.generate_csbm(
        n_nodes,
        _K,
        _HOMOPHILY,
        _AVG_DEGREE,
        feat_dim=32,
        mu=1.5,
        seed=_SEED,
    )
    ms = _time_median_ms(
        lambda: gg.generate_csbm(
            n_nodes,
            _K,
            _HOMOPHILY,
            _AVG_DEGREE,
            feat_dim=32,
            mu=1.5,
            seed=_SEED,
        ),
        reps,
    )
    return BenchRow("cSBM (F=32)", "plekton", n_nodes, int(ei.shape[1] // 2), ms, _peak_mib())


def _worker_ba(n_nodes: int, reps: int) -> BenchRow:
    import plekton as gg  # noqa: PLC0415

    ei, _, _ = gg.generate_ba(n_nodes, m=5, m0=5, n_communities=_K, seed=_SEED)
    ms = _time_median_ms(
        lambda: gg.generate_ba(n_nodes, m=5, m0=5, n_communities=_K, seed=_SEED),
        reps,
    )
    return BenchRow("BA (m=5)", "plekton", n_nodes, int(ei.shape[1] // 2), ms, _peak_mib())


def _worker_nx_sbm(n_nodes: int, reps: int) -> BenchRow:
    import networkx as nx  # noqa: PLC0415

    p_in, p_out = _sbm_probs(n_nodes)
    block_sizes = [n_nodes // _K] * _K
    probs = [[p_in if i == j else p_out for j in range(_K)] for i in range(_K)]
    graph = nx.stochastic_block_model(block_sizes, probs, seed=_SEED)
    ms = _time_median_ms(lambda: nx.stochastic_block_model(block_sizes, probs, seed=_SEED), reps)
    return BenchRow(
        "SBM (sparse)",
        "networkx",
        n_nodes,
        graph.number_of_edges(),
        ms,
        _peak_mib(),
    )


def _worker_mixing(n_nodes: int, reps: int) -> BenchRow:
    import numpy as np  # noqa: PLC0415
    import plekton as gg  # noqa: PLC0415

    rng = np.random.default_rng(_SEED)
    sex = [f"s{v}" for v in rng.integers(0, 2, size=n_nodes).tolist()]
    age = [f"a{v}" for v in rng.integers(0, 5, size=n_nodes).tolist()]
    degrees = [6] * n_nodes
    attrs = {"sex": sex, "age": age}
    targets = {"sex": {"within": 0.2}, "age": {"within": 0.75}}
    res = gg.mixing_graph(n_nodes, attributes=attrs, degrees=degrees, mixing=targets, seed=_SEED)
    ms = _time_median_ms(
        lambda: gg.mixing_graph(
            n_nodes,
            attributes=attrs,
            degrees=degrees,
            mixing=targets,
            seed=_SEED,
        ),
        reps,
    )
    return BenchRow(
        "mixing_graph (2 attrs, d=6)",
        "plekton",
        n_nodes,
        int(res["edges"].shape[0]),
        ms,
        _peak_mib(),
    )


def _worker_sparsify(n_nodes: int, reps: int) -> BenchRow:
    import numpy as np  # noqa: PLC0415
    import plekton as gg  # noqa: PLC0415

    p_in, p_out = _sbm_probs(n_nodes)
    ei, _, _ = gg.generate_sbm(n_nodes, _K, p_in, p_out, seed=_SEED)
    edges = np.ascontiguousarray(ei[:, ei[0] < ei[1]].T)
    ms = _time_median_ms(
        lambda: gg.sparsify.approx_effective_resistance(edges, n_nodes, seed=_SEED),
        reps,
    )
    return BenchRow(
        "sparsify approx_er (JL+CG)",
        "plekton",
        n_nodes,
        int(edges.shape[0]),
        ms,
        _peak_mib(),
    )


_WORKERS: dict[str, Callable[[int, int], BenchRow]] = {
    "sbm": _worker_sbm,
    "dcsbm": _worker_dcsbm,
    "csbm": _worker_csbm,
    "ba": _worker_ba,
    "nx_sbm": _worker_nx_sbm,
    "mixing": _worker_mixing,
    "sparsify": _worker_sparsify,
}


def _run_isolated(worker_key: str, n_nodes: int, reps: int) -> BenchRow:
    """Spawn a fresh Python subprocess to measure time and isolated peak RAM."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker_key,
        "--n-nodes",
        str(n_nodes),
        "--reps",
        str(reps),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    payload: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    return BenchRow(
        task=str(payload["task"]),
        engine=str(payload["engine"]),
        n_nodes=int(payload["n_nodes"]),
        n_edges=int(payload["n_edges"]),
        median_ms=float(payload["median_ms"]),
        peak_mib=float(payload["peak_mib"]),
    )


def run_suite(*, quick: bool) -> list[BenchRow]:
    """Execute every benchmark in its own isolated subprocess."""
    sizes = (2_000, 10_000) if quick else (2_000, 10_000, 50_000)
    reps = 3 if quick else 5
    has_nx = importlib.util.find_spec("networkx") is not None
    rows: list[BenchRow] = []

    for n_nodes in sizes:
        rows.extend(_run_isolated(key, n_nodes, reps) for key in ("sbm", "dcsbm", "csbm", "ba"))
        if has_nx and n_nodes <= 10_000:
            rows.append(_run_isolated("nx_sbm", n_nodes, reps))

    rows.append(_run_isolated("mixing", 2_000, reps))
    rows.append(_run_isolated("sparsify", 2_000, reps))
    return rows


def _format_markdown(rows: list[BenchRow], machine: dict[str, object]) -> str:
    """Format benchmark rows and host metadata as a GitHub-flavored Markdown table."""
    ram_str = f" · {machine['memory_gib']} GiB RAM" if machine["memory_gib"] is not None else ""
    header = (
        f"Host: {machine['processor']} ({machine['cores']} cores{ram_str}) · "
        f"{machine['system']} · Python {machine['python']}\n\n"
        "| Task | Engine | Nodes | Edges | Median (ms) | Peak RAM (MiB) |\n"
        "|---|---|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| {r.task} | {r.engine} | {r.n_nodes:,} | {r.n_edges:,} | "
        f"{r.median_ms:.2f} | {r.peak_mib:.1f} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[BenchRow]) -> None:
    """Write benchmark rows to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "engine", "n_nodes", "n_edges", "median_ms", "peak_mib"])
        for row in rows:
            writer.writerow(
                [
                    row.task,
                    row.engine,
                    row.n_nodes,
                    row.n_edges,
                    f"{row.median_ms:.4f}",
                    f"{row.peak_mib:.2f}",
                ],
            )


def _write_json(
    path: Path,
    rows: list[BenchRow],
    machine: dict[str, object],
    *,
    quick: bool,
) -> None:
    """Write benchmark rows, measurement settings, and machine metadata to JSON."""
    payload = {
        "recorded_at": datetime.now(tz=UTC).isoformat(),
        "choices": {
            "quick": quick,
            "reps": 3 if quick else 5,
            "warm_up": "1 untimed call per subprocess; GC disabled during timed repetitions",
            "isolation": "fresh Python subprocess per (task, engine, n_nodes) for exact ru_maxrss",
            "k": _K,
            "avg_degree": _AVG_DEGREE,
            "homophily": _HOMOPHILY,
            "seed": _SEED,
        },
        "machine": machine,
        "results": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Parse CLI flags, run benchmarks, and emit Markdown (plus optional CSV/JSON)."""
    parser = argparse.ArgumentParser(description="Benchmark plekton throughput and peak RAM.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run smaller sizes (up to 10,000 nodes) and 3 repetitions.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--worker", choices=sorted(_WORKERS), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-nodes", type=int, default=2_000, help=argparse.SUPPRESS)
    parser.add_argument("--reps", type=int, default=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        row = _WORKERS[args.worker](args.n_nodes, args.reps)
        sys.stdout.write(json.dumps(asdict(row)) + "\n")
        return

    machine = _machine_info()
    rows = run_suite(quick=args.quick)
    sys.stdout.write(_format_markdown(rows, machine))
    if args.csv is not None:
        _write_csv(args.csv, rows)
    if args.json is not None:
        _write_json(args.json, rows, machine, quick=args.quick)


if __name__ == "__main__":
    main()
