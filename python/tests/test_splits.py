# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.splits`` — leakage-free grouped & temporal splitters."""

from __future__ import annotations

import numpy as np
import pytest
from plekton.splits import (
    DegenerateSplitError,
    grouped_split,
    grouped_temporal_split,
    split_diagnostics,
    temporal_split,
)

_FRACTION_TOL = 0.10
_N_SPLITS = 3
_SUM_TOL = 1e-9


def _split_groups(group_keys: list[str], masks: tuple[np.ndarray, ...]) -> list[set[str]]:
    return [{group_keys[i] for i in range(len(group_keys)) if m[i]} for m in masks]


def _concentrated_corpus() -> tuple[list[str], np.ndarray]:
    """One dominant group holds 88% of rows; one smaller group is 100% positive."""
    pos_groups = [
        ("g0", 9076, 162),
        ("g1", 784, 168),
        ("g2", 122, 122),
        ("g3", 8, 6),
        ("g4", 2, 1),
        ("g5", 1, 1),
    ]
    neg_groups = [
        ("n0", 72),
        ("n1", 57),
        ("n2", 18),
        ("n3", 136),
        ("n4", 5),
        ("n5", 6),
        ("n6", 5),
        ("n7", 5),
        ("n8", 3),
    ]
    keys: list[str] = []
    labels: list[int] = []
    for key, rows, pos in pos_groups:
        keys += [key] * rows
        labels += [1] * pos + [0] * (rows - pos)
    for key, rows in neg_groups:
        keys += [key] * rows
        labels += [0] * rows
    return keys, np.array(labels)


def test_grouped_split_is_disjoint_and_covers_all_nodes() -> None:
    """Masks are pairwise disjoint and cover every node."""
    group_keys = [f"grp-{i % 20}" for i in range(400)]
    train, val, test = grouped_split(group_keys, seed=7)
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()
    assert int((train | val | test).sum()) == len(group_keys)


def test_grouped_split_no_group_spans_two_splits() -> None:
    """Every group lands in exactly one split."""
    group_keys = [f"grp-{i % 13}" for i in range(260)]
    train, val, test = grouped_split(group_keys, seed=3)
    g_train, g_val, g_test = _split_groups(group_keys, (train, val, test))
    assert g_train.isdisjoint(g_val)
    assert g_train.isdisjoint(g_test)
    assert g_val.isdisjoint(g_test)
    assert g_train | g_val | g_test == set(group_keys)


def test_grouped_split_is_deterministic() -> None:
    """Same seed reproduces identical masks; another seed changes the partition."""
    group_keys = [f"g{i % 37}" for i in range(500)]
    a = grouped_split(group_keys, seed=11)
    b = grouped_split(group_keys, seed=11)
    for ma, mb in zip(a, b, strict=True):
        assert (ma == mb).all()
    c = grouped_split(group_keys, seed=12)
    assert not all((ma == mc).all() for ma, mc in zip(a, c, strict=True))


def test_grouped_split_rejects_bad_fractions() -> None:
    """Fractions not summing to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="sum"):
        grouped_split(["a", "b", "c"], fracs=(0.5, 0.3, 0.1), seed=0)


def test_grouped_split_stratifies_positives_across_splits() -> None:
    """Positive-bearing groups are distributed across every active split."""
    group_keys: list[str] = []
    labels: list[int] = []
    for _ in range(500):
        group_keys.append("dom")
        labels.append(0)
    labels[0] = labels[1] = 1
    for g in range(5):
        for _ in range(10):
            group_keys.append(f"pos{g}")
            labels.append(1)
    for g in range(4):
        for _ in range(8):
            group_keys.append(f"neg{g}")
            labels.append(0)
    y = np.array(labels)

    train, val, test = grouped_split(group_keys, seed=1, labels=y, strict=False)
    for m in (train, val, test):
        assert m.any()
        assert int(y[m].sum()) > 0
    dom_idx = [i for i, k in enumerate(group_keys) if k == "dom"]
    assert train[dom_idx].all()


def test_split_diagnostics_reports_realised_fractions_and_priors() -> None:
    """Diagnostics report realised fractions, counts, and per-split positive priors."""
    keys = [f"g{i // 100}" for i in range(1000)]
    y = np.zeros(len(keys), dtype=int)
    y[::10] = 1
    masks = grouped_split(keys, seed=5, labels=y)

    diag = split_diagnostics(masks, keys, labels=y)
    assert len(diag["realised_fracs"]) == _N_SPLITS
    assert abs(sum(diag["realised_fracs"]) - 1.0) < _SUM_TOL
    assert sum(diag["counts"]) == len(keys)
    assert len(diag["positive_priors"]) == _N_SPLITS
    assert diag["n_groups"] == 10


def test_grouped_split_raises_on_degenerate_sizes_and_attaches_diagnostics() -> None:
    """A dominant group collapsing the test split raises DegenerateSplitError."""
    keys, y = _concentrated_corpus()
    with pytest.raises(DegenerateSplitError, match="test") as raised:
        grouped_split(keys, fracs=(0.64, 0.16, 0.20), seed=7, labels=y)
    assert raised.value.diagnostics["n_groups"] == 15
    assert raised.value.diagnostics["realised_fracs"][2] < _FRACTION_TOL


def test_grouped_split_rejects_skewed_class_priors() -> None:
    """A 100%-positive group landing alone triggers the prior-skew guard."""
    keys: list[str] = []
    labels: list[int] = []

    def add(key: str, rows: int, pos: int) -> None:
        keys.extend([key] * rows)
        labels.extend([1] * pos + [0] * (rows - pos))

    add("big", 1000, 20)
    add("mid", 600, 60)
    add("pure-pos", 300, 300)
    for i in range(20):
        add(f"neg{i}", 50, 0)

    with pytest.raises(DegenerateSplitError, match="prior"):
        grouped_split(keys, fracs=(0.64, 0.16, 0.20), seed=1, labels=np.array(labels))


def test_grouped_split_accepts_well_conditioned_corpus() -> None:
    """A balanced multi-group corpus passes cleanly within tolerances."""
    keys: list[str] = []
    labels: list[int] = []
    for i in range(200):
        n = 50
        pos = 5 if i % 3 == 0 else 0
        keys.extend([f"group-{i}"] * n)
        labels.extend([1] * pos + [0] * (n - pos))
    y = np.array(labels)

    train, val, test = grouped_split(keys, fracs=(0.64, 0.16, 0.20), seed=3, labels=y)
    diag = split_diagnostics((train, val, test), keys, labels=y)
    for realised, requested in zip(diag["realised_fracs"], (0.64, 0.16, 0.20), strict=True):
        assert abs(realised - requested) < _FRACTION_TOL
    assert all(prior > 0.0 for prior in diag["positive_priors"])


def test_temporal_split_orders_chronologically_and_applies_embargo() -> None:
    """Chronological cuts purge records whose end time crosses the boundary + embargo."""
    starts = np.arange(100, dtype=float)
    ends = starts.copy()
    ends[60] = 90.0  # straddles train/val boundary
    train, val, test = temporal_split(
        starts,
        fracs=(0.64, 0.16, 0.20),
        end_timestamps=ends,
        embargo_seconds=1.0,
    )
    assert not train[60]
    assert not (train & val).any()
    assert not (val & test).any()
    assert starts[train].max() < starts[val].min() < starts[test].min()


def test_grouped_temporal_split_refuses_straddling_groups() -> None:
    """A group present in both past and future raises DegenerateSplitError."""
    starts = np.arange(100, dtype=float)
    keys = [f"g{i % 5}" for i in range(100)]
    with pytest.raises(DegenerateSplitError, match=r"straddle|appear on both sides"):
        grouped_temporal_split(starts, keys, fracs=(0.64, 0.16, 0.20))
