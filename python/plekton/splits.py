# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Leakage-free train/validation/test splitters for graph machine learning.

Random node splits leak neighbourhood information whenever nodes are grouped
into cliques, accounts, hosts, or temporal windows. This module provides three
explicit regimes with loud degeneracy checks:

- :func:`grouped_split` — group-disjoint stratified bin-packing so no group
  straddles the train/val/test boundary and positive-class priors stay balanced.
- :func:`temporal_split` — chronological split with optional interval embargo
  purging records whose end time crosses the cut.
- :func:`grouped_temporal_split` — requires both chronological order and group
  disjointness, refusing loudly when groups straddle the temporal cut.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plekton._seed import resolve_seed

_N_SPLITS = 3
_SPLIT_NAMES = ("train", "val", "test")
_FRACTION_SUM_TOL = 1e-9

__all__ = [
    "DegenerateSplitError",
    "SplitInvariantError",
    "grouped_split",
    "grouped_temporal_split",
    "split_diagnostics",
    "temporal_split",
]


class SplitInvariantError(RuntimeError):
    """A split violated disjointness, coverage, or group purity.

    Raised rather than asserted so invariants remain enforced even under ``python -O``.
    """


class DegenerateSplitError(RuntimeError):
    """A split could not honour the requested proportions or class balance.

    Raised when realised split sizes or per-split positive-class priors drift too
    far from the request — typically because one group dominates the corpus.
    Structured diagnostics are attached at :attr:`diagnostics`.
    """

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        """Carry structured split diagnostics alongside the error message."""
        super().__init__(message)
        self.diagnostics = diagnostics


def split_diagnostics(
    masks: tuple[np.ndarray, np.ndarray, np.ndarray],
    group_keys: list[str],
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Describe the realised shape of a train/val/test split.

    Returns a dict with ``counts``, ``realised_fracs``, ``groups_per_split``,
    ``positive_priors`` (per split, ``0.0`` when unlabelled or empty),
    ``global_positive_prior``, and ``n_groups``.
    """
    n = len(group_keys)
    keys = np.asarray(group_keys)
    counts = [int(m.sum()) for m in masks]
    priors: list[float] = []
    if labels is not None:
        y = np.asarray(labels)
        priors.extend(float(y[m].mean()) if int(m.sum()) else 0.0 for m in masks)
        global_prior = float(y.mean()) if n else 0.0
    else:
        priors = [0.0, 0.0, 0.0]
        global_prior = 0.0

    return {
        "counts": counts,
        "realised_fracs": [c / n if n else 0.0 for c in counts],
        "groups_per_split": [len(set(keys[m].tolist())) for m in masks],
        "positive_priors": priors,
        "global_positive_prior": global_prior,
        "n_groups": len(set(group_keys)),
    }


class _GroupPacker:
    """Water-fill whole groups into three splits against node and positive-class budgets."""

    def __init__(
        self,
        groups: dict[str, list[int]],
        fracs: tuple[float, float, float],
        labels: np.ndarray | None,
        seed: int,
    ) -> None:
        self.groups = groups
        self.labels = None if labels is None else np.asarray(labels)
        self.rng = np.random.default_rng(seed)
        n = sum(len(v) for v in groups.values())
        self.masks = [np.zeros(n, dtype=bool) for _ in range(_N_SPLITS)]
        self.counts = [0, 0, 0]
        self.positive_counts = [0, 0, 0]
        self.targets = [f * n for f in fracs]
        total_positives = 0 if self.labels is None else int(self.labels.sum())
        self.positive_targets = [f * total_positives for f in fracs]
        self.active = [s for s, f in enumerate(fracs) if f > 0.0]

    def assign(self, key: str, split: int) -> None:
        """Assign every node of one group to one split, updating both budgets."""
        idx = self.groups[key]
        self.masks[split][idx] = True
        self.counts[split] += len(idx)
        if self.labels is not None:
            self.positive_counts[split] += int(self.labels[idx].sum())

    def by_size_desc(self, pool: list[str]) -> list[str]:
        """Order groups largest-first, breaking equal-size ties with the seeded RNG."""
        shuffled = list(pool)
        self.rng.shuffle(shuffled)
        return sorted(shuffled, key=lambda k: -len(self.groups[k]))

    def _neediest_nodes(self) -> int:
        return min(self.active, key=lambda s: self.counts[s] - self.targets[s])

    def _neediest_positives(self) -> int:
        return min(self.active, key=lambda s: self.positive_counts[s] - self.positive_targets[s])

    def _seed_each_split(self, ranked_positives: list[str]) -> set[str]:
        seeded: set[str] = set()
        largest_first = sorted(self.active, key=lambda s: -self.targets[s])
        for split, key in zip(largest_first, ranked_positives, strict=False):
            self.assign(key, split)
            seeded.add(key)
        return seeded

    def _rescue_empty_splits(self) -> None:
        for split in self.active:
            if self.counts[split]:
                continue
            donor = max(self.active, key=lambda s: self.counts[s])
            donor_keys = [k for k, idx in self.groups.items() if self.masks[donor][idx].all()]
            smallest = min(donor_keys, key=lambda k: len(self.groups[k]))
            self.masks[donor][self.groups[smallest]] = False
            self.counts[donor] -= len(self.groups[smallest])
            self.assign(smallest, split)

    def pack(self, positive_keys: list[str], negative_keys: list[str]) -> list[np.ndarray]:
        """Run the assignment and return the three boolean masks."""
        ranked_positives = self.by_size_desc(positive_keys)
        seeded = self._seed_each_split(ranked_positives)
        for k in (key for key in ranked_positives if key not in seeded):
            self.assign(k, self._neediest_positives())
        for k in self.by_size_desc(negative_keys):
            self.assign(k, self._neediest_nodes())
        self._rescue_empty_splits()
        return self.masks


def _assert_group_disjoint(masks: list[np.ndarray], groups: dict[str, list[int]], n: int) -> None:
    """Verify disjointness, total coverage, and zero group leakage across splits."""
    train_mask, val_mask, test_mask = masks
    for name, overlap in (
        ("train/val", train_mask & val_mask),
        ("train/test", train_mask & test_mask),
        ("val/test", val_mask & test_mask),
    ):
        if overlap.any():
            msg = f"{name} masks overlap"
            raise SplitInvariantError(msg)
    if int((train_mask | val_mask | test_mask).sum()) != n:
        msg = "masks do not cover all nodes"
        raise SplitInvariantError(msg)
    for k, idx in groups.items():
        spans = int(train_mask[idx].any()) + int(val_mask[idx].any()) + int(test_mask[idx].any())
        if spans != 1:
            msg = f"group {k!r} spans {spans} splits (leakage)"
            raise SplitInvariantError(msg)


def _degeneracy_problems(
    diag: dict[str, Any],
    fracs: tuple[float, float, float],
    size_tolerance: float,
    prior_tolerance: float,
) -> list[str]:
    """Identify every way the realised split misses the requested shape."""
    active = [s for s, f in enumerate(fracs) if f > 0.0]
    n = sum(diag["counts"])
    problems: list[str] = []
    for s in active:
        realised = diag["realised_fracs"][s]
        if realised < fracs[s] * size_tolerance:
            problems.append(
                f"{_SPLIT_NAMES[s]} split realised {realised:.3f} of the corpus "
                f"for a requested {fracs[s]:.2f} ({diag['counts'][s]} of {n} nodes, "
                f"{diag['groups_per_split'][s]} of {diag['n_groups']} groups)",
            )
    global_prior = diag["global_positive_prior"]
    if global_prior > 0.0:
        for s in active:
            prior = diag["positive_priors"][s]
            ratio = prior / global_prior if global_prior else 0.0
            if prior <= 0.0 or ratio > prior_tolerance or ratio < 1.0 / prior_tolerance:
                problems.append(
                    f"{_SPLIT_NAMES[s]} positive prior {prior:.2%} is {ratio:.1f}x the "
                    f"corpus rate {global_prior:.2%}",
                )
    return problems


def grouped_split(
    group_keys: list[str],
    fracs: tuple[float, float, float] = (0.64, 0.16, 0.20),
    seed: int | None = None,
    labels: np.ndarray | None = None,
    *,
    strict: bool = True,
    size_tolerance: float = 0.5,
    prior_tolerance: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-disjoint train/val/test split keyed on ``group_keys``.

    Every node sharing a group key is placed in a single split so no group
    straddles the train/val/test boundary. When binary ``labels`` (0/1) are
    supplied, positive-bearing groups are stratified across active splits.
    ``seed=None`` draws a seed (pass one to reproduce the split); it breaks ties
    between groups of equal size.

    Raises:
        DegenerateSplitError: when ``strict=True`` and realised sizes or
            positive-class priors fall outside tolerances.
        ValueError: when ``fracs`` is not a 3-tuple summing to 1.0.

    """
    if len(fracs) != _N_SPLITS or abs(sum(fracs) - 1.0) > _FRACTION_SUM_TOL:
        msg = f"fracs must be three fractions summing to 1.0; got {fracs}"
        raise ValueError(msg)

    groups: dict[str, list[int]] = {}
    for i, k in enumerate(group_keys):
        groups.setdefault(k, []).append(i)

    if labels is not None:
        labelled = np.asarray(labels)
        positive_keys = [k for k, idx in groups.items() if bool(labelled[idx].any())]
        negative_keys = [k for k, idx in groups.items() if not bool(labelled[idx].any())]
    else:
        positive_keys, negative_keys = [], list(groups)

    masks = _GroupPacker(groups, fracs, labels, resolve_seed(seed)).pack(
        positive_keys,
        negative_keys,
    )
    _assert_group_disjoint(masks, groups, len(group_keys))
    train_mask, val_mask, test_mask = masks

    diag = split_diagnostics((train_mask, val_mask, test_mask), group_keys, labels)
    problems = _degeneracy_problems(diag, fracs, size_tolerance, prior_tolerance)
    if problems and strict:
        raise DegenerateSplitError(
            "group-disjoint split is degenerate — "
            + "; ".join(problems)
            + ". One group probably dominates the dataset: adjust sampling, relax "
            "`fracs`, or pass `strict=False` to accept it deliberately.",
            diag,
        )
    return train_mask, val_mask, test_mask


def temporal_split(
    timestamps: np.ndarray | list[float],
    fracs: tuple[float, float, float] = (0.64, 0.16, 0.20),
    *,
    end_timestamps: np.ndarray | list[float] | None = None,
    embargo_seconds: float = 0.0,
    labels: np.ndarray | None = None,
    strict: bool = True,
    size_tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split chronologically: train on the past, validate/test on the future.

    When ``end_timestamps`` is supplied, records whose interval crosses
    ``boundary + embargo_seconds`` are purged to prevent temporal look-ahead.
    """
    if len(fracs) != _N_SPLITS:
        msg = f"fracs must be a 3-tuple, got {fracs!r}"
        raise ValueError(msg)
    starts = np.asarray(timestamps, dtype=float)
    n = starts.size
    if n == 0:
        msg = "temporal_split needs at least one timestamp"
        raise ValueError(msg)
    ends = starts if end_timestamps is None else np.asarray(end_timestamps, dtype=float)
    if ends.size != n:
        msg = f"end_timestamps has {ends.size} entries, timestamps has {n}"
        raise ValueError(msg)

    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]

    def boundary(fraction: float) -> float:
        index = round(fraction * n)
        return float(sorted_starts[min(max(index - 1, 0), n - 1)])

    train_end = boundary(fracs[0])
    val_end = boundary(fracs[0] + fracs[1])

    train_mask = starts <= train_end
    val_mask = (starts > train_end) & (starts <= val_end)
    test_mask = starts > val_end

    purged = 0
    if embargo_seconds >= 0.0 and end_timestamps is not None:
        crosses_train = train_mask & (ends > train_end + embargo_seconds)
        crosses_val = val_mask & (ends > val_end + embargo_seconds)
        purged = int(crosses_train.sum() + crosses_val.sum())
        train_mask = train_mask & ~crosses_train
        val_mask = val_mask & ~crosses_val

    if strict:
        problems = []
        for index, (mask, want) in enumerate(
            zip((train_mask, val_mask, test_mask), fracs, strict=True),
        ):
            got = float(mask.sum()) / n
            if want > 0.0 and got < want * size_tolerance:
                problems.append(f"split {index} realised {got:.3f} of a requested {want:.3f}")
        if problems:
            diag = split_diagnostics((train_mask, val_mask, test_mask), [""] * n, labels)
            diag["purged_by_embargo"] = purged
            raise DegenerateSplitError(
                "temporal split is degenerate — "
                + "; ".join(problems)
                + ". Widen the time window, shorten `embargo_seconds`, or pass `strict=False`.",
                diag,
            )
    return train_mask, val_mask, test_mask


def grouped_temporal_split(
    timestamps: np.ndarray | list[float],
    group_keys: list[str],
    fracs: tuple[float, float, float] = (0.64, 0.16, 0.20),
    *,
    end_timestamps: np.ndarray | list[float] | None = None,
    embargo_seconds: float = 0.0,
    labels: np.ndarray | None = None,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split chronologically and refuse if any group straddles the cut."""
    if len(group_keys) != len(np.asarray(timestamps, dtype=float)):
        msg = "group_keys and timestamps must be the same length"
        raise ValueError(msg)

    train_mask, val_mask, test_mask = temporal_split(
        timestamps,
        fracs,
        end_timestamps=end_timestamps,
        embargo_seconds=embargo_seconds,
        labels=labels,
        strict=strict,
    )
    keys = np.asarray(group_keys)
    per_split = [set(keys[m].tolist()) for m in (train_mask, val_mask, test_mask)]
    straddlers = sorted(
        (per_split[0] | per_split[1]) & per_split[2] | (per_split[0] & per_split[1]),
    )
    if straddlers:
        diag = split_diagnostics((train_mask, val_mask, test_mask), group_keys, labels)
        diag["groups_straddling_cut"] = straddlers[:50]
        diag["n_groups_straddling_cut"] = len(straddlers)
        msg = (
            f"{len(straddlers)} group(s) appear on both sides of the chronological cut, so "
            "the split is not group-disjoint. Use `temporal_split` or `grouped_split` "
            "separately rather than claiming both."
        )
        raise DegenerateSplitError(msg, diag)
    return train_mask, val_mask, test_mask
