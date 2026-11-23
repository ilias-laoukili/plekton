# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.metrics`` — structural metrics over edge lists."""

from __future__ import annotations

import numpy as np
import pytest

gg = pytest.importorskip("plekton")


def test_triangle_metrics() -> None:
    """A single triangle has density 1, transitivity 1, clustering 1, 1 component."""
    edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    assert gg.metrics.density(3, 3) == pytest.approx(1.0)
    assert gg.metrics.components(3, edges) == 1
    assert gg.metrics.transitivity(3, edges) == pytest.approx(1.0)
    assert gg.metrics.clustering(3, edges) == pytest.approx(1.0)
    # Regular graph: assortativity denominator is zero -> None (never fabricated 0.0)
    assert gg.metrics.assortativity(3, edges) is None


def test_homophily_and_attribute_mixing() -> None:
    """Homophily and attribute_mixing agree on within-category share."""
    edges = np.array([[0, 1], [2, 3], [0, 2]], dtype=np.int64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    assert gg.metrics.homophily(edges, labels) == pytest.approx(2.0 / 3.0)
    mix = gg.metrics.attribute_mixing(edges, labels)
    assert mix is not None
    assert mix["within"] == pytest.approx(2.0 / 3.0)
    assert mix["random"] == pytest.approx(0.5)
    assert mix["r"] == pytest.approx((2.0 / 3.0 - 0.5) / 0.5)


def test_degree_summary_and_ks_distance() -> None:
    """Degree summary captures hubs and KS distance is 0 for identical distributions."""
    star = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int64)
    summary = gg.metrics.degree_summary(5, star)
    assert summary is not None
    assert summary["max"] == 4
    assert summary["p50"] == 1
    assert summary["gini"] > 0.0

    assert gg.metrics.degree_ks(5, star, 5, star) == pytest.approx(0.0)
    ring = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int64)
    ks = gg.metrics.degree_ks(5, star, 5, ring)
    assert ks is not None
    assert ks > 0.5
