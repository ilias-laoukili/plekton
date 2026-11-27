# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Graphs with a mixing target per node attribute.

A zero-copy wrapper over the Rust generator (``plekton::mixing``). Each
attribute gets its own target: the share of ties joining two nodes of the same
category (``within``), or the share between every pair of categories
(``matrix``). Above the random rate the graph is homophilous on that attribute;
below it, heterophilous::

    from plekton import mixing_graph

    # Marriages: one partner each, inside the household, heterophilous on sex.
    graph = mixing_graph(
        6,
        attributes={"sex": ["female", "male"] * 3},
        groups=["h1"] * 6,
        degrees=[1] * 6,
        max_degree=1,
        mixing={"sex": {"within": 0.0}},
        seed=1,
    )
    graph.edges                     # read-only int64 array of shape (E, 2)
    graph.metadata["mixing"]["sex"]  # {"target": 0.0, "within": 0.0, "random": 0.5, "r": -1.0}

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from plekton._graph import trusted_graph
from plekton._plekton import (
    generate_mixing_graph_json_py as generate_mixing_graph_json,
)
from plekton._plekton import (
    mixing_graph_native,
)
from plekton._seed import resolve_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plekton._graph import Graph

__all__ = ["generate_mixing_graph_json", "mixing_graph"]


def mixing_graph(  # noqa: PLR0913 - one keyword argument per field of MixingRequest
    nodes: int,
    *,
    attributes: dict[str, list[str]] | None = None,
    groups: list[str | None] | None = None,
    degrees: list[int] | None = None,
    mean_degree: float | None = None,
    degree_spread: float = 0.0,
    max_degree: int | None = None,
    mixing: dict[str, dict[str, Any]] | None = None,
    clustering: float | None = None,
    strength: dict[str, Any] | None = None,
    directed: bool = False,
    popularity_spread: float = 0.0,
    seed: int | None = None,
) -> Graph:
    """Generate a graph with per-attribute mixing targets.

    Returns a :class:`plekton.Graph`: undirected (each edge once, ``u < v``) or,
    with ``directed=True``, one row per arc. ``metadata`` holds the
    ``wanted_degree``, the realised ``mean_degree``, one ``mixing`` report per
    attribute (``target``, ``within``, ``random``, ``r``), the ``clustering``
    report and the ``seed`` that ran. When no node can take a tie (no node at all,
    or every wanted degree ``0``), ``wanted_degree`` and ``mean_degree`` are
    ``None``, with the reason in ``metadata["metrics_skipped"]``. With
    ``strength``, the tie strengths are in ``edge_attrs["strength"]``, one per edge.
    """
    used = resolve_seed(seed)
    result: dict[str, Any] = mixing_graph_native(
        nodes,
        attributes=attributes,
        groups=groups,
        degrees=degrees,
        mean_degree=mean_degree,
        degree_spread=degree_spread,
        max_degree=max_degree,
        mixing=mixing,
        clustering=clustering,
        strength=strength,
        directed=directed,
        popularity_spread=popularity_spread,
        seed=used,
    )
    strengths = result.get("strengths")
    metadata: dict[str, Any] = {
        "wanted_degree": result["wanted_degree"],
        "mean_degree": result["mean_degree"],
        "mixing": result["mixing"],
        "clustering": result["clustering"],
        "seed": used,
    }
    if "metrics_skipped" in result:
        metadata["metrics_skipped"] = result["metrics_skipped"]
    return trusted_graph(
        result["edges"],
        nodes,
        directed=bool(result["directed"]),
        metadata=metadata,
        edge_attrs=None if strengths is None else {"strength": strengths},
    )
