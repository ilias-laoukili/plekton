# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""``plekton.Graph`` — the one edge container every plekton function takes and returns.

A graph is ``n_nodes`` nodes and an ``(E, 2)`` int64 array of edges, one row per
edge. Undirected graphs store each edge once, as ``u < v``; directed graphs store
each arc ``(u, v)`` once, and ``(u, v)`` and ``(v, u)`` are two different arcs.
Both are simple: no self-loop, no repeated edge. Node labels, node features,
per-edge attributes and generator metadata ride along, all read-only.

The PyG layout, ``edge_index`` of shape ``(2, 2E)`` with every undirected edge in
both directions, is derived on demand rather than stored.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, final

import numpy as np

from plekton._graph_io import GraphInput, coerce_graph_json
from plekton._plekton import json_to_graph as _json_to_graph

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt
    from numpy.typing import NDArray

__all__ = ["Graph", "as_graph", "node_labels", "trusted_graph"]

# An edge row holds two endpoints; an edge_index holds two rows.
_PAIR: Final = 2


def _readonly(arr: np.ndarray) -> np.ndarray:
    """Return a non-writeable view of ``arr`` (no copy)."""
    view = arr.view()
    view.flags.writeable = False
    return view


def _freeze(value: object) -> object:
    """Deep read-only copy of a metadata value: mappings, sequences and arrays."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    return value


def _thaw(value: object) -> object:
    """Plain, picklable copy of a frozen metadata value."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _check_n_nodes(n_nodes: int) -> int:
    if isinstance(n_nodes, bool):
        msg = "n_nodes must be an integer, not a bool"
        raise TypeError(msg)
    value = operator.index(n_nodes)  # refuses floats with a TypeError
    if value < 0:
        msg = f"n_nodes must be >= 0, got {value}"
        raise ValueError(msg)
    return value


def _check_integer_dtype(arr: np.ndarray, what: str) -> None:
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        msg = f"{what} must hold integer node ids, got dtype {arr.dtype}"
        raise TypeError(msg)


def _check_in_range(rows: np.ndarray, n_nodes: int, what: str) -> None:
    """Refuse any id outside ``[0, n_nodes)`` in ``(K, 2)`` rows, naming the first pair."""
    if int(rows.min()) >= 0 and int(rows.max()) < n_nodes:
        return
    bad = int(np.flatnonzero(((rows < 0) | (rows >= n_nodes)).any(axis=1))[0])
    u, v = (int(i) for i in rows[bad])
    msg = f"{what} ({u}, {v}) at position {bad} is out of range for n_nodes={n_nodes}"
    raise IndexError(msg)


def _first_repeat(rows: np.ndarray) -> tuple[int, int] | None:
    """Positions ``(i, j)``, ``i < j``, of the first repeated row, or ``None``."""
    if rows.shape[0] < _PAIR:
        return None
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    ordered = rows[order]
    same = np.flatnonzero((ordered[1:] == ordered[:-1]).all(axis=1))
    if same.size == 0:
        return None
    k = int(same[0])
    first, second = sorted((int(order[k]), int(order[k + 1])))
    return first, second


def _check_simple(edges: np.ndarray, *, directed: bool) -> None:
    """Refuse self-loops and repeats; undirected graphs also need ``u < v``."""
    u, v = edges[:, 0], edges[:, 1]
    loops = np.flatnonzero(u == v)
    if loops.size:
        i = int(loops[0])
        msg = f"edge ({u[i]}, {v[i]}) at row {i} is a self-loop; a plekton.Graph is a simple graph"
        raise ValueError(msg)
    if directed:
        repeat = _first_repeat(edges)
        if repeat is not None:
            i, j = repeat
            msg = f"arc ({u[i]}, {v[i]}) appears twice (rows {i} and {j})"
            raise ValueError(msg)
        return
    # Undirected: compare the edges as unordered pairs, so (u, v) and (v, u) collide.
    repeat = _first_repeat(np.sort(edges, axis=1))
    if repeat is not None:
        i, j = repeat
        if u[i] == u[j]:
            msg = f"edge ({u[i]}, {v[i]}) appears twice (rows {i} and {j})"
        else:
            msg = (
                f"edge {{{min(u[i], v[i])}, {max(u[i], v[i])}}} appears in both directions "
                f"(rows {i} and {j}); an undirected plekton.Graph lists each edge once. "
                "A PyG edge_index, transposed or not, is read with Graph.from_edge_index(...)"
            )
        raise ValueError(msg)
    reversed_rows = np.flatnonzero(u > v)
    if reversed_rows.size:
        i = int(reversed_rows[0])
        msg = (
            f"edge ({u[i]}, {v[i]}) at row {i} is not canonical: an undirected plekton.Graph "
            "stores every edge as u < v. Sort each row first, e.g. np.sort(edges, axis=1)"
        )
        raise ValueError(msg)


def _check_edges(edges: npt.ArrayLike, n_nodes: int, *, directed: bool) -> NDArray[np.int64]:
    arr = np.asarray(edges)
    if arr.size == 0:
        return np.zeros((0, _PAIR), dtype=np.int64)
    _check_integer_dtype(arr, "edges")
    if arr.ndim != _PAIR or arr.shape[1] != _PAIR:
        if arr.ndim == _PAIR and arr.shape[0] == _PAIR:
            msg = (
                f"edges has shape {arr.shape}, one column per edge; a plekton.Graph takes one "
                "row per edge, shape (E, 2). Use Graph.from_edge_index(edge_index, n_nodes=...) "
                "for a PyG edge_index, or pass the transpose .T of a one-way (2, E) list"
            )
        else:
            msg = f"edges must have shape (E, 2), got {arr.shape}"
        raise ValueError(msg)
    _check_in_range(arr, n_nodes, "edge")
    out: NDArray[np.int64] = np.ascontiguousarray(arr, dtype=np.int64)
    _check_simple(out, directed=directed)
    return out


def _check_labels(labels: npt.ArrayLike | None, n_nodes: int) -> NDArray[np.int64] | None:
    if labels is None:
        return None
    arr = np.asarray(labels)
    if arr.size == 0 and n_nodes == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        msg = f"labels must hold integer community ids, got dtype {arr.dtype}"
        raise TypeError(msg)
    if arr.shape != (n_nodes,):
        msg = f"labels must have shape ({n_nodes},), one per node, got {arr.shape}"
        raise ValueError(msg)
    if n_nodes and int(arr.min()) < 0:
        msg = "labels must be non-negative"
        raise ValueError(msg)
    out: NDArray[np.int64] = np.ascontiguousarray(arr, dtype=np.int64)
    return out


def _check_features(x: npt.ArrayLike | None, n_nodes: int) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
    ):
        msg = f"x must hold numeric node features, got dtype {arr.dtype}"
        raise TypeError(msg)
    if arr.ndim != _PAIR or arr.shape[0] != n_nodes:
        msg = f"x must have shape ({n_nodes}, F), one row per node, got {arr.shape}"
        raise ValueError(msg)
    return np.ascontiguousarray(arr)


def _check_edge_attrs(
    edge_attrs: Mapping[str, npt.ArrayLike] | None,
    n_edges: int,
) -> dict[str, np.ndarray]:
    if edge_attrs is None:
        return {}
    out: dict[str, np.ndarray] = {}
    for name, values in edge_attrs.items():
        if not isinstance(name, str):
            msg = f"edge_attrs keys must be strings, got {name!r}"
            raise TypeError(msg)
        arr = np.asarray(values)
        if arr.ndim == 0 or arr.shape[0] != n_edges:
            msg = (
                f"edge_attrs[{name!r}] has shape {arr.shape}; it needs one entry per edge, "
                f"in the order of `edges` (E = {n_edges})"
            )
            raise ValueError(msg)
        out[name] = np.ascontiguousarray(arr)
    return out


def _fill(  # one argument per stored field
    graph: Graph,
    edges: np.ndarray,
    n_nodes: int,
    labels: np.ndarray | None,
    x: np.ndarray | None,
    *,
    directed: bool,
    metadata: Mapping[str, Any] | None,
    edge_attrs: Mapping[str, np.ndarray],
) -> Graph:
    """Store already-checked parts in ``graph``, all of them read-only."""
    state: dict[str, object] = {
        "_edges": _readonly(edges),
        "_n_nodes": n_nodes,
        "_labels": None if labels is None else _readonly(labels),
        "_x": None if x is None else _readonly(x),
        "_directed": directed,
        "_metadata": _freeze(metadata or {}),
        "_edge_attrs": MappingProxyType({k: _readonly(v) for k, v in edge_attrs.items()}),
    }
    for name, value in state.items():
        object.__setattr__(graph, name, value)
    return graph


def trusted_graph(
    edges: np.ndarray,
    n_nodes: int,
    *,
    labels: np.ndarray | None = None,
    x: np.ndarray | None = None,
    directed: bool = False,
    metadata: Mapping[str, Any] | None = None,
    edge_attrs: Mapping[str, np.ndarray] | None = None,
) -> Graph:
    """Wrap arrays plekton itself produced, without the O(E log E) checks.

    For Rust generator output, subgraphs and unpickled graphs, whose edges are
    simple by construction. Only the cheap structural invariants are checked.
    """
    if edges.dtype != np.int64 or edges.ndim != _PAIR or edges.shape[1] != _PAIR:
        msg = f"internal: expected int64 (E, 2) edges, got {edges.dtype} {edges.shape}"
        raise RuntimeError(msg)
    if labels is not None and labels.shape != (n_nodes,):
        msg = f"internal: expected ({n_nodes},) labels, got {labels.shape}"
        raise RuntimeError(msg)
    return _fill(
        object.__new__(Graph),
        np.ascontiguousarray(edges),
        n_nodes,
        labels,
        x,
        directed=directed,
        metadata=metadata,
        edge_attrs=dict(edge_attrs or {}),
    )


def _restore(state: dict[str, Any]) -> Graph:
    """Unpickling hook: rebuild a ``Graph`` from the plain state ``__reduce__`` saved."""
    return trusted_graph(
        state["edges"],
        state["n_nodes"],
        labels=state["labels"],
        x=state["x"],
        directed=state["directed"],
        metadata=state["metadata"],
        edge_attrs=state["edge_attrs"],
    )


@final
class Graph:
    """An immutable simple graph: ``n_nodes`` nodes and one row per edge.

    Parameters
    ----------
    edges
        Integer array-like of shape ``(E, 2)``, one row per edge. Undirected
        graphs list each edge once as ``u < v``; directed graphs list each arc
        ``(u, v)`` once, and ``(v, u)`` is a different arc.
    n_nodes
        Number of nodes, isolated ones included: node ids are ``0..n_nodes``.
    labels
        Optional community id per node, shape ``(n_nodes,)``.
    x
        Optional numeric feature matrix, shape ``(n_nodes, F)``.
    directed
        Whether edges are arcs.
    metadata
        Optional mapping of graph-level values (for generators: ``homophily`` and
        the ``seed`` that produced the graph). Stored as a deep read-only copy.
    edge_attrs
        Optional mapping of per-edge arrays, each with one entry per row of
        ``edges`` (for ``mixing_graph``: ``"strength"``).

    Raises
    ------
    TypeError
        If ids are not integers (floats are refused, never truncated).
    ValueError
        If ``edges`` is not ``(E, 2)`` (a ``(2, E)`` array is refused with a
        pointer to :meth:`from_edge_index`), holds a self-loop or a repeated
        edge, lists an undirected edge in both directions, or stores one as
        ``(v, u)`` with ``v > u``.
    IndexError
        If an id falls outside ``[0, n_nodes)``.

    Notes
    -----
    Immutability covers the ``Graph``: its attributes cannot be reassigned and
    every array it exposes is read-only. **It does not cover the caller's
    array.** When ``edges`` (or ``labels``, ``x``, an edge attribute) already has
    the stored dtype and layout, the ``Graph`` keeps a read-only *view* of it,
    without copying, and writing to the original array afterwards changes the
    graph underneath. Pass a copy if the original will be modified.

    """

    __slots__ = ("_directed", "_edge_attrs", "_edges", "_labels", "_metadata", "_n_nodes", "_x")

    _edges: NDArray[np.int64]
    _n_nodes: int
    _labels: NDArray[np.int64] | None
    _x: np.ndarray | None
    _directed: bool
    _metadata: Mapping[str, Any]
    _edge_attrs: Mapping[str, np.ndarray]

    def __init__(
        self,
        edges: npt.ArrayLike,
        *,
        n_nodes: int,
        labels: npt.ArrayLike | None = None,
        x: npt.ArrayLike | None = None,
        directed: bool = False,
        metadata: Mapping[str, Any] | None = None,
        edge_attrs: Mapping[str, npt.ArrayLike] | None = None,
    ) -> None:
        """Validate and store a graph; see the class docstring for the rules."""
        if not isinstance(directed, bool):
            msg = f"directed must be a bool, got {type(directed).__name__}"
            raise TypeError(msg)
        n = _check_n_nodes(n_nodes)
        checked = _check_edges(edges, n, directed=directed)
        _fill(
            self,
            checked,
            n,
            _check_labels(labels, n),
            _check_features(x, n),
            directed=directed,
            metadata=metadata,
            edge_attrs=_check_edge_attrs(edge_attrs, checked.shape[0]),
        )

    # ── Alternative constructors ──────────────────────────────────────────────

    @classmethod
    def from_edge_index(
        cls,
        edge_index: npt.ArrayLike,
        *,
        n_nodes: int,
        labels: npt.ArrayLike | None = None,
        x: npt.ArrayLike | None = None,
        directed: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Graph:
        """Build a graph from a PyG ``edge_index`` of shape ``(2, M)``.

        Undirected (the default): every edge must appear **exactly once in each
        direction**, so ``M = 2E``; a self-loop, a repeated column or a column
        without its reverse is refused. The edges keep the order of their
        ``u < v`` columns, so ``Graph.from_edge_index(g.edge_index, ...)`` gives
        back ``g.edges`` exactly. Directed: each column is one arc.
        """
        n = _check_n_nodes(n_nodes)
        arr = np.asarray(edge_index)
        if arr.size == 0:
            empty = np.zeros((0, _PAIR), dtype=np.int64)
            return cls(empty, n_nodes=n, labels=labels, x=x, directed=directed, metadata=metadata)
        _check_integer_dtype(arr, "edge_index")
        if arr.ndim != _PAIR or arr.shape[0] != _PAIR:
            msg = f"edge_index must have shape (2, M), got {arr.shape}"
            raise ValueError(msg)
        if directed:
            return cls(arr.T, n_nodes=n, labels=labels, x=x, directed=True, metadata=metadata)
        _check_in_range(arr.T, n, "edge_index column")
        columns = np.ascontiguousarray(arr.T, dtype=np.int64)
        src, dst = columns[:, 0], columns[:, 1]
        loops = np.flatnonzero(src == dst)
        if loops.size:
            i = int(loops[0])
            msg = f"edge_index column {i} is a self-loop ({src[i]}, {dst[i]})"
            raise ValueError(msg)
        forward = columns[src < dst]
        backward = columns[src > dst][:, ::-1]
        for rows, direction in ((forward, "u < v"), (backward, "u > v")):
            repeat = _first_repeat(np.ascontiguousarray(rows))
            if repeat is not None:
                a, b = (int(i) for i in rows[repeat[0]])
                msg = (
                    f"edge {{{a}, {b}}} appears more than once in the {direction} direction; "
                    "an undirected edge_index lists every edge exactly once in each direction"
                )
                raise ValueError(msg)
        f_sorted = forward[np.lexsort((forward[:, 1], forward[:, 0]))]
        b_sorted = backward[np.lexsort((backward[:, 1], backward[:, 0]))]
        if forward.shape != backward.shape or not np.array_equal(f_sorted, b_sorted):
            missing = _first_unmatched(f_sorted, b_sorted)
            msg = (
                f"edge {{{missing[0]}, {missing[1]}}} is not listed in both directions; an "
                "undirected edge_index lists every edge exactly once in each direction "
                "(pass directed=True for a list of arcs)"
            )
            raise ValueError(msg)
        return _fill(
            object.__new__(cls),
            np.ascontiguousarray(forward),
            n,
            _check_labels(labels, n),
            _check_features(x, n),
            directed=False,
            metadata=metadata,
            edge_attrs={},
        )

    @classmethod
    def from_json(cls, doc: GraphInput) -> Graph:
        """Load a ``plekton.graph/v1`` document (JSON string, parsed dict, or path).

        Node ids must be ``0..n_nodes`` in order. The edges are validated like any
        other input, and the document's ``metadata`` becomes ``Graph.metadata``.
        """
        edges, labels, x, metadata_json = _json_to_graph(coerce_graph_json(doc))
        metadata: dict[str, Any] = json.loads(metadata_json)
        return cls(edges, n_nodes=int(labels.shape[0]), labels=labels, x=x, metadata=metadata)

    # ── Read-only views ───────────────────────────────────────────────────────

    @property
    def edges(self) -> NDArray[np.int64]:
        """Read-only int64 ``(E, 2)`` array, one row per edge (``u < v`` if undirected)."""
        return self._edges

    @property
    def n_nodes(self) -> int:
        """Number of nodes, isolated ones included."""
        return self._n_nodes

    @property
    def n_edges(self) -> int:
        """Number of edges (arcs, if directed)."""
        return int(self._edges.shape[0])

    @property
    def labels(self) -> NDArray[np.int64] | None:
        """Read-only community id per node, or ``None``."""
        return self._labels

    @property
    def x(self) -> np.ndarray | None:
        """Read-only ``(n_nodes, F)`` feature matrix, or ``None``."""
        return self._x

    @property
    def directed(self) -> bool:
        """Whether the edges are arcs."""
        return self._directed

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only graph-level values (generators record ``homophily`` and ``seed``)."""
        return self._metadata

    @property
    def edge_attrs(self) -> Mapping[str, np.ndarray]:
        """Read-only per-edge arrays, aligned with the rows of :attr:`edges`."""
        return self._edge_attrs

    @property
    def edge_index(self) -> NDArray[np.int64]:
        """PyG ``edge_index``, rebuilt on every access (not cached).

        Undirected: shape ``(2, 2E)``, the ``E`` edges first, then the same edges
        reversed, which costs 32 bytes per edge for as long as the result is kept.
        Directed: shape ``(2, E)``, one column per arc.
        """
        edges = self._edges
        if self._directed:
            directed_out: NDArray[np.int64] = np.ascontiguousarray(edges.T)
            return directed_out
        n = edges.shape[0]
        out: NDArray[np.int64] = np.empty((_PAIR, 2 * n), dtype=np.int64)
        out[0, :n] = edges[:, 0]
        out[0, n:] = edges[:, 1]
        out[1, :n] = edges[:, 1]
        out[1, n:] = edges[:, 0]
        return out

    # ── Derived graphs ────────────────────────────────────────────────────────

    def edge_subgraph(self, mask: npt.ArrayLike) -> Graph:
        """Keep the edges where ``mask`` is ``True`` — every node stays.

        ``mask`` is a boolean array with one entry per edge, as the
        ``plekton.sparsify`` selectors return. Labels, features and the selected
        rows of every edge attribute are kept; ``metadata`` is dropped, because
        values measured on the full graph no longer describe the subgraph.
        """
        keep = np.asarray(mask)
        if keep.dtype != np.bool_ or keep.shape != (self.n_edges,):
            msg = (
                f"mask must be a boolean array of shape ({self.n_edges},), "
                f"got {keep.dtype} {keep.shape}"
            )
            raise ValueError(msg)
        return trusted_graph(
            self._edges[keep],
            self._n_nodes,
            labels=self._labels,
            x=self._x,
            directed=self._directed,
            edge_attrs={name: values[keep] for name, values in self._edge_attrs.items()},
        )

    # ── Python protocol ───────────────────────────────────────────────────────

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse every assignment: a ``Graph`` never changes."""
        msg = f"plekton.Graph is immutable; cannot set {name!r}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion: a ``Graph`` never changes."""
        msg = f"plekton.Graph is immutable; cannot delete {name!r}"
        raise AttributeError(msg)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle support (the read-only mappings themselves cannot be pickled)."""
        state: dict[str, Any] = {
            "edges": self._edges,
            "n_nodes": self._n_nodes,
            "labels": self._labels,
            "x": self._x,
            "directed": self._directed,
            "metadata": _thaw(self._metadata),
            "edge_attrs": dict(self._edge_attrs),
        }
        return (_restore, (state,))

    def __repr__(self) -> str:
        """Summarise the graph without printing its arrays."""
        parts = [f"n_nodes={self._n_nodes}", f"n_edges={self.n_edges}"]
        if self._directed:
            parts.append("directed=True")
        if self._labels is not None:
            parts.append("labels")
        if self._x is not None:
            parts.append(f"x={self._x.shape}")
        if self._edge_attrs:
            parts.append(f"edge_attrs={sorted(self._edge_attrs)}")
        return f"Graph({', '.join(parts)})"


def _first_unmatched(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Return an edge present in one list of pairs but not the other."""
    seen_a = {(int(u), int(v)) for u, v in a}
    seen_b = {(int(u), int(v)) for u, v in b}
    return min(seen_a ^ seen_b)


def as_graph(
    graph: Graph | npt.ArrayLike,
    n_nodes: int | None,
    *,
    caller: str,
    allow_directed: bool = False,
) -> Graph:
    """Accept a ``Graph``, or an ``(E, 2)`` edge array together with ``n_nodes``.

    A raw array is validated as an undirected ``Graph`` on every call; build the
    ``Graph`` once when the same edges feed several functions.
    """
    if isinstance(graph, Graph):
        if n_nodes is not None:
            msg = f"{caller}: n_nodes comes from the Graph; pass n_nodes= only with an edge array"
            raise TypeError(msg)
        result = graph
    else:
        if n_nodes is None:
            msg = (
                f"{caller}: pass a plekton.Graph, or an (E, 2) edge array together with n_nodes=..."
            )
            raise TypeError(msg)
        result = Graph(graph, n_nodes=n_nodes)
    if result.directed and not allow_directed:
        msg = f"{caller} needs an undirected graph; this plekton.Graph is directed"
        raise ValueError(msg)
    return result


def node_labels(
    graph: Graph,
    labels: npt.ArrayLike | None,
    *,
    caller: str,
    hint: str = "pass labels=...",
) -> NDArray[np.int64]:
    """``labels`` if given (validated against ``graph``), else ``graph.labels``."""
    if labels is not None:
        checked = _check_labels(labels, graph.n_nodes)
        if checked is None:  # pragma: no cover - labels is not None here
            msg = "internal: labels vanished"
            raise RuntimeError(msg)
        return checked
    if graph.labels is None:
        msg = f"{caller}: the graph carries no labels; {hint}"
        raise ValueError(msg)
    return graph.labels
