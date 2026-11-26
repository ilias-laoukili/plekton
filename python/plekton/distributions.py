# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Empirical feature distributions for synthetic graphs.

Bridges abstract synthetic graphs (Gaussian node features) and graphs whose
features follow the exact empirical distribution of a real dataset via
per-class bootstrap sampling.

Quick start
-----------
>>> from plekton.distributions import fit_from_dataset, generate_realistic
>>> model = fit_from_dataset(pyg_data)
>>> graph = generate_realistic(500, 2, 0.8, 8.0, model, seed=42)
>>> print(graph.x.shape)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plekton._generators import generate_sbm
from plekton._seed import resolve_seed

if TYPE_CHECKING:
    from torch_geometric.data import Data

_FEATURE_DIM_RANK = 2


class RealisticFeatureModel:
    """Per-class empirical feature matrices for bootstrap sampling."""

    def __init__(self, class_features: dict[int, np.ndarray]) -> None:
        """Bind the model to one 2-D float32 feature bank per class."""
        if not class_features:
            msg = "class_features must not be empty"
            raise ValueError(msg)
        dims = {arr.ndim for arr in class_features.values()}
        if dims != {_FEATURE_DIM_RANK}:
            msg = "Each class_features array must be 2-D [N_class, F]"
            raise ValueError(msg)
        feat_dims = {arr.shape[1] for arr in class_features.values()}
        if len(feat_dims) != 1:
            msg = f"All class feature arrays must have the same number of features; got {feat_dims}"
            raise ValueError(msg)
        self.class_features: dict[int, np.ndarray] = {
            int(k): v.astype(np.float32) for k, v in class_features.items()
        }
        self._feat_dim: int = next(iter(self.class_features.values())).shape[1]
        self._class_ids: list[int] = sorted(self.class_features.keys())

    @property
    def feat_dim(self) -> int:
        """Number of features per node."""
        return self._feat_dim

    @property
    def n_classes(self) -> int:
        """Number of distinct classes in the fitted model."""
        return len(self._class_ids)

    @property
    def class_ids(self) -> list[int]:
        """Sorted list of class labels present in the model."""
        return list(self._class_ids)

    def sample(
        self,
        class_id: int,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample ``n`` feature vectors for ``class_id`` by bootstrap."""
        mapped = self._class_ids[class_id % self.n_classes]
        pool = self.class_features[mapped]
        indices = rng.integers(0, len(pool), size=n)
        return pool[indices]

    def __repr__(self) -> str:
        """Show per-class bank sizes rather than dumping raw arrays."""
        sizes = {k: v.shape[0] for k, v in self.class_features.items()}
        return (
            f"RealisticFeatureModel(n_classes={self.n_classes}, "
            f"feat_dim={self.feat_dim}, class_sizes={sizes})"
        )


def fit_from_dataset(data: Data) -> RealisticFeatureModel:
    """Fit a :class:`RealisticFeatureModel` from a PyG ``Data`` object."""
    if data.x is None:
        msg = "data.x is None — the dataset must have node features."
        raise ValueError(msg)
    if data.y is None:
        msg = "data.y is None — the dataset must have class labels."
        raise ValueError(msg)

    x_np: np.ndarray = data.x.float().numpy()
    y_np: np.ndarray = data.y.long().numpy()

    class_ids = np.unique(y_np)
    class_features: dict[int, np.ndarray] = {}
    for cid in class_ids:
        rows = x_np[y_np == cid]
        if len(rows) == 0:
            msg = f"Class {cid} has zero samples after grouping."
            raise ValueError(msg)
        class_features[int(cid)] = rows.astype(np.float32)

    return RealisticFeatureModel(class_features)


def _derive_p_in_p_out(
    n_nodes: int,
    n_communities: int,
    homophily: float,
    avg_degree: float,
) -> tuple[float, float]:
    """Derive ``(p_in, p_out)`` matching target homophily and average degree."""
    block_size = n_nodes / n_communities
    p_in = min(1.0, homophily * avg_degree / (block_size - 1.0)) if block_size > 1.0 else 0.0
    p_out = (
        min(1.0, (1.0 - homophily) * avg_degree / ((n_communities - 1) * block_size))
        if n_communities > 1
        else 0.0
    )
    return p_in, p_out


def generate_realistic(  # noqa: PLR0913 - flat SBM + feature-model generation surface
    n_nodes: int,
    n_communities: int,
    homophily: float,
    avg_degree: float,
    feature_model: RealisticFeatureModel,
    *,
    p_in: float | None = None,
    p_out: float | None = None,
    theta_exponent: float | None = None,
    seed: int | None = None,
) -> Data:
    """Generate a PyG ``Data`` graph with SBM/DC-SBM topology and bootstrapped features.

    One seed drives both the topology and the feature bootstrap; ``seed=None``
    draws it, and ``data.seed`` records the seed that ran.
    """
    import torch  # noqa: PLC0415 - optional extra
    from torch_geometric.data import Data  # noqa: PLC0415 - optional extra

    if (p_in is None) != (p_out is None):
        msg = "p_in and p_out must both be provided or both be None."
        raise ValueError(msg)

    if p_in is not None and p_out is not None:
        eff_p_in, eff_p_out = float(p_in), float(p_out)
    else:
        eff_p_in, eff_p_out = _derive_p_in_p_out(
            n_nodes,
            n_communities,
            float(homophily),
            float(avg_degree),
        )

    used = resolve_seed(seed)
    graph = generate_sbm(
        n_nodes,
        n_communities,
        eff_p_in,
        eff_p_out,
        theta_exponent=theta_exponent,
        seed=used,
    )
    labels_np = graph.labels
    if labels_np is None:  # pragma: no cover - generate_sbm always labels its nodes
        msg = "internal: generate_sbm returned no labels"
        raise RuntimeError(msg)

    rng = np.random.default_rng(used)
    feat_rows = [
        feature_model.sample(int(labels_np[node_idx]), 1, rng) for node_idx in range(n_nodes)
    ]
    x_np = np.concatenate(feat_rows, axis=0)

    return Data(
        edge_index=torch.from_numpy(graph.edge_index),
        x=torch.from_numpy(x_np),
        y=torch.from_numpy(labels_np.copy()),
        num_nodes=n_nodes,
        effective_h=graph.metadata["homophily"],
        seed=used,
    )
