# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Tests for ``plekton.distributions`` — bootstrap feature sampling.

The ``RealisticFeatureModel`` and ``fit_from_dataset`` tests are pure
numpy/torch — no compiled Rust extension is needed.
``test_generate_realistic_*`` requires the extension and is guarded
with a ``requires_plekton`` skip mark.

Run with:  python -m pytest python/tests/test_distributions.py -v
"""

from __future__ import annotations

import re

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from plekton.distributions import (
    RealisticFeatureModel,
    fit_from_dataset,
    generate_realistic,
)
from torch_geometric.data import Data

# The fixture bank carries two classes.
_N_CLASSES = 2
# Rows in class 1.
_N_CLASS_1_ROWS = 40
# Rows in class 0.
_N_CLASS_0_ROWS = 60

try:
    import plekton as _rs  # noqa: F401 — availability probe for the skip mark

    _HAS_PLEKTON = True
except ImportError:
    _HAS_PLEKTON = False

requires_plekton = pytest.mark.skipif(
    not _HAS_PLEKTON,
    reason="plekton Rust extension not installed (run `maturin develop`)",
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

FEAT_DIM = 8  # small feature dimension for fast tests


def _make_model(n_classes: int = 2, n_samples_per_class: int = 20) -> RealisticFeatureModel:
    """Build a ``RealisticFeatureModel`` from synthetic random arrays."""
    rng = np.random.default_rng(0)
    class_features = {
        class_id: rng.random((n_samples_per_class, FEAT_DIM), dtype=np.float32)
        for class_id in range(n_classes)
    }
    return RealisticFeatureModel(class_features)


def _make_pyg_data(n_nodes: int = 100) -> Data:
    """Build a minimal PyG ``Data`` object with random features and binary labels."""
    rng = np.random.default_rng(42)
    x = torch.from_numpy(rng.random((n_nodes, FEAT_DIM), dtype=np.float32))
    y = torch.cat([torch.zeros(60, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index, num_nodes=n_nodes)


# ─── RealisticFeatureModel ────────────────────────────────────────────────────


def test_realistic_feature_model_rejects_empty_dict() -> None:
    """Constructing a model from an empty dict must raise ``ValueError``."""
    with pytest.raises(ValueError, match="must not be empty"):
        RealisticFeatureModel({})


def test_realistic_feature_model_rejects_inconsistent_feat_dim() -> None:
    """Arrays with different numbers of features must raise ``ValueError``."""
    rng = np.random.default_rng(0)
    class_features = {
        0: rng.random((10, 8), dtype=np.float32),
        1: rng.random((10, 16), dtype=np.float32),  # different width
    }
    with pytest.raises(ValueError, match="same number of features"):
        RealisticFeatureModel(class_features)


def test_sample_returns_correct_shape() -> None:
    """``sample(class_id, n, rng)`` must return an array of shape ``(n, feat_dim)``."""
    model = _make_model()
    rng = np.random.default_rng(1)
    n_samples = 10
    result = model.sample(class_id=0, n=n_samples, rng=rng)
    assert result.shape == (n_samples, FEAT_DIM), (
        f"expected shape ({n_samples}, {FEAT_DIM}), got {result.shape}"
    )
    assert result.dtype == np.float32, f"expected float32, got {result.dtype}"


def test_sample_unknown_class_wraps_modulo() -> None:
    """A ``class_id`` beyond the known range wraps via ``% n_classes`` without error."""
    model = _make_model(n_classes=2)
    rng = np.random.default_rng(2)
    # class_id=5 wraps to 5 % 2 = 1, which is valid.
    result = model.sample(class_id=5, n=4, rng=rng)
    assert result.shape == (4, FEAT_DIM)


# ─── fit_from_dataset ────────────────────────────────────────────────────────


def test_fit_from_dataset_groups_by_label() -> None:
    """Class sizes in the fitted model must equal the label counts in the data."""
    data = _make_pyg_data(n_nodes=100)  # 60 label-0, 40 label-1
    model = fit_from_dataset(data)

    assert model.n_classes == _N_CLASSES, f"expected _N_CLASSES classes, got {model.n_classes}"
    assert model.class_features[0].shape[0] == _N_CLASS_0_ROWS, (
        f"class 0 should have 60 samples, got {model.class_features[0].shape[0]}"
    )
    assert model.class_features[1].shape[0] == _N_CLASS_1_ROWS, (
        f"class 1 should have 40 samples, got {model.class_features[1].shape[0]}"
    )


def test_fit_from_dataset_missing_x_raises() -> None:
    """Fitting from a ``Data`` object without ``x`` must raise ``ValueError``."""
    data = Data(
        x=None,
        y=torch.zeros(10, dtype=torch.long),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=10,
    )
    with pytest.raises(ValueError, match=re.escape("data.x is None")):
        fit_from_dataset(data)


# ─── generate_realistic ──────────────────────────────────────────────────────


@requires_plekton
def test_generate_realistic_shape_and_reproducibility() -> None:
    """Same seed must produce identical feature matrices; x.shape must be [N, feat_dim]."""
    data = _make_pyg_data(n_nodes=100)
    model = fit_from_dataset(data)

    n_nodes = 50
    result_a = generate_realistic(
        n_nodes=n_nodes,
        n_communities=2,
        homophily=0.7,
        avg_degree=6.0,
        feature_model=model,
        seed=99,
    )
    result_b = generate_realistic(
        n_nodes=n_nodes,
        n_communities=2,
        homophily=0.7,
        avg_degree=6.0,
        feature_model=model,
        seed=99,
    )

    assert result_a.x is not None, "feature matrix must not be None"
    assert result_a.x.shape == (n_nodes, FEAT_DIM), (
        f"expected x.shape ({n_nodes}, {FEAT_DIM}), got {result_a.x.shape}"
    )
    assert torch.allclose(result_a.x, result_b.x), (
        "same seed must produce identical feature matrices"
    )
