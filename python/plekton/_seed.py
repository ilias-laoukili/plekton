# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""One seed convention for every randomised function in ``plekton``.

``seed=None`` means "draw one". The draw happens here, in Python, so that the
value handed to Rust is always explicit and the seed a function records — for
instance in ``Graph.metadata["seed"]`` — is exactly the seed that ran.
"""

from __future__ import annotations

import operator

import numpy as np

__all__ = ["resolve_seed"]

# Rust takes the seed as a u64.
_U64_MAX = (1 << 64) - 1


def resolve_seed(seed: int | None) -> int:
    """Return the seed a randomised call actually uses.

    ``None`` draws fresh OS entropy through :class:`numpy.random.SeedSequence`
    and keeps its low 64 bits. An explicit seed must be an integer in
    ``[0, 2**64)``; a ``bool`` or a float is refused rather than coerced.
    """
    if seed is None:
        entropy = np.random.SeedSequence().entropy
        if not isinstance(entropy, int):  # pragma: no cover - a fresh SeedSequence draws an int
            msg = f"numpy SeedSequence drew non-integer entropy {entropy!r}"
            raise TypeError(msg)
        return entropy & _U64_MAX
    if isinstance(seed, bool):
        msg = "seed must be an integer or None, not a bool"
        raise TypeError(msg)
    value = operator.index(seed)
    if not 0 <= value <= _U64_MAX:
        msg = f"seed must be in [0, 2**64), got {value}"
        raise ValueError(msg)
    return value
