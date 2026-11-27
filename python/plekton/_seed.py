# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""One seed convention for every randomised function in ``plekton``.

``seed=None`` means "draw one". The draw happens here, in Python, so that the
value handed to Rust is always explicit and the seed a function records — for
instance in ``Graph.metadata["seed"]`` — is exactly the seed that ran.
"""

from __future__ import annotations

import hashlib
import json
import operator
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["derive_seed", "resolve_seed"]

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


def derive_seed(base_seed: int, config: Mapping[str, Any], replica: int) -> int:
    """Derive the seed of one graph from a base seed, its configuration and its replica.

    The seed is the first 8 bytes, read big-endian, of the SHA-256 of the canonical
    JSON of ``[base_seed, config, replica]`` — ``config`` without its ``"seed"``
    key, keys sorted, no whitespace, ASCII only — so it lies in ``[0, 2**64)``. It
    depends on that entry and nothing else: adding a value to a sweep grid changes
    no other entry's seed, and the same entry gets the same seed in every process,
    since the hash is a fixed function of those bytes (unlike Python's ``hash()``,
    which changes from one process to the next). The JSON value is what counts:
    ``4`` and ``4.0`` are different configurations and give different seeds. A NaN
    or an infinity has no JSON form and raises :class:`ValueError`.
    """
    base = resolve_seed(base_seed)
    if isinstance(replica, bool):
        msg = "replica must be an integer, not a bool"
        raise TypeError(msg)
    index = operator.index(replica)
    if index < 0:
        msg = f"replica must be >= 0, got {index}"
        raise ValueError(msg)
    entry = {key: value for key, value in config.items() if key != "seed"}
    payload = json.dumps(
        [base, entry, index],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
