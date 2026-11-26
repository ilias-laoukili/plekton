# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ilias Laoukili
"""Shared pytest hooks.

With ``PLEKTON_REQUIRE_EXTRAS=1``, set in the CI job that installs every optional
extra, no test may be skipped. In this suite a skip only ever means that torch,
torch_geometric, networkx or the compiled extension could not be imported, which in
that job means the install is broken. So a skip, whether at collection
(``pytest.importorskip`` at the top of a module) or at setup (a ``skipif`` mark),
becomes a failure that keeps the skip's reason.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

_REQUIRE_EXTRAS = os.environ.get("PLEKTON_REQUIRE_EXTRAS") == "1"


def _fail_if_skipped(report: pytest.TestReport | pytest.CollectReport) -> None:
    """Turn a skip into a failure that names the skip's reason."""
    if not (_REQUIRE_EXTRAS and report.skipped) or hasattr(report, "wasxfail"):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
    report.outcome = "failed"
    report.longrepr = f"skipped although PLEKTON_REQUIRE_EXTRAS=1: {reason}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport() -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Fail a test that a ``skipif`` mark or a ``pytest.skip`` call would skip."""
    report = yield
    _fail_if_skipped(report)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report() -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Fail a module that ``pytest.importorskip`` would skip at collection."""
    report = yield
    _fail_if_skipped(report)
    return report
