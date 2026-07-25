"""Keep legacy fork test modules isolated from one another during root runs."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
LEGACY_TEST_ROOTS = (
    (REPO_ROOT / "SMR" / "tests").resolve(),
    (REPO_ROOT / "FFC" / "NK_Grid" / "tests").resolve(),
    (REPO_ROOT / "FFC" / "NK_Grid" / "data_processor" / "tests").resolve(),
)


def pytest_ignore_collect(collection_path, config) -> bool:
    """Legacy suites are executed in isolated subprocesses by the shared tests."""

    path = Path(str(collection_path)).resolve()
    return any(path == root or root in path.parents for root in LEGACY_TEST_ROOTS)
