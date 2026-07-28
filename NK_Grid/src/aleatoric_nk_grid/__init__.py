"""Shared, article-agnostic N×K grid engine."""

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

_PACKAGE_PATH = Path(__file__).resolve()
_PACKAGE_DIR = _PACKAGE_PATH.parent
_SOURCE_LAYOUT = (
    _PACKAGE_PATH.parents[1].name == "src"
    and (_PACKAGE_PATH.parents[2] / "pyproject.toml").is_file()
)
_INSTALLED_LAYOUT = False
if not _SOURCE_LAYOUT:
    try:
        _DIST = distribution("aleatoric-nk-grid")
    except PackageNotFoundError:
        _DIST = None
    if _DIST is not None:
        _INSTALLED_LAYOUT = (
            Path(_DIST.locate_file("aleatoric_nk_grid")).resolve() == _PACKAGE_DIR
        )
if _PACKAGE_DIR.name != "aleatoric_nk_grid" or not (
    _SOURCE_LAYOUT or _INSTALLED_LAYOUT
):
    raise RuntimeError(
        f"aleatoric_nk_grid resolved outside the installed shared engine: "
        f"{_PACKAGE_PATH}"
    )

from .ingest import InputSchema, LoadedInput, load_input, load_schema
from .nk_grid import NKGridConfig, run_nk_grid

__all__ = [
    "InputSchema",
    "LoadedInput",
    "NKGridConfig",
    "load_input",
    "load_schema",
    "run_nk_grid",
]

__version__ = "1.0.0"
