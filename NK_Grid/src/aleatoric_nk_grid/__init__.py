"""Shared, article-agnostic N×K grid engine."""

from importlib import import_module
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

__all__ = [
    "InputSchema",
    "LoadedInput",
    "NKGridConfig",
    "load_input",
    "load_schema",
    "run_nk_grid",
]

_EXPORT_MODULES = {
    "InputSchema": ".ingest",
    "LoadedInput": ".ingest",
    "NKGridConfig": ".config",
    "load_input": ".ingest",
    "load_schema": ".ingest",
    "run_nk_grid": ".nk_grid",
}


def __getattr__(name: str):
    """Load execution dependencies only when an execution API is requested."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__version__ = "1.0.0"
