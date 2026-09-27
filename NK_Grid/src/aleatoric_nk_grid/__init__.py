"""Shared, article-agnostic N×K grid engine."""

from importlib import import_module
from pathlib import Path

# Every process imports the engine from its own checkout through PYTHONPATH.
_PACKAGE_PATH = Path(__file__).resolve()
if (
    _PACKAGE_PATH.parent.name != "aleatoric_nk_grid"
    or _PACKAGE_PATH.parents[1].name != "src"
    or not (_PACKAGE_PATH.parents[2] / "pyproject.toml").is_file()
):
    raise RuntimeError(
        f"aleatoric_nk_grid resolved outside a checkout's NK_Grid/src: {_PACKAGE_PATH}"
    )

__all__ = [
    "InputSchema",
    "LoadedInput",
    "NKGridConfig",
    "load_input",
    "load_schema",
]

_EXPORT_MODULES = {
    "InputSchema": ".ingest",
    "LoadedInput": ".ingest",
    "NKGridConfig": ".config",
    "load_input": ".ingest",
    "load_schema": ".ingest",
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
