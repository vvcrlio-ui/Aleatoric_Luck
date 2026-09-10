"""Experiment configuration and snapshot codecs, independent of execution backends."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grid_contract import validate_size_grid


DEFAULT_MODEL_PARAMS_PATH = Path(__file__).resolve().parents[2] / "model_params.yaml"


@dataclass(frozen=True)
class NKGridConfig:
    schema: Path
    out: Path
    outcome: str
    models: tuple[str, ...]
    seed: int
    test_size: float
    n_seeds: int
    n_draws: int
    n_sizes_n: int
    n_sizes_k: int
    max_n: int
    max_k: int
    batch_size: int
    n_jobs: int
    min_n: int = 10
    model_params: Path = DEFAULT_MODEL_PARAMS_PATH
    failed_abs_threshold: int = 50
    failed_ratio_threshold: float = 0.05
    native_process_max_attempts: int = 2
    native_process_timeout_seconds: float = 21_600.0
    preset: str | None = None
    allow_large_run: bool = False
    dry_run: bool = False
    rerun_completed: bool = True
    # Direct construction is used by the test/dev API. Production manifests
    # always override these explicit values.
    experiment_id: str = "nkgrid-test-v1"
    data_version: str = "test-data-v1"
    model_spec_version: str = "nkgrid-test-models-v1"
    repeat_plan: tuple[tuple[int, int], ...] | None = None
    n_grid: tuple[int, ...] | None = None
    k_grid: tuple[int, ...] | None = None
    prediction_export_cells: tuple[tuple[str, int, int], ...] = ()
    # Compatibility default: local prunes verified-complete parts; dynamic keeps WAL.
    checkpoint_retention: str = "default"
    # Applied to the full source grid; frozen three-point grids stay unchanged.
    grid_selection: str = "all"


def resolve_repeat_pairs(config: NKGridConfig) -> tuple[tuple[int, int], ...]:
    """Resolve legacy counts or explicit absolute pairs into one representation."""

    if config.repeat_plan is not None:
        if config.n_seeds != 1 or config.n_draws != 1:
            raise ValueError("repeat_plan cannot be combined with n_seeds or n_draws")
        pairs = tuple((seed, draw) for seed, draw in config.repeat_plan)
    else:
        pairs = tuple(
            (config.seed + offset, draw)
            for offset in range(config.n_seeds)
            for draw in range(config.n_draws)
        )
    group_repeat_pairs_by_seed(pairs)
    return tuple(sorted(pairs))


def group_repeat_pairs_by_seed(
    repeat_pairs: Sequence[tuple[int, int]],
) -> dict[int, tuple[int, ...]]:
    """Validate and group absolute repeat pairs without silently deduplicating."""

    grouped: dict[int, list[int]] = {}
    seen: set[tuple[int, int]] = set()
    for pair in repeat_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("repeat_plan entries must be (seed, draw) pairs")
        seed, draw = pair
        if isinstance(seed, bool) or isinstance(draw, bool) or not isinstance(seed, int) or not isinstance(draw, int) or seed < 0 or draw < 0:
            raise ValueError("repeat_plan seed and draw must be non-negative integers")
        if (seed, draw) in seen:
            raise ValueError(f"repeat_plan contains duplicate pair ({seed}, {draw})")
        seen.add((seed, draw))
        grouped.setdefault(seed, []).append(draw)
    if not grouped:
        raise ValueError("repeat_plan must not be empty")
    return {seed: tuple(sorted(draws)) for seed, draws in sorted(grouped.items())}


def execution_groups_for_models(models: Sequence[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the ordered preprocessing groups used by the task-table codec."""

    selected = tuple(str(model) for model in models)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("models must be non-empty and unique")
    passthrough = tuple(model for model in selected if model in {"lightgbm", "xgboost"})
    imputed = tuple(model for model in selected if model not in passthrough)
    return tuple(
        (name, group)
        for name, group in (("imputed_core", imputed), ("passthrough", passthrough))
        if group
    )


def config_to_json(config: NKGridConfig) -> dict[str, Any]:
    """Encode paths and nested tuple fields as JSON-native values."""

    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        return value

    return {
        field.name: encode(getattr(config, field.name))
        for field in sorted(fields(config), key=lambda field: field.name)
    }


def config_from_json(
    payload: Mapping[str, Any], *, strict: bool = False,
) -> NKGridConfig:
    """Restore configuration containers from a snapshot.

    ``strict`` preserves the dynamic contract reader's existing grid validation
    and scalar normalization. Static seed snapshots retain their existing
    validation boundary in the engine. Model-parameter validation is unchanged.
    """

    values = dict(payload)
    for key in ("schema", "out", "model_params"):
        if strict:
            values[key] = Path(str(values[key]))
        elif values.get(key) is not None:
            values[key] = Path(values[key])
    values["models"] = tuple(str(value) for value in values["models"]) if strict else tuple(values["models"])
    for key in ("n_grid", "k_grid"):
        if values.get(key) is not None:
            values[key] = validate_size_grid(values[key], key) if strict else tuple(values[key])
    if values.get("repeat_plan") is not None:
        values["repeat_plan"] = tuple(
            (int(pair[0]), int(pair[1])) if strict else tuple(pair)
            for pair in values["repeat_plan"]
        )
    if values.get("prediction_export_cells") is not None:
        values["prediction_export_cells"] = tuple(
            (str(cell[0]), int(cell[1]), int(cell[2])) if strict else tuple(cell)
            for cell in values["prediction_export_cells"]
        )
    return NKGridConfig(**values)
