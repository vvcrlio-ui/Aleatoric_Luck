"""Experiment configuration, independent of execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


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
    n_jobs: int
    min_n: int = 10
    model_params: Path = DEFAULT_MODEL_PARAMS_PATH
    native_process_max_attempts: int = 2
    native_process_timeout_seconds: float = 21_600.0
    preset: str | None = None
    allow_large_run: bool = False
    # Direct construction is used by the test/dev API. Production manifests
    # always override these explicit values.
    experiment_id: str = "nkgrid-test-v1"
    data_version: str = "test-data-v1"
    model_spec_version: str = "nkgrid-test-models-v1"
    repeat_plan: tuple[tuple[int, int], ...] | None = None
    n_grid: tuple[int, ...] | None = None
    k_grid: tuple[int, ...] | None = None
    checkpoint_retention: str = "keep"
    # Applied to the full source grid; frozen three-point grids stay unchanged.
    grid_selection: str = "all"
    prediction_cache: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None

    def __post_init__(self):
        from .prediction_contract import normalize_prediction_options
        normalize_prediction_options(self.prediction_cache, self.execution)


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
    """Return the ordered preprocessing groups shared by the models of one cell."""

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
