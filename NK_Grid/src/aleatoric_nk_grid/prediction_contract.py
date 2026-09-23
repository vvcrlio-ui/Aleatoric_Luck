"""Explicit, immutable opt-in contract for FFC prediction persistence.

Legacy configurations remain disabled.  Scientific model library changes are
never inferred from enabling persistence.
"""
from __future__ import annotations

from collections.abc import Mapping


def normalize_prediction_options(cache=None, execution=None):
    if cache is not None and not isinstance(cache, Mapping):
        raise ValueError("prediction_cache must be a mapping")
    if execution is not None and not isinstance(execution, Mapping):
        raise ValueError("execution must be a mapping")
    options, workflow = dict(cache or {}), dict(execution or {})
    mode = options.get("mode", "off")
    if mode not in {"off", "holdout", "holdout_oof"}:
        raise ValueError("prediction_cache.mode must be off, holdout, or holdout_oof")
    if mode == "off":
        if workflow.get("workflow") == "base_then_sl":
            raise ValueError("base_then_sl requires holdout_oof prediction cache")
        if options.get("required"):
            raise ValueError("required prediction cache cannot be disabled")
        return {}, workflow
    allowed = {"mode", "scope", "base_library", "dtype", "required", "shard_target_mib",
               "store_reported_sl_holdout", "quota_reserve_gb", "file_reserve", "max_bytes",
               "max_files", "temporary_max_bytes", "oof_folds", "trusted_sources", "variants",
               "pipelines", "compression_level", "layout", "compaction_max_bytes"}
    if set(options) - allowed:
        raise ValueError("Unknown prediction cache options: " + ", ".join(sorted(set(options) - allowed)))
    if options.get("trusted_sources") or options.get("pipelines"):
        raise ValueError("Cross-run training reuse and custom pipeline recipes require an explicit importer; "
                         "use the verified-cache offline SL command for sensitivity recombination")
    if options.get("compression_level", 6) != 6:
        raise ValueError("The version-1 cache codec freezes zlib compression level 6")
    result = {"mode": mode, "scope": "all", "dtype": "float64", "required": True,
              "layout": "shared-v1", "compaction_max_bytes": 512 * 1024**2,
              "shard_target_mib": 128, "store_reported_sl_holdout": False,
              "quota_reserve_gb": 500, "file_reserve": 1_000_000, "oof_folds": 5,
              **options}
    if result["dtype"] != "float64" or result["scope"] != "all":
        raise ValueError("Initial prediction cache supports only float64 and explicit scope=all")
    if type(result["required"]) is not bool or not result["required"]:
        raise ValueError("Enabled prediction persistence is required for completion")
    if type(result["store_reported_sl_holdout"]) is not bool:
        raise ValueError("store_reported_sl_holdout must be boolean")
    for name in ("shard_target_mib", "oof_folds", "max_bytes", "max_files", "temporary_max_bytes"):
        if name in result and (type(result[name]) is not int or result[name] < 1):
            raise ValueError(name + " must be a positive integer")
    if result["oof_folds"] < 2:
        raise ValueError("OOF requires at least two folds")
    if result['layout'] not in ('legacy', 'shared-v1'):
        raise ValueError('Unsupported prediction storage layout')
    if type(result['compaction_max_bytes']) is not int or result['compaction_max_bytes'] < 1:
        raise ValueError('compaction_max_bytes must be positive')
    if result["quota_reserve_gb"] < 0 or result["file_reserve"] < 0:
        raise ValueError("Storage reserves cannot be negative")
    if not result.get("base_library"):
        raise ValueError("Enabled cache requires an explicit frozen base_library identity")
    if workflow.get("workflow") == "base_then_sl":
        if mode != "holdout_oof":
            raise ValueError("base_then_sl requires OOF and holdout predictions")
        if workflow.get("barrier_scope") != "submission_plan" or workflow.get("protocol_version") != 2:
            raise ValueError("base_then_sl requires submission_plan barrier and protocol_version=2")
        if workflow.get('verification_schedule', 'before_sl') not in ('before_sl', 'final_only'):
            raise ValueError('verification_schedule must be before_sl or final_only')
    elif workflow.get("workflow") not in {None, "capture"}:
        raise ValueError("Unknown prediction workflow")
    return result, workflow


def prediction_cache_enabled(config):
    # A configuration predating these fields cannot have enabled persistence,
    # so absence reads as disabled rather than crashing this legacy guard.
    return bool(normalize_prediction_options(getattr(config, "prediction_cache", None),
                                             getattr(config, "execution", None))[0])


def reject_unsupported_prediction_backend(config, backend):
    if prediction_cache_enabled(config):
        raise ValueError(f"{backend} does not implement the required prediction cache contract; "
                         "use the protocol v2 base_then_sl controller")
