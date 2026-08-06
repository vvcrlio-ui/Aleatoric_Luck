"""Stage B planning: budget, grouping threshold, resource requests, acceptance.

Every decision here is a *derivation* from a calibration artifact plus values
the caller states explicitly (cluster partition, wall-clock limit, safety
factor, parallel efficiency).  Nothing is defaulted into existence: if an input
is missing the function raises and says which measurement is required.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .calibrated_cost_model import CalibrationUnusable, CalibratedCostModel
from .flat_task_table import (
    ResourceRequest,
    TaskRow,
    build_rows,
    pack_lpt,
    resource_class_for_rows,
    sbatch_resource_args,
    write_chunk_snapshot,
    write_task_table,
)
from .nk_grid import NKGridConfig


#: Candidate chunk wall-clock targets from ``flat-task-table`` section 3.
DEFAULT_T_TARGET_CANDIDATES_SECONDS: tuple[int, ...] = (30 * 60, 60 * 60)

#: ``flat-task-table`` acceptance 6: fixed startup must stay a rounding error.
MAX_T0_SHARE = 0.05

#: ``flat-task-table`` acceptance 5: packing quality, excluding rows that are
#: individually larger than the budget and therefore have no neighbour.
MAX_CHUNK_P95_OVER_P50 = 3.0


# ---------------------------------------------------------------------------
# B2: chunk budget
# ---------------------------------------------------------------------------


def choose_t_target_seconds(
    model: CalibratedCostModel,
    *,
    candidates: Sequence[int] = DEFAULT_T_TARGET_CANDIDATES_SECONDS,
    max_t0_share: float = MAX_T0_SHARE,
) -> int:
    """Smallest candidate chunk length whose fixed startup stays under budget.

    Shorter chunks are strictly better for preemption and queue turnaround, so
    the rule is "take the shortest one that still amortises t0", not "take the
    longest one that fits".
    """

    if not candidates:
        raise ValueError("candidates must not be empty")
    if not 0 < max_t0_share < 1:
        raise ValueError("max_t0_share must be in (0, 1)")
    for candidate in sorted(int(value) for value in candidates):
        if candidate <= 0:
            raise ValueError("T_target candidates must be positive")
        if model.t0_total_seconds / candidate < max_t0_share:
            return candidate
    raise CalibrationUnusable(
        f"t0 = {model.t0_total_seconds:.2f}s exceeds {max_t0_share:.0%} of every candidate "
        f"chunk length {tuple(candidates)}; either shorten startup or raise T_target"
    )


def t0_share(model: CalibratedCostModel, packed_rows: Sequence[TaskRow]) -> float:
    """Fixed startup as a fraction of the median chunk's estimated duration."""

    durations = _chunk_costs(packed_rows)
    if not durations:
        raise ValueError("packed_rows must contain at least one chunk")
    median = statistics.median(durations)
    if median <= 0:
        raise ValueError("median chunk cost must be positive")
    return float(model.t0_total_seconds / median)


# ---------------------------------------------------------------------------
# B3: super_learner grouping threshold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitTradeoff:
    """Both sides of the "split super_learner out" decision at one K."""

    k_features: int
    n_samples: int
    extra_core_seconds: float
    merged_serial_peak_bytes: float
    split_serial_peak_bytes: float

    @property
    def memory_reduction_bytes(self) -> float:
        return max(0.0, self.merged_serial_peak_bytes - self.split_serial_peak_bytes)


def split_super_learner_tradeoff(
    model: CalibratedCostModel,
    *,
    n_samples: int,
    k_features: int,
    core_models: Sequence[str],
    super_learner_cpus: int,
    parallel_efficiency: float | None = None,
) -> SplitTradeoff:
    """Quantify what splitting Super Learner into its own group costs and buys.

    Cost side (core-seconds): the cell pays a second ``imputed`` preprocessing
    pass, and -- if the split class requests more than one CPU -- Super Learner
    additionally burns ``F/eta`` core-seconds instead of ``F``.

    Benefit side: the *serial* class no longer has to size ``--mem`` for Super
    Learner, which is the real reason to split.  On pure core-seconds splitting
    can never win, so a threshold justified on time alone would be wrong.
    """

    if super_learner_cpus < 1:
        raise ValueError("super_learner_cpus must be >= 1")
    mode_cost = model.preprocess_cost.get("imputed")
    if mode_cost is None:
        raise CalibrationUnusable("calibration has no preprocess_cost for mode 'imputed'")
    extra = mode_cost.predict(n_samples, k_features)

    if super_learner_cpus > 1:
        if parallel_efficiency is None:
            raise CalibrationUnusable(
                "splitting onto more than one CPU changes core-seconds by 1/eta, and the "
                "parallel efficiency eta has not been measured yet (cost-calibration item 3); "
                "pass parallel_efficiency explicitly or plan the split at one CPU"
            )
        if not 0 < parallel_efficiency <= 1:
            raise ValueError("parallel_efficiency must be in (0, 1]")
        fit = model.fit_cost.get("super_learner")
        if fit is None:
            raise CalibrationUnusable("calibration has no fit_cost for 'super_learner'")
        super_seconds = fit.predict(n_samples, k_features)
        extra += super_seconds * (1.0 / parallel_efficiency - 1.0)

    model.require_trustworthy_memory()
    merged_models = tuple(core_models) + ("super_learner",)
    merged_peak = max(
        _peak_for_model(model, name, n_samples, k_features) for name in merged_models
    )
    split_peak = max(
        _peak_for_model(model, name, n_samples, k_features) for name in core_models
    )
    return SplitTradeoff(
        k_features=int(k_features),
        n_samples=int(n_samples),
        extra_core_seconds=float(extra),
        merged_serial_peak_bytes=float(merged_peak),
        split_serial_peak_bytes=float(split_peak),
    )


def split_super_learner_min_k(
    model: CalibratedCostModel,
    *,
    n_samples: int,
    k_grid: Sequence[int],
    core_models: Sequence[str],
    super_learner_cpus: int,
    min_memory_reduction_bytes: int,
    parallel_efficiency: float | None = None,
) -> int | None:
    """Smallest K at which splitting frees enough memory to be worth its cost.

    ``min_memory_reduction_bytes`` is a cluster policy input (for example one
    memory-per-CPU allocation unit), not something this module can infer.
    Returns ``None`` when no K in the grid clears the bar -- that is a valid,
    explicit answer meaning "never split".
    """

    if min_memory_reduction_bytes <= 0:
        raise ValueError("min_memory_reduction_bytes must be positive")
    for k_features in sorted({int(value) for value in k_grid}):
        tradeoff = split_super_learner_tradeoff(
            model,
            n_samples=n_samples,
            k_features=k_features,
            core_models=core_models,
            super_learner_cpus=super_learner_cpus,
            parallel_efficiency=parallel_efficiency,
        )
        if tradeoff.memory_reduction_bytes >= min_memory_reduction_bytes:
            return k_features
    return None


def _peak_for_model(model: CalibratedCostModel, name: str, n_samples: int, k_features: int) -> float:
    fit = model.peak_rss_bytes.get(name)
    if fit is None:
        raise CalibrationUnusable(f"calibration has no peak_rss_bytes fit for model {name!r}")
    return fit.predict(n_samples, k_features)


# ---------------------------------------------------------------------------
# B4: resource requests
# ---------------------------------------------------------------------------


def format_slurm_memory(num_bytes: float) -> str:
    """Round bytes up to whole mebibytes, the smallest unit Slurm accepts."""

    if num_bytes <= 0:
        raise ValueError("memory must be positive")
    mebibytes = int(math.ceil(num_bytes / (1024 ** 2)))
    if mebibytes % 1024 == 0:
        return f"{mebibytes // 1024}G"
    return f"{mebibytes}M"


def resource_requests(
    packed_rows: Sequence[TaskRow],
    model: CalibratedCostModel,
    *,
    cpus_per_task: Mapping[str, int],
    partition: Mapping[str, str],
    time_limit: Mapping[str, str],
    memory_safety_factor: float,
) -> dict[str, ResourceRequest]:
    """Build one fully specified request per resource class actually present.

    ``--mem`` comes from the fitted per-cell peak, taken as a maximum over the
    chunks in the class because chunks run their rows sequentially.  A class
    that asks for more than one CPU is additionally floored at the measured
    concurrent task peak: a single-cell fit cannot describe what several
    workers do to one node's memory.
    """

    if memory_safety_factor < 1:
        raise ValueError("memory_safety_factor must be >= 1")
    model.require_trustworthy_memory()

    by_class: dict[str, list[TaskRow]] = {}
    for chunk_rows in _chunks(packed_rows).values():
        by_class.setdefault(resource_class_for_rows(chunk_rows), []).extend(chunk_rows)
    if not by_class:
        raise ValueError("packed_rows must contain at least one chunk")

    requests: dict[str, ResourceRequest] = {}
    for resource_class, rows in sorted(by_class.items()):
        for name, mapping in (
            ("cpus_per_task", cpus_per_task),
            ("partition", partition),
            ("time_limit", time_limit),
        ):
            if resource_class not in mapping:
                raise ValueError(
                    f"no {name} supplied for resource class {resource_class!r}; "
                    "Stage B requires an explicit cluster value for every class in use"
                )
        cpus = int(cpus_per_task[resource_class])
        if cpus < 1:
            raise ValueError("cpus_per_task must be >= 1")
        peak = max(model.row_peak_rss_bytes(row) for row in rows) * float(memory_safety_factor)
        if cpus > 1:
            if model.task_peak_rss_status != "measured" or model.task_peak_rss_bytes is None:
                raise CalibrationUnusable(
                    f"resource class {resource_class!r} requests {cpus} CPUs, but the "
                    "concurrent task memory peak was never measured "
                    f"(task_cgroup_peak_n_jobs_8.status={model.task_peak_rss_status!r}); "
                    "--mem cannot be derived from single-cell fits alone"
                )
            peak = max(peak, float(model.task_peak_rss_bytes) * float(memory_safety_factor))
        requests[resource_class] = ResourceRequest(
            resource_class=resource_class,
            cpus_per_task=cpus,
            partition=str(partition[resource_class]),
            memory=format_slurm_memory(peak),
            time_limit=str(time_limit[resource_class]),
        )
    return requests


# ---------------------------------------------------------------------------
# Packing by resource class
# ---------------------------------------------------------------------------


def pack_by_resource_class(
    rows: Sequence[TaskRow],
    *,
    budget: float,
    cost_function,
) -> tuple[tuple[TaskRow, ...], dict[str, tuple[int, int]]]:
    """Pack each resource class separately and number its chunks contiguously.

    Packing everything together is unsafe: ``resource_class_for_rows`` only
    reports ``super_learner`` when *every* row in the chunk is Super Learner,
    so one mixed chunk silently demotes Super Learner to the single-CPU serial
    class.  Contiguous per-class numbering additionally lets each class be
    submitted as one ``--array=a-b`` range instead of an index list.

    Returns the packed rows and, per class, the inclusive chunk-id range.
    """

    by_class: dict[str, list[TaskRow]] = {}
    for row in rows:
        by_class.setdefault(resource_class_for_rows([row]), []).append(row)

    packed: list[TaskRow] = []
    ranges: dict[str, tuple[int, int]] = {}
    next_chunk_id = 0
    for resource_class, class_rows in sorted(by_class.items()):
        class_packed = pack_lpt(class_rows, budget=budget, cost_function=cost_function)
        local_ids = sorted({row.chunk_id for row in class_packed})
        remap = {local: next_chunk_id + offset for offset, local in enumerate(local_ids)}
        packed.extend(replace(row, chunk_id=remap[row.chunk_id]) for row in class_packed)
        ranges[resource_class] = (next_chunk_id, next_chunk_id + len(local_ids) - 1)
        next_chunk_id += len(local_ids)
    packed.sort(key=lambda row: (row.chunk_id, row.row_id))
    return tuple(packed), ranges


# ---------------------------------------------------------------------------
# B5: acceptance reports
# ---------------------------------------------------------------------------


def _chunks(rows: Sequence[TaskRow]) -> dict[int, list[TaskRow]]:
    grouped: dict[int, list[TaskRow]] = {}
    for row in rows:
        if row.chunk_id < 0:
            raise ValueError("rows must be packed before they are reported on")
        grouped.setdefault(int(row.chunk_id), []).append(row)
    return grouped


def _chunk_costs(rows: Sequence[TaskRow]) -> list[float]:
    return [sum(row.est_cost for row in chunk) for chunk in _chunks(rows).values()]


def packing_report(packed_rows: Sequence[TaskRow], *, budget: float) -> dict[str, object]:
    """Chunk duration spread, with over-budget singleton chunks separated out.

    A row whose own cost exceeds the budget cannot be packed with anything, so
    including it would make the spread look bad for a reason the packer cannot
    fix.  It is reported separately instead of being hidden.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    grouped = _chunks(packed_rows)
    oversized = {
        chunk_id: chunk
        for chunk_id, chunk in grouped.items()
        if len(chunk) == 1 and chunk[0].est_cost > budget
    }
    packable = [
        sum(row.est_cost for row in chunk)
        for chunk_id, chunk in grouped.items()
        if chunk_id not in oversized
    ]
    report: dict[str, object] = {
        "chunk_count": len(grouped),
        "oversized_chunk_count": len(oversized),
        "oversized_chunk_costs": sorted(
            (float(chunk[0].est_cost) for chunk in oversized.values()), reverse=True
        ),
        "budget": float(budget),
    }
    if packable:
        packable.sort()
        p50 = statistics.median(packable)
        p95 = packable[min(len(packable) - 1, int(math.ceil(0.95 * len(packable)) - 1))]
        report.update(
            {
                "p50_chunk_cost": float(p50),
                "p95_chunk_cost": float(p95),
                "p95_over_p50": float(p95 / p50) if p50 > 0 else math.inf,
                "meets_acceptance": bool(p50 > 0 and p95 / p50 < MAX_CHUNK_P95_OVER_P50),
            }
        )
    else:
        report.update(
            {"p50_chunk_cost": None, "p95_chunk_cost": None, "p95_over_p50": None, "meets_acceptance": False}
        )
    return report


def compare_budgets(
    rows: Sequence[TaskRow],
    model: CalibratedCostModel,
    *,
    budgets: Sequence[float],
) -> list[dict[str, object]]:
    """Pack the same design at several chunk budgets for acceptance 9."""

    cost_function = model.cost_function()
    comparison: list[dict[str, object]] = []
    for budget in budgets:
        packed, _ = pack_by_resource_class(rows, budget=float(budget), cost_function=cost_function)
        report = packing_report(packed, budget=float(budget))
        report["t0_share"] = t0_share(model, packed)
        report["meets_t0_acceptance"] = bool(report["t0_share"] < MAX_T0_SHARE)
        comparison.append(report)
    return comparison


# ---------------------------------------------------------------------------
# End-to-end plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterPolicy:
    """Machine-specific values Stage B must be told, never infer."""

    cpus_per_task: Mapping[str, int]
    partition: Mapping[str, str]
    time_limit: Mapping[str, str]
    memory_safety_factor: float
    max_concurrent: Mapping[str, int]


def build_calibrated_plan(
    config: NKGridConfig,
    *,
    calibration_path: Path | str,
    n_grid: Sequence[int],
    k_grid: Sequence[int],
    cluster: ClusterPolicy,
    table_path: Path | str,
    snapshot_path: Path | str,
    output_dir: Path | str,
    panel: str,
    split_super_learner_min_k: int | None = None,
    t_target_seconds: int | None = None,
    max_directory_entries: int | None = None,
    strict: bool = True,
) -> dict[str, object]:
    """Derive the whole submission plan from one calibration artifact.

    ``strict`` refuses to emit a plan that fails its own acceptance criteria.
    That is deliberate: an unbalanced or startup-dominated plan is exactly the
    thing this design exists to prevent, and discovering it after a week of
    queue time is not a discovery.
    """

    model = CalibratedCostModel.from_json(calibration_path)
    budget = int(t_target_seconds) if t_target_seconds is not None else choose_t_target_seconds(model)
    rows = build_rows(
        config,
        n_grid=n_grid,
        k_grid=k_grid,
        split_super_learner_min_k=split_super_learner_min_k,
    )
    packed, ranges = pack_by_resource_class(
        rows, budget=float(budget), cost_function=model.cost_function()
    )
    report = packing_report(packed, budget=float(budget))
    report["t0_share"] = t0_share(model, packed)
    report["meets_t0_acceptance"] = bool(report["t0_share"] < MAX_T0_SHARE)

    requests = resource_requests(
        packed,
        model,
        cpus_per_task=cluster.cpus_per_task,
        partition=cluster.partition,
        time_limit=cluster.time_limit,
        memory_safety_factor=cluster.memory_safety_factor,
    )
    if strict and not (report["meets_acceptance"] and report["meets_t0_acceptance"]):
        raise CalibrationUnusable(
            "packing plan fails its acceptance criteria "
            f"(p95/p50={report.get('p95_over_p50')}, t0_share={report.get('t0_share')}); "
            "pass strict=False to inspect the failing plan"
        )

    written_table = write_task_table(Path(table_path), packed)
    written_snapshot = write_chunk_snapshot(
        Path(snapshot_path),
        table_path=written_table,
        panel=panel,
        config=config,
        output_dir=Path(output_dir),
    )

    submissions = []
    for resource_class, (first, last) in sorted(ranges.items()):
        if resource_class not in cluster.max_concurrent:
            raise ValueError(f"no max_concurrent supplied for resource class {resource_class!r}")
        class_rows = [row for row in packed if first <= row.chunk_id <= last]
        submissions.append(
            {
                "resource_class": resource_class,
                "chunk_first": first,
                "chunk_last": last,
                "array": f"{first}-{last}%{int(cluster.max_concurrent[resource_class])}",
                "sbatch_args": list(sbatch_resource_args(class_rows, requests)),
            }
        )

    chunk_count = len({row.chunk_id for row in packed})
    filesystem = {
        "chunk_count": chunk_count,
        # One CSV plus one sidecar manifest per chunk, in one output directory.
        "output_directory_entries": chunk_count * 2,
        # sbatch writes one .out and one .err per array task.
        "log_directory_entries": chunk_count * 2,
        "max_directory_entries": max_directory_entries,
        "within_directory_budget": (
            None if max_directory_entries is None else chunk_count * 2 <= int(max_directory_entries)
        ),
    }
    if max_directory_entries is not None and not filesystem["within_directory_budget"]:
        raise ValueError(
            f"plan would create {chunk_count * 2} entries in one output directory, over the "
            f"stated budget of {max_directory_entries}; pack more rows per chunk or write "
            "several chunks per file before submitting"
        )

    return {
        "filesystem": filesystem,
        "calibration": {
            "source": model.source,
            "t0_total_seconds": model.t0_total_seconds,
            "validation_points_checked": model.validation_checked,
            "low_r2_models": list(model.low_r2_models),
        },
        "t_target_seconds": budget,
        "split_super_learner_min_k": split_super_learner_min_k,
        "task_table": str(written_table),
        "snapshot": str(written_snapshot),
        "row_count": len(packed),
        "packing": report,
        "resource_requests": {
            name: {
                "cpus_per_task": request.cpus_per_task,
                "partition": request.partition,
                "memory": request.memory,
                "time_limit": request.time_limit,
            }
            for name, request in sorted(requests.items())
        },
        "submissions": submissions,
    }


def _cluster_from_payload(payload: Mapping[str, object]) -> ClusterPolicy:
    try:
        return ClusterPolicy(
            cpus_per_task={str(k): int(v) for k, v in dict(payload["cpus_per_task"]).items()},
            partition={str(k): str(v) for k, v in dict(payload["partition"]).items()},
            time_limit={str(k): str(v) for k, v in dict(payload["time_limit"]).items()},
            memory_safety_factor=float(payload["memory_safety_factor"]),
            max_concurrent={str(k): int(v) for k, v in dict(payload["max_concurrent"]).items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid cluster policy in planning request: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    """Build a submission plan from one explicit planning-request file.

    A single reviewable JSON beats twenty command-line flags here: the request
    is the record of which cluster values a production submission was built
    from, and it can be diffed and version-controlled alongside the plan.
    """

    import argparse

    # The canonical config deserializer lives with the table writer.
    from .flat_task_table import _config_from_json

    parser = argparse.ArgumentParser(description="Plan calibrated NK-grid chunks")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument(
        "--allow-failing-acceptance",
        action="store_true",
        help="Emit a plan that fails acceptance 5/6 instead of refusing (inspection only).",
    )
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    plan = build_calibrated_plan(
        _config_from_json(payload["config"]),
        calibration_path=payload["calibration"],
        n_grid=[int(value) for value in payload["n_grid"]],
        k_grid=[int(value) for value in payload["k_grid"]],
        cluster=_cluster_from_payload(payload["cluster"]),
        table_path=payload["task_table"],
        snapshot_path=payload["snapshot"],
        output_dir=payload["output_dir"],
        panel=str(payload["panel"]),
        split_super_learner_min_k=payload.get("split_super_learner_min_k"),
        t_target_seconds=payload.get("t_target_seconds"),
        max_directory_entries=payload.get("max_directory_entries"),
        strict=not args.allow_failing_acceptance,
    )
    Path(args.plan_out).write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(plan["packing"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
