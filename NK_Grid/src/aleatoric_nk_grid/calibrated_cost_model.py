"""Stage B: turn a calibration artifact into packing costs and resource requests.

This module deliberately invents nothing.  Every number is either read from a
calibration JSON or supplied explicitly by the caller; every missing or
untrustworthy input raises instead of falling back to a default.  That rule
exists because the first calibration round produced a file that *looked*
complete while its memory column was a single constant -- a silent default here
would reproduce exactly that failure one layer higher.

Stage A (``flat_task_table``) already made the cost function and the resource
values injectable.  This module supplies them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .flat_task_table import CostFunction, TaskRow, preprocess_mode_for_group


#: Only this calibration format is understood. Older files lack either the
#: isolated memory scopes, schema-derived shape, or v5 measured timing fields;
#: they must be rejected rather than reinterpreted.
CALIBRATION_FORMAT_VERSION = 5

#: A power-law fit is only usable for packing inside this predicted/actual band.
#: Matches ``cost-calibration`` acceptance criterion 4.
VALIDATION_RATIO_BAND = (0.5, 2.0)


class CalibrationUnusable(ValueError):
    """Raised when a calibration artifact cannot safely drive Stage B."""


@dataclass(frozen=True)
class PowerLaw:
    """``value(N, K) = exp(log_c) * K**a * N**b`` as fitted by the harness."""

    log_c: float
    a: float
    b: float
    r2: float

    def predict(self, n_samples: int, k_features: int) -> float:
        if n_samples < 1 or k_features < 1:
            raise ValueError("power-law prediction requires N >= 1 and K >= 1")
        return float(
            math.exp(
                self.log_c
                + self.a * math.log(float(k_features))
                + self.b * math.log(float(n_samples))
            )
        )


def _power_law(payload: object, *, what: str) -> PowerLaw:
    if payload is None:
        raise CalibrationUnusable(f"calibration has no fitted {what}")
    if not isinstance(payload, Mapping):
        raise CalibrationUnusable(f"calibration entry for {what} is not an object")
    try:
        return PowerLaw(
            log_c=float(payload["log_c"]),
            a=float(payload["a"]),
            b=float(payload["b"]),
            r2=float(payload["r2"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationUnusable(f"calibration entry for {what} is malformed: {exc}") from exc


def _fits(section: Mapping[str, object], *, what: str) -> dict[str, PowerLaw]:
    """Keep only the entries that actually carry a fit, remembering the rest."""

    fitted: dict[str, PowerLaw] = {}
    for name, payload in section.items():
        if payload is None:
            continue
        fitted[str(name)] = _power_law(payload, what=f"{what}[{name}]")
    return fitted


@dataclass(frozen=True)
class CalibratedCostModel:
    """Everything Stage B needs, read from one calibration artifact."""

    source: str
    fit_cost: Mapping[str, PowerLaw]
    preprocess_cost: Mapping[str, PowerLaw]
    peak_rss_bytes: Mapping[str, PowerLaw]
    t0_total_seconds: float
    task_peak_rss_bytes: int | None
    task_peak_rss_status: str
    memory_scope_suspect_count: int
    validation_failures: tuple[Mapping[str, object], ...]
    validation_checked: int
    low_r2_models: tuple[str, ...]

    # -- construction -----------------------------------------------------

    @classmethod
    def from_json(
        cls,
        path: Path | str,
        *,
        require_validation: bool = True,
        min_r2: float | None = None,
    ) -> "CalibratedCostModel":
        """Load a calibration file, refusing anything Stage B cannot trust.

        ``require_validation`` enforces the harness's own out-of-sample check:
        a cost model whose production-scale predictions were out of band is not
        a cost model, it is a guess.  ``min_r2`` optionally rejects poorly
        fitted models outright; by default they are only reported.
        """

        resolved = Path(path).resolve()
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationUnusable(f"cannot read calibration file {resolved}: {exc}") from exc

        version = payload.get("format_version")
        if version != CALIBRATION_FORMAT_VERSION:
            raise CalibrationUnusable(
                f"calibration format_version={version!r} is not supported; "
                f"Stage B requires {CALIBRATION_FORMAT_VERSION}. Earlier formats lack "
                "required measured calibration fields and must be re-measured."
            )

        fit_cost = _fits(payload.get("fit_cost") or {}, what="fit_cost")
        preprocess_cost = _fits(payload.get("preprocess_cost") or {}, what="preprocess_cost")
        peak_rss = _fits(payload.get("peak_rss_bytes") or {}, what="peak_rss_bytes")
        if not fit_cost:
            raise CalibrationUnusable("calibration contains no fitted fit_cost model")
        if not preprocess_cost:
            raise CalibrationUnusable("calibration contains no fitted preprocess_cost mode")

        t0_section = payload.get("t0_seconds") or {}
        try:
            t0_total = float(t0_section["total"]["median"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationUnusable(f"calibration has no usable t0_seconds.total.median: {exc}") from exc
        if t0_total <= 0:
            raise CalibrationUnusable("t0_seconds.total.median must be positive")

        memory = payload.get("memory_measurement") or {}
        task_peak = memory.get("task_cgroup_peak_n_jobs_8") or {}
        task_status = str(task_peak.get("status", "absent"))
        task_bytes = task_peak.get("bytes")
        suspect = int(memory.get("cell_memory_scope_suspect_count", 0) or 0)

        checked, failures = _validation_summary(payload.get("validation") or ())
        if require_validation:
            if checked == 0:
                raise CalibrationUnusable(
                    "calibration carries no completed validation points; its coefficients "
                    "were never checked against production-scale measurements"
                )
            if failures:
                raise CalibrationUnusable(
                    f"{len(failures)}/{checked} calibration validation points fall outside "
                    f"predicted/actual band {VALIDATION_RATIO_BAND}; refusing to pack against "
                    "an unvalidated cost model (pass require_validation=False to inspect anyway)"
                )

        # Reported by default, enforced only when the caller sets a floor.
        # A low r2 means the power law describes that model badly even inside
        # its fitted range, which is a fact the packer's user should see.
        threshold = 0.8 if min_r2 is None else float(min_r2)
        low_r2 = tuple(sorted(name for name, fit in fit_cost.items() if fit.r2 < threshold))
        if min_r2 is not None and low_r2:
            raise CalibrationUnusable(f"fit_cost r2 below {min_r2} for: {', '.join(low_r2)}")

        return cls(
            source=str(resolved),
            fit_cost=fit_cost,
            preprocess_cost=preprocess_cost,
            peak_rss_bytes=peak_rss,
            t0_total_seconds=t0_total,
            task_peak_rss_bytes=int(task_bytes) if task_bytes is not None else None,
            task_peak_rss_status=task_status,
            memory_scope_suspect_count=suspect,
            validation_failures=failures,
            validation_checked=checked,
            low_r2_models=low_r2,
        )

    # -- cost (B1) --------------------------------------------------------

    def row_cost_seconds(self, row: TaskRow) -> float:
        """Core-seconds for one indivisible cell group: preprocess + every fit.

        The group is the atomic unit precisely because its models share one
        preprocessing pass, so the preprocessing term is counted exactly once.
        """

        mode = preprocess_mode_for_group(row.group)
        preprocess = self.preprocess_cost.get(mode)
        if preprocess is None:
            raise CalibrationUnusable(
                f"calibration has no preprocess_cost for mode {mode!r} "
                f"(available: {', '.join(sorted(self.preprocess_cost)) or 'none'})"
            )
        total = preprocess.predict(row.n_samples, row.k_features)
        for model in row.models:
            fit = self.fit_cost.get(model)
            if fit is None:
                raise CalibrationUnusable(
                    f"calibration has no fit_cost for model {model!r} "
                    f"(available: {', '.join(sorted(self.fit_cost)) or 'none'})"
                )
            total += fit.predict(row.n_samples, row.k_features)
        return float(total)

    def cost_function(self) -> CostFunction:
        """Return the injectable cost function Stage A's packer expects."""

        return self.row_cost_seconds

    # -- memory (B4) ------------------------------------------------------

    def row_peak_rss_bytes(self, row: TaskRow) -> float:
        """Peak resident bytes for one row: the largest model in its group.

        Models inside a group run sequentially, so the group's peak is the
        maximum rather than the sum.  The fitted quantity is already the
        whole-process-tree cell peak, so preprocessing is included.
        """

        peaks = []
        for model in row.models:
            fit = self.peak_rss_bytes.get(model)
            if fit is None:
                raise CalibrationUnusable(
                    f"calibration has no peak_rss_bytes fit for model {model!r}"
                )
            peaks.append(fit.predict(row.n_samples, row.k_features))
        if not peaks:
            raise CalibrationUnusable("task row has no models")
        return float(max(peaks))

    def require_trustworthy_memory(self) -> None:
        """Refuse to size ``--mem`` from a calibration that flagged its own memory."""

        if self.memory_scope_suspect_count:
            raise CalibrationUnusable(
                f"{self.memory_scope_suspect_count} measurements carry memory_scope_suspect; "
                "the cgroup accounting scope was wrong, so --mem cannot be derived from them"
            )
        if not self.peak_rss_bytes:
            raise CalibrationUnusable("calibration contains no fitted peak_rss_bytes model")


def _validation_summary(
    entries: Iterable[Mapping[str, object]],
) -> tuple[int, tuple[Mapping[str, object], ...]]:
    low, high = VALIDATION_RATIO_BAND
    checked = 0
    failures: list[Mapping[str, object]] = []
    for entry in entries:
        ratio = entry.get("ratio")
        if ratio is None:
            continue
        try:
            value = float(ratio)
        except (TypeError, ValueError):
            continue
        checked += 1
        if not low <= value <= high:
            failures.append(dict(entry))
    return checked, tuple(failures)
