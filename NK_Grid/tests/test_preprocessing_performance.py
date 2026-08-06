from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aleatoric_nk_grid import calibrate_cost as cc


_WORKER = r"""
import json
import resource
import statistics
import sys
import time
from pathlib import Path

from aleatoric_nk_grid.calibrate_cost import build_session
from aleatoric_nk_grid.preprocessing import _preprocess_cell_reference, preprocess_cell

schema_path = Path(sys.argv[1])
k = int(sys.argv[2])
mode = sys.argv[3]
session = build_session(schema_path, "y", seed=0)
orders = session.orders_for(0, 0)
groups_by_name = {group.name: group for group in session.groups}
groups = [groups_by_name[name] for name in orders.feature_names[:k]]
columns = [feature for group in groups for feature in group.features]
train = session.split.X_train.loc[orders.row_index[:4242], columns]
test = session.split.X_test.loc[:, columns]
function = _preprocess_cell_reference if mode == "reference" else preprocess_cell
peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if mode == "vectorized":
    # Exclude first-allocation noise from the scaling estimate. The reference
    # measurement remains one call because it is deliberately minutes long.
    function(train, test, groups, session.imputation, model_name="ols")
repetitions = 7 if mode == "vectorized" else 1
times = []
for _ in range(repetitions):
    started = time.perf_counter()
    result = function(train, test, groups, session.imputation, model_name="ols")
    times.append(time.perf_counter() - started)
seconds = statistics.median(times)
peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({
    "mode": mode,
    "k": k,
    "n": len(train),
    "columns": len(columns),
    "seconds": seconds,
    "seconds_samples": times,
    "peak_rss_before": peak_before,
    "peak_rss_after": peak_after,
    "peak_rss_delta": max(0, peak_after - peak_before),
    "unobserved": result.K_unobserved,
}))
"""


def _measure(schema_path: Path, k: int, mode: str) -> dict[str, float | int | str]:
    src_root = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root)
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER, str(schema_path), str(k), mode],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


@pytest.mark.slow
def test_preprocess_cell_scaling_and_peak_memory_benchmark(tmp_path, record_property):
    """Record the required isolated-process time and peak-RSS benchmark."""

    thread_report = cc.check_thread_env()
    assert thread_report["ok"], thread_report
    schema_path, _ = cc.generate_synthetic_bundle(
        tmp_path / "bundle",
        cc.SyntheticDataParams(
            n_train=5303,
            shape=cc.PanelShape(
                schema_path="<performance-fixture>",
                sources=tuple(
                    cc.ShapeSource("continuous", ("float64",)) for _ in range(8053)
                ),
                dtype_source="declared",
                dtype_metadata_declared=8053,
                dtype_metadata_total=8053,
            ),
            seed=0,
        ),
    )
    measurements = [
        _measure(schema_path, k, mode)
        for k in (1000, 3125, 8053)
        for mode in ("reference", "vectorized")
    ]
    record_property("preprocess_cell_benchmark", json.dumps(measurements))

    optimized = {
        int(row["k"]): float(row["seconds"])
        for row in measurements
        if row["mode"] == "vectorized"
    }
    for low, high in ((1000, 3125), (3125, 8053)):
        exponent = math.log(optimized[high] / optimized[low]) / math.log(high / low)
        assert exponent <= 1.3
    assert optimized[8053] <= 60.0
