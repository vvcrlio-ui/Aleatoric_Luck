"""Atomic sidecar persistence for opt-in per-row predictions."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


PREDICTION_EXPORT_COLUMNS = (
    "dataset",
    "model",
    "seed",
    "draw",
    "N",
    "K",
    "row_id",
    "y_true",
    "y_pred",
)


def prediction_export_path(out_path: Path) -> Path:
    """Return the final Parquet sidecar derived from the main CSV path."""

    return Path(out_path).with_suffix(".predictions.parquet")


def prediction_export_parts_dir(out_path: Path) -> Path:
    """Return the resumable per-cell prediction-part directory."""

    return Path(out_path).with_suffix(".prediction-parts")


def prediction_export_part_path(
    out_path: Path,
    *,
    model: str,
    seed: int,
    draw: int,
    n_samples: int,
    k_features: int,
) -> Path:
    """Return one deterministic cell-part path without dataset assumptions."""

    filename = (
        f"model={model}__seed={int(seed)}__draw={int(draw)}"
        f"__N={int(n_samples)}__K={int(k_features)}.parquet"
    )
    return prediction_export_parts_dir(out_path) / filename


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary(temporary: Path, target: Path) -> Path:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_directory(target.parent)
    return target


def write_prediction_part_atomic(
    rows: Sequence[Mapping[str, object]], target: Path
) -> Path:
    """Publish a complete per-cell Parquet part or leave no partial target."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=PREDICTION_EXPORT_COLUMNS)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        return _publish_temporary(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_prediction_export_atomic(
    part_paths: Iterable[Path], out_path: Path
) -> Path:
    """Stream authoritative cell parts into one atomically published sidecar."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Prediction export requires the NK Grid parquet extra") from exc

    target = prediction_export_path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    writer = None
    try:
        for part_path in part_paths:
            table = pq.read_table(Path(part_path))
            if tuple(table.column_names) != PREDICTION_EXPORT_COLUMNS:
                raise ValueError(
                    f"Prediction part has an unexpected schema: {part_path}"
                )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
        if writer is None:
            empty = pa.table(
                {
                    "dataset": pa.array([], type=pa.string()),
                    "model": pa.array([], type=pa.string()),
                    "seed": pa.array([], type=pa.int64()),
                    "draw": pa.array([], type=pa.int64()),
                    "N": pa.array([], type=pa.int64()),
                    "K": pa.array([], type=pa.int64()),
                    "row_id": pa.array([], type=pa.string()),
                    "y_true": pa.array([], type=pa.float64()),
                    "y_pred": pa.array([], type=pa.float64()),
                }
            )
            writer = pq.ParquetWriter(temporary, empty.schema)
            writer.write_table(empty)
        writer.close()
        writer = None
        return _publish_temporary(temporary, target)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
