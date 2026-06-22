"""File IO helpers for FFCWS prediction experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from .config import is_url


class DataError(ValueError):
    """Raised when input data are malformed or incomplete."""


def read_table(path_or_url: str | Path) -> pd.DataFrame:
    value = str(path_or_url)
    if is_url(value):
        return pd.read_csv(value, na_values=["NA", "NaN", ""], low_memory=False)

    zip_path, member = split_zip_member(value)
    if zip_path is not None and member is not None:
        return read_zip_table(zip_path, member)

    path = Path(value)
    if not path.exists():
        raise DataError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, na_values=["NA", "NaN", ""], low_memory=False)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", na_values=["NA", "NaN", ""], low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise DataError(f"Unsupported table format: {path}")


def read_text_resource(path_or_url: str | Path) -> str:
    value = str(path_or_url)
    if is_url(value):
        return download_text(value)

    zip_path, member = split_zip_member(value)
    if zip_path is not None and member is not None:
        if not zip_path.exists():
            raise DataError(f"Zip file does not exist: {zip_path}")
        with ZipFile(zip_path) as archive:
            if member not in set(archive.namelist()):
                raise DataError(f"Zip member '{member}' not found in {zip_path}")
            return archive.read(member).decode("utf-8", errors="replace")

    path = Path(value)
    if not path.exists():
        raise DataError(f"Text file does not exist: {path}")
    return path.read_text(errors="replace")


def split_zip_member(value: str) -> tuple[Path | None, str | None]:
    if value.startswith("zip://"):
        rest = value[len("zip://") :]
        if "!" not in rest:
            raise DataError("Zip table paths must use zip:///path/file.zip!member.csv")
        zip_file, member = rest.split("!", 1)
        return Path(zip_file), member.lstrip("/")
    if ".zip!" in value:
        zip_file, member = value.split("!", 1)
        return Path(zip_file), member.lstrip("/")
    return None, None


def read_zip_table(zip_path: Path, member: str) -> pd.DataFrame:
    if not zip_path.exists():
        raise DataError(f"Zip file does not exist: {zip_path}")
    suffix = Path(member).suffix.lower()
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if member not in names:
            raise DataError(f"Zip member '{member}' not found in {zip_path}")
        with archive.open(member) as handle:
            if suffix in {".csv", ".txt"}:
                return pd.read_csv(handle, na_values=["NA", "NaN", ""], low_memory=False)
            if suffix in {".tsv", ".tab"}:
                return pd.read_csv(handle, sep="\t", na_values=["NA", "NaN", ""], low_memory=False)
            raise DataError(f"Unsupported zip table member format: {member}")


def write_table(df: pd.DataFrame, path: Path, prefer_parquet: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet or path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            path.with_suffix(path.suffix + ".unavailable.txt").write_text(
                "Parquet output requested, but neither pyarrow nor fastparquet is "
                "installed. Wrote CSV fallback instead.\n"
            )
            return fallback
    df.to_csv(path, index=False)
    return path


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))


def download_text(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")
