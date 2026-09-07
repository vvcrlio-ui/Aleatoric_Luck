"""Publish validated adapter bundles without overwriting live input files.

Each stable schema is replaced atomically only after every new bundle is
complete. This is atomic per schema, not a transaction across multiple panels:
an interruption during the final switches can leave some panels on their old
version, but each schema references one complete immutable version. Old release
and universe files are retained for readers that already loaded an old schema.

This module uses only the standard library. Validation belongs to the adapter
and must finish for every staged schema before calling this function.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence


def validate_dataset_name(dataset: str) -> str:
    """Reject names that could escape an adapter's staging or output roots."""
    if not isinstance(dataset, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset):
        raise ValueError(f"adapter publication requires a safe dataset name: {dataset!r}")
    return dataset


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    # Windows does not expose POSIX directory fds; file data is still flushed.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    """Create missing ancestors and persist each new directory entry."""
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"No existing parent directory for {path}")
        current = parent
    if not current.is_dir():
        raise NotADirectoryError(current)
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        _sync_directory(directory)
        _sync_directory(directory.parent)


def _write_file(path: Path, value: Path | bytes) -> None:
    with path.open("xb") as output:
        if isinstance(value, Path):
            with value.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        else:
            output.write(value)
        output.flush()
        os.fsync(output.fileno())


def _expected_hash(value: Path | bytes) -> str:
    return _sha256(value) if isinstance(value, Path) else hashlib.sha256(value).hexdigest()


def _verify_release(directory: Path, files: dict[str, Path | bytes]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"immutable input release is not a directory: {directory}")
    if {path.name for path in directory.iterdir()} != set(files):
        raise ValueError(f"immutable input release inventory differs: {directory}")
    for name, value in files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != _expected_hash(value):
            raise ValueError(f"immutable input release content differs: {path}")


def _publish_release(target: Path, files: dict[str, Path | bytes]) -> None:
    _mkdir_durable(target.parent)
    if target.exists() or target.is_symlink():
        _verify_release(target, files)
        return
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=target.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name, value in files.items():
            _write_file(staged / name, value)
        _verify_release(staged, files)
        _sync_directory(staged)
        try:
            os.rename(staged, target)
        except OSError:
            # Concurrent publication of identical content is safe to reuse.
            if not target.exists():
                raise
            _verify_release(target, files)
        _sync_directory(target.parent)


def _publish_universe(target: Path, source: Path) -> None:
    _mkdir_durable(target.parent)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or _sha256(target) != _sha256(source):
            raise ValueError(f"immutable feature universe differs: {target}")
        return
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        _write_file(staged, source)
        # Linking publishes a complete file without replacing another writer's
        # existing version. Different basenames may share the same content hash.
        try:
            os.link(staged, target)
        except FileExistsError:
            if target.is_symlink() or _sha256(target) != _sha256(source):
                raise ValueError(f"immutable feature universe differs: {target}")
        _sync_directory(target.parent)
    _sync_directory(target.parent.parent)


def _source_path(value: str, schema: Path) -> Path:
    path = Path(value)
    resolved = (schema.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_file():
        raise ValueError(f"validated adapter input is missing: {resolved}")
    return resolved


def publish_validated_inputs(
    staged_schemas: Sequence[Path], *, schema_root: Path, ard_root: Path,
) -> dict[Path, Path]:
    """Publish complete content-addressed releases and switch stable schemas.

    Returns a mapping from the supplied staged schema paths to stable schema
    paths. All schemas must already have passed adapter validation. No old data
    or universe is overwritten or deleted. Failures while preparing any bundle
    leave all stable schemas untouched; final schema replacements are atomic
    individually, and are deliberately not presented as a multi-schema commit.
    """
    schema_root = Path(schema_root).resolve()
    ard_root = Path(ard_root).resolve()
    prepared: list[tuple[Path, Path, bytes, Path, dict[str, Path | bytes], Path, Path]] = []
    datasets: set[str] = set()
    for supplied in staged_schemas:
        supplied = Path(supplied)
        staged_schema = supplied.resolve()
        schema = json.loads(staged_schema.read_text(encoding="utf-8"))
        dataset = validate_dataset_name(schema.get("dataset"))
        if dataset in datasets:
            raise ValueError(f"adapter publication requires distinct dataset names: {dataset!r}")
        datasets.add(dataset)
        sources = {
            field: _source_path(schema[field], staged_schema)
            for field in ("table", "test_table", "feature_manifest") if schema.get(field) is not None
        }
        universe = _source_path(schema["feature_universe"]["definition_file"], staged_schema)
        provenance_path = sources["table"].parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        normalized = json.loads(json.dumps(schema))
        for role, source in sources.items():
            normalized[role] = f"{role}/{source.name}"
        normalized["feature_universe"]["definition_file"] = f"universe/{universe.name}"
        normalized_provenance = dict(provenance)
        # Avoid a cycle: the final schema names its release, while provenance
        # binds the final schema. Release identity uses its path-free schema.
        normalized_provenance["schema_sha256"] = hashlib.sha256(_json_bytes(normalized)).hexdigest()
        digest = hashlib.sha256(_json_bytes({
            "publication_version": 1, "schema": normalized,
            # Provenance binds the final schema bytes, including relative
            # locators. Distinct layouts need distinct releases; moving the
            # whole article together leaves this relative layout unchanged.
            "release_locator": Path(os.path.relpath(
                ard_root / dataset / ".releases" / "<release-id>", schema_root,
            )).as_posix(),
            "files": {role: _sha256(source) for role, source in sources.items()},
            "feature_universe_sha256": _sha256(universe),
            "provenance": normalized_provenance,
        })).hexdigest()
        release = ard_root / dataset / ".releases" / digest
        published_universe = schema_root / ".universes" / _sha256(universe) / universe.name
        files: dict[str, Path | bytes] = {}
        for role, source in sources.items():
            if source.name == "provenance.json" or source.name in files:
                raise ValueError(f"adapter bundle has colliding filenames: {source.name}")
            files[source.name] = source
            schema[role] = Path(os.path.relpath(release / source.name, schema_root)).as_posix()
        schema["feature_universe"]["definition_file"] = Path(os.path.relpath(published_universe, schema_root)).as_posix()
        schema_bytes = _json_bytes(schema)
        provenance["schema_sha256"] = hashlib.sha256(schema_bytes).hexdigest()
        files["provenance.json"] = _json_bytes(provenance)
        prepared.append((supplied, schema_root / f"{dataset}.json", schema_bytes,
                         release, files, published_universe, universe))

    # Complete every immutable bundle before preparing or switching any entry.
    for _, _, _, release, files, published_universe, universe in prepared:
        _publish_universe(published_universe, universe)
        _publish_release(release, files)
    if not prepared:
        return {}
    _mkdir_durable(schema_root)
    with tempfile.TemporaryDirectory(prefix=".publish-schemas-", dir=schema_root) as temporary:
        replacements = []
        for _, target, value, *_ in prepared:
            staged = Path(temporary) / target.name
            _write_file(staged, value)
            replacements.append((staged, target))
        for staged, target in replacements:
            os.replace(staged, target)
            _sync_directory(schema_root)
    return {supplied: target for supplied, target, *_ in prepared}
