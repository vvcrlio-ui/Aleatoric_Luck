"""Protocol-v2 numerical adapter: persist first, return only cache references.

The SL branch never opens a numerical session or loads source training data.
Only the cache-only combiner sees training labels; evaluation labels are read
after fitting. A missing/corrupt column cannot invoke the base branch.
"""
from __future__ import annotations

from contextlib import ExitStack
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import threading

import numpy as np

from .prediction_cache import (PredictionCacheWriter, WriterRecoveryIndex, CacheStorageError,
                               cache_identity, read_record, seal_stopped_writers,
                               retire_private_fold_shards, safe_cache_path, fold_record_identity,
                               FORMAT as CACHE_FORMAT)
from .shared_queue import QueueError, digest, file_digest


def array_identity(arrays):
    return {name: {"dtype": np.asarray(value).dtype.str,
                   "shape": list(np.asarray(value).shape),
                   "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
            for name, value in sorted(arrays.items())}


def _sl_convergence(records, combiner_converged):
    """Retain base warnings even when the lightweight combiner converges."""
    nonconverged = [record.metadata.get("pipeline_id") or record.metadata["task"]["pipeline_id"]
        for record in records if record.status == "nonconverged" or record.metadata.get("converged") is False]
    known = all(record.metadata.get("converged") is not None or record.status == "nonconverged"
                for record in records)
    base_converged = False if nonconverged else (True if known else None)
    overall = (False if base_converged is False or combiner_converged is False
               else True if base_converged is True and combiner_converged is True else None)
    return {"base_nonconverged": nonconverged, "base_converged": base_converged,
            "combiner_converged": combiner_converged, "converged": overall}


class PredictionTaskExecutor:
    def __init__(self, identity, *, worker, repo_root, on_storage_fault=None, recovery_reference=None):
        from .prediction_workflow import validate_contract
        self.workflow = identity["prediction_workflow"]
        self.contract = validate_contract(self.workflow["contract"])
        self._workflow_sha256 = digest(self.contract)
        self._immutable_digests, self._cost_identities = {}, {}
        self.phase = self.workflow["phase"]
        self.root = Path(self.contract["cache_root"])
        if not (self.root / "manifest.json").is_file():
            raise QueueError("Controller must freeze prediction cache manifest before workers start")
        manifest = json.loads((self.root / "manifest.json").read_bytes())
        if (manifest.get("format") != CACHE_FORMAT or manifest.get("compression") != "zlib"
                or manifest.get("dtype") != "float64" or manifest.get("contract") != self.contract
                or manifest.get("workflow_sha256") != self._workflow_sha256
                or manifest.get("plan_sha256") != identity.get("plan_sha256")):
            raise QueueError("Prediction cache manifest differs from the frozen worker plan/contract")
        self.worker, self.repo_root = worker, repo_root
        self.drain_requested = threading.Event()
        self.on_storage_fault = on_storage_fault
        self.stack, self.sessions = ExitStack(), OrderedDict()
        self.stack.callback(self._close_sessions)
        self.shared_maps = None
        if self.contract.get('storage', {}).get('layout') == 'shared-v1':
            from .prediction_maps import SharedMaps
            self.shared_maps = SharedMaps(self.contract)
            self.stack.callback(self.shared_maps.close)
        self.global_recovery = None
        self.recovery_catalogs = []
        self.legacy_fold_lookup = bool(recovery_reference and recovery_reference.get("contains_legacy_fold_identities", True))
        if recovery_reference is not None:
            if (recovery_reference.get("format") not in ("prediction-recovery-v1", "prediction-recovery-v2")
                    or recovery_reference.get("plan_sha256") != identity.get("plan_sha256")
                    or recovery_reference.get("workflow_sha256") != self._workflow_sha256):
                raise QueueError("Cross-round recovery index identity differs from this worker plan")
            for catalog in recovery_reference.get('catalogs', [recovery_reference]):
                recovery_path = Path(catalog["path"]).resolve()
                if not recovery_path.is_relative_to(Path(self.contract["output_root"]).resolve()):
                    raise QueueError("Cross-round recovery index escapes the submission output root")
                if file_digest(recovery_path) != catalog.get("sha256"):
                    raise QueueError("Cross-round recovery index checksum changed")
                connection = sqlite3.connect(recovery_path.as_uri() + "?mode=ro&immutable=1",
                                              uri=True, check_same_thread=False)
                self.stack.callback(connection.close)
                self.recovery_catalogs.append(connection)
        self.panels = {p["panel_id"]: p for p in self.contract["panels"]}
        if self.phase == "sl":
            self._verify_sl_receipt(identity)
        # The worker_slot lock fences previous incarnations of this logical slot.
        # A live old shard still refuses repair through its independent OS lock.
        seal_stopped_writers(self.root, writer_id=worker, writer_revoked=True)
        self.recovery = WriterRecoveryIndex(self.root, worker)
        self.writer = self.stack.enter_context(PredictionCacheWriter(self.root, writer_id=worker,
            shard_target_mib=self.contract.get("storage", {}).get("shard_target_mib", 128)))
        self.fold_root = Path(self.contract["output_root"]) / "prediction-training-checkpoints"
        self.completed_fold_parents, self.fold_candidates = set(), set()
        self.pinned_fold_paths = set()
        self.fold_writer = self.fold_recovery = None
        self.fold_live_bytes = 0
        self.fold_byte_limit = 64 * 1024 * 1024
        if self.phase == "base":
            seal_stopped_writers(self.fold_root, writer_id=worker, writer_revoked=True)
            self.fold_recovery = WriterRecoveryIndex(self.fold_root, worker)
            for entries in self.fold_recovery.records.values():
                for reference, _ in entries:
                    self.legacy_fold_lookup |= reference.get("fold_identity_version") != 2
                    self.fold_live_bytes += int(reference["length"]) + 1024
                    self.fold_candidates.add(reference["path"])
                    parent = reference.get("parent_identity")
                    if parent and parent not in self.completed_fold_parents and self._lookup(parent) is not None:
                        self.completed_fold_parents.add(parent)
            if recovery_reference is not None:
                # Immutable round indexes can still be read by another slot.
                # Only newly generated private shards may be retired in-round.
                self.pinned_fold_paths.update(self.fold_candidates)
            self.fold_writer = self.stack.enter_context(PredictionCacheWriter(self.fold_root,
                writer_id=worker, shard_target_mib=8))
            self._prune_folds()
        self.stats = {"cache_hits": 0, "base_fit_count": 0, "cache_write_seconds": 0.,
                      "cache_read_seconds": 0., "full_fit_seconds": 0., "oof_fit_seconds": 0.}

    def __enter__(self):
        return self

    def request_drain(self):
        self.drain_requested.set()

    def _close_sessions(self):
        while self.sessions:
            _, session = self.sessions.popitem(); session.close()

    def __exit__(self, *exc):
        try:
            self.stack.close()
            self._prune_folds()
        except OSError as error:
            self.writer.failed = True
            if self.on_storage_fault is not None:
                self.on_storage_fault(error)
            raise

    def _verify_sl_receipt(self, identity):
        """Validate the frozen barrier once before any SL task/index lookup."""
        output = Path(self.contract["output_root"])
        receipt_path = output / "base-verified.json"
        if not receipt_path.is_file():
            raise QueueError("SL worker cannot start before the verified base barrier")
        receipt = json.loads(receipt_path.read_bytes())
        if (receipt.get("complete") is not True or receipt.get("plan_sha256") != identity.get("plan_sha256")
                or receipt.get("workflow_sha256") != self._workflow_sha256
                or digest(receipt) != self.workflow.get("base_receipt_sha256")):
            raise QueueError("SL worker base receipt differs from its immutable queue")
        if file_digest(output / "base-records.sqlite") != receipt.get("records_index_sha256"):
            raise QueueError("SL worker verified base record index changed")
        from .prediction_evidence import verify_cache_evidence
        for evidence in receipt.get("cache_indexes", []):
            verify_cache_evidence(self.root, evidence)
        self.base_records = sqlite3.connect((output / 'base-records.sqlite').resolve().as_uri() + '?mode=ro&immutable=1',
                                            uri=True, check_same_thread=False)
        self.stack.callback(self.base_records.close)

    def _prune_folds(self):
        if self.fold_writer is None:
            return
        self.fold_candidates.update(self.fold_writer.sealed_shards)
        self.fold_writer.sealed_shards.clear()
        candidates = self.fold_candidates - self.pinned_fold_paths
        if not candidates:
            return
        sizes = {relative: sum(int(reference["length"]) + 1024
                    for entries in self.fold_recovery.records.values() for reference, _ in entries
                    if reference["path"] == relative) for relative in candidates}
        try:
            retired = retire_private_fold_shards(self.fold_root, sorted(candidates),
                completed_parent_identities=self.completed_fold_parents, worker_exclusive=True)
        except OSError:
            self.writer.failed = True
            raise
        for relative in retired:
            self.fold_recovery.remove_path(relative)
            self.fold_live_bytes -= sizes[relative]
        self.fold_candidates.difference_update(retired)

    def _persist_fold(self, identity, arrays, metadata):
        from .single_model_worker import json_result
        try:
            # Reserve bounded JSON/index overhead and enforce exact encoded
            # frame size before writing. Failed cells cannot grow without limit.
            reference = self.fold_writer.append(identity, arrays, json_result(metadata),
                remaining_bytes=self.fold_byte_limit - self.fold_live_bytes - 1024)
            self.fold_recovery.add(reference)
            self.fold_live_bytes += int(reference["length"]) + 1024
        except OSError:
            self.writer.failed = True
            self.fold_writer.failed = True
            raise
        return reference

    def _lookup(self, identity):
        return self._combined_lookup(identity, self.recovery, self.root, "main")

    def _combined_lookup(self, identity, local, root, kind):
        key = cache_identity(identity) if isinstance(identity, dict) else identity
        found = local.find(key)
        for connection in getattr(self, 'recovery_catalogs', [self.global_recovery] if self.global_recovery else []):
            for content, raw_reference in connection.execute(
                    "SELECT content_sha256,reference FROM records WHERE root_kind=? AND identity=?", (kind, key)):
                record = read_record(root, json.loads(raw_reference), expected_identity=key, require_sealed=True)
                actual = record.reference["content_sha256"]
                if (content is not None and content != actual) or (
                        found is not None and found.reference["content_sha256"] != actual):
                    raise QueueError("Cross-round exact cache identity has conflicting prediction content")
                found = record
        return found

    def _frozen_digest(self, panel, definition=None, *, phase="base"):
        """Memoize only immutable contract inputs, never mutable data arrays."""
        if not hasattr(self, "_immutable_digests"):
            self._immutable_digests = {}
        identifier = None if definition is None else definition["pipeline_id" if phase == "base" else "variant_id"]
        key = (panel["panel_id"], phase if definition is not None else "input", identifier)
        if key not in self._immutable_digests:
            self._immutable_digests[key] = digest(panel["cell_spec"] if definition is None else definition)
        return self._immutable_digests[key]

    def _append(self, identity, arrays, metadata, *, kind="prediction"):
        from .single_model_worker import json_result
        reference = self.writer.append(identity, arrays, json_result(metadata), kind=kind)
        try:
            self.recovery.add(reference)
        except CacheStorageError:
            self.writer.failed = True
            raise
        return reference

    def __call__(self, value):
        from .prediction_workflow import PredictionTask, cost_identity
        from .single_model_worker import json_result
        task = PredictionTask(**value)
        if task.phase != self.phase or task.base_library_id != self.contract["base_library_id"]:
            raise QueueError("Worker task phase/library differs from immutable queue")
        panel = self.panels[task.panel_id]
        started = time.perf_counter()
        row = self._base(task, panel) if self.phase == "base" else self._sl(task, panel)
        row.update(asdict(task))
        cost_key = (task.phase, task.panel_id, task.pipeline_id, task.variant_id)
        if cost_key not in self._cost_identities:
            self._cost_identities[cost_key] = cost_identity(task, self.contract)
        row["_cost_identity"] = dict(self._cost_identities[cost_key])
        row["_prediction_worker_seconds"] = time.perf_counter() - started
        self.stats["cache_write_seconds"] = self.writer.write_seconds
        self.stats["checkpoint_write_seconds"] = self.fold_writer.write_seconds if self.fold_writer else 0.
        return json_result(row)

    def _session(self, panel):
        if panel["panel_id"] not in self.sessions:
            from .execution_contract import CellExecutionSpec
            from .nk_grid import NKGridExecutionSession
            # At most one panel's large raw frames per worker. Writer lifetime
            # is independent and spans all panel/seed/draw tasks in this phase.
            while self.sessions:
                _, old = self.sessions.popitem(last=False); old.close()
            store = None
            scratch = os.environ.get('SLURM_TMPDIR')
            if scratch and Path(scratch).is_dir() and self.contract.get('storage', {}).get('layout') == 'shared-v1':
                from .cell_cache import NodeInputStore, wait_lock
                namespace = digest(panel['cell_spec'])
                node_root = Path(scratch) / 'nk-raw-input'; node_root.mkdir(exist_ok=True)
                with wait_lock(node_root / 'admission.lock'):
                    # One namespace per job/node keeps the total, not just each
                    # panel, within 1 GiB. Other panels use private sessions.
                    namespaces = [p.name for p in node_root.iterdir() if p.is_dir()]
                    if not namespaces or namespaces == [namespace]:
                        store = NodeInputStore(node_root / namespace,
                            namespace=namespace, max_bytes=1024 * 1024**2)
            self.sessions[panel["panel_id"]] = NKGridExecutionSession.open(
                CellExecutionSpec.from_payload(panel["cell_spec"]), repo_root=self.repo_root, input_store=store)
        return self.sessions[panel["panel_id"]]

    def _base(self, task, panel):
        recipe = next(p for p in panel["pipelines"] if p["pipeline_id"] == task.pipeline_id)
        session = self._session(panel)
        arguments = dict(seed=task.seed, draw=task.draw, n_samples=task.N,
                         k_features=task.K, model=task.model, oof_folds=recipe["oof_folds"])
        prepared = session.prediction_cell_inputs(**arguments, pipeline_id=task.pipeline_id)
        samples = prepared["sample_arrays"]
        maps = None
        if getattr(self, 'shared_maps', None) is not None:
            maps, ordered_arrays = self.shared_maps.get(task, recipe['oof_folds'], samples)
        else:
            ordered_arrays = array_identity(samples)
        cell = {"panel_id": task.panel_id, "seed": task.seed, "draw": task.draw,
                "N": task.N, "K": task.K, "input_spec_sha256": self._frozen_digest(panel),
                "ordered_arrays": ordered_arrays}
        identity = {"format": "prediction-task-v1", "cell_identity": cell,
                    "task": asdict(task), "pipeline": recipe,
                    "training": prepared["metadata"]}
        if getattr(self, 'shared_maps', None) is not None:
            # The immutable manifest already holds all pipeline definitions and
            # the full training configuration. Repeated records name that entry.
            training = dict(prepared['metadata'])
            training['model_params_sha256'] = digest(training.pop('model_params'))
            identity.update(format='prediction-task-v2',
                pipeline={'pipeline_sha256': self._frozen_digest(panel, recipe)}, training=training)
        parent_identity = cache_identity(identity)
        expected_fits = 1 + len(prepared["folds"])
        previous = self._lookup(parent_identity)
        if previous is not None:
            self.stats["cache_hits"] += 1
            row = dict(previous.metadata["result"])
            row.update(prediction_cache_ref=dict(previous.reference), _base_fit_count=0,
                       _resumed_fit_count=0, _expected_base_fit_count=expected_fits, _combiner_fit_count=0,
                       _fit_seconds=0., _full_fit_seconds=0., _oof_fit_seconds=0.,
                       _prediction_cache_hit=True)
            return row
        resumed = {}
        for fold in [-1, *range(len(prepared["folds"]))]:
            old = self._combined_lookup(fold_record_identity(parent_identity, fold),
                                        self.fold_recovery, self.fold_root, "fold")
            if old is None and self.legacy_fold_lookup:
                old = self._combined_lookup({"parent": identity, "fold": fold},
                                            self.fold_recovery, self.fold_root, "fold")
            if old is not None:
                resumed[fold] = {"prediction": old.arrays["holdout_prediction"],
                                 "positions": np.asarray(old.metadata["positions"], dtype=np.int64),
                                 "metadata": old.metadata["fit"]}

        def persist_fold(fold, record):
            self._persist_fold(fold_record_identity(parent_identity, fold),
                         {"holdout_prediction": record["prediction"]},
                         {"status": "ok", "positions": record["positions"].tolist(),
                          "fit": record["metadata"], "partial_fold": True})
            if self.drain_requested.is_set():
                from .queue_service import SafeBoundaryDrain
                raise SafeBoundaryDrain()

        answer = session.run_prediction_cell(**arguments, pipeline_id=task.pipeline_id,
                    mode="holdout_oof", resume_folds=resumed, persist_fold=persist_fold)
        row, metadata = dict(answer["row"]), dict(answer["metadata"])
        if (set(answer['sample_arrays']) != set(samples) or any(
                answer['sample_arrays'][key].dtype != value.dtype or
                not np.array_equal(answer['sample_arrays'][key], value) for key, value in samples.items())):
            raise QueueError("Sample/feature/fold order changed between identity and training")
        if row["status"] == "failed":
            return row
        for name, keys in ({} if maps is not None else {"training": ("train_ids", "train_positions", "y_train", "oof_fold"),
                           "evaluation": ("holdout_ids", "holdout_positions", "y_holdout"),
                           "features": ("feature_names", "source_names")}).items():
            if maps is None:
                maps = {}
            maps[name] = self.writer.append_sample_map({"kind": name},
                                {key: samples[key] for key in keys if key in samples})
        row.update(asdict(task))
        row.update(_base_fit_count=metadata.get("base_fit_count", 0),
                   _resumed_fit_count=metadata.get("resumed_fits", 0),
                   _expected_base_fit_count=expected_fits, _combiner_fit_count=0,
                   _full_fit_seconds=metadata.get("full_fit_seconds", 0.),
                   _oof_fit_seconds=metadata.get("oof_fit_seconds", 0.),
                   score_complete=True, prediction_cache_complete=True,
                   oof_complete="oof_prediction" in answer["arrays"], _prediction_cache_hit=False)
        metadata.update(status=row["status"], reason=row.get("error", ""), task=asdict(task),
                        input_spec_sha256=self._frozen_digest(panel),
                        pipeline_sha256=self._frozen_digest(panel, recipe), sample_map_refs=maps, result=row)
        if getattr(self, 'shared_maps', None) is not None and 'model_params' in metadata:
            metadata['model_params_sha256'] = digest(metadata.pop('model_params'))
            metadata['model_params_ref'] = {'path': 'manifest.json', 'panel_id': task.panel_id,
                                            'pipeline_id': task.pipeline_id}
        reference = self._append(identity, answer["arrays"], metadata)
        self.completed_fold_parents.add(parent_identity)
        self._prune_folds()
        row["prediction_cache_ref"] = reference
        for name in ("base_fit_count", "full_fit_seconds", "oof_fit_seconds"):
            self.stats[name] += metadata.get(name, 0)
        return row

    def _sl(self, task, panel):
        from .prediction_workflow import PredictionTask, base_record_for_task
        from .offline_sl import recombine_cache_records, IncompletePredictionCache
        variant = next(v for v in panel["variants"] if v["variant_id"] == task.variant_id)
        records, rows = [], []
        started = time.perf_counter()
        for pipeline_id in variant["pipeline_ids"]:
            recipe = next(p for p in panel["pipelines"] if p["pipeline_id"] == pipeline_id)
            base_task = PredictionTask(seed=task.seed, draw=task.draw, N=task.N, K=task.K,
                model=recipe["model"], phase="base", panel_id=task.panel_id, pipeline_id=pipeline_id,
                variant_id="", base_library_id=task.base_library_id)
            saved = base_record_for_task(self.contract, base_task, connection=getattr(self, 'base_records', None))
            record = read_record(self.root, saved["reference"], require_sealed=True)
            if (record.metadata.get("task") != asdict(base_task)
                    or record.metadata.get("pipeline_sha256") != self._frozen_digest(panel, recipe)
                    or record.metadata.get("input_spec_sha256") != self._frozen_digest(panel)):
                raise QueueError("Base cache provenance does not match the frozen SL input")
            records.append(record); rows.append(saved["row"])
        self.stats["cache_read_seconds"] += time.perf_counter() - started
        identity = {"format": "prediction-task-v1", "task": asdict(task), "variant": variant,
                    "source_record_hashes": [r.reference["sha256"] for r in records],
                    "cell_identity": records[0].identity["cell_identity"]}
        previous = self._lookup(identity)
        if previous is not None:
            self.stats["cache_hits"] += 1
            row = dict(previous.metadata["result"], prediction_cache_ref=dict(previous.reference),
                       _base_fit_count=0, _resumed_fit_count=0, _expected_base_fit_count=0,
                       _combiner_fit_count=0, _fit_seconds=0., _prediction_cache_hit=True)
            convergence = _sl_convergence(records, previous.metadata.get("combiner_converged",
                previous.metadata.get("converged") if row["status"] == "ok" else None))
            row.update(converged=convergence["converged"],
                       _combiner_converged=convergence["combiner_converged"],
                       _base_nonconverged=convergence["base_nonconverged"])
            return row
        training = read_record(self.root, records[0].metadata["sample_map_refs"]["training"],
                               require_sealed=True)
        y_train = training.arrays["y_train"]
        names = [r.metadata["model_name"] for r in records]
        row = dict(rows[0])
        for key in ("prediction_cache_ref", "_cost_identity"):
            row.pop(key, None)
        row.update(asdict(task))
        row.update(_base_fit_count=0, _fit_seconds=0., _full_fit_seconds=0., _oof_fit_seconds=0.,
                   _resumed_fit_count=0, _expected_base_fit_count=0, _combiner_fit_count=0,
                   _prediction_cache_hit=False)
        combiner_config = dict(variant.get("combiner") or {})
        random_state_rule = combiner_config.pop("random_state_rule", None)
        if random_state_rule is not None:
            if random_state_rule != "cell-model-seed":
                raise QueueError("Unknown frozen SL combiner random-state rule")
            seeds = {r.metadata.get("model_seed") for r in records}
            if len(seeds) != 1 or None in seeds:
                raise QueueError("Formal SL source model seeds differ or are absent")
            combiner_config["random_state"] = seeds.pop()
        try:
            answer = recombine_cache_records(records, y_train=y_train, selected_models=names,
                        variant_id=task.variant_id, task=panel["task_kind"],
                        combiner_config=combiner_config)
        except IncompletePredictionCache as exc:
            if variant["missing_policy"] != "skip":
                raise
            row.update(status="skipped", error=str(exc))
            from .nk_grid import METRIC_COLUMNS, CLASSIFICATION_METRIC_COLUMNS
            for metric_name in (*METRIC_COLUMNS, *CLASSIFICATION_METRIC_COLUMNS):
                if metric_name in row:
                    row[metric_name] = None
            arrays, combo = {}, {"base_fit_count": 0, "combiner_fit_count": 0}
        else:
            # Deliberately read evaluation labels only after the combiner fit.
            evaluation = read_record(self.root, records[0].metadata["sample_map_refs"]["evaluation"],
                                     require_sealed=True)
            from .nk_grid import compute_regression_metrics, compute_classification_metrics
            metric = compute_classification_metrics if panel["task_kind"] == "classification" else compute_regression_metrics
            row.update(metric(evaluation.arrays["y_holdout"], answer["holdout_prediction"], y_train))
            row.update(status="ok", error="")
            arrays = {"holdout_prediction": answer["holdout_prediction"],
                      "coefficients": answer["coefficients"],
                      "intercept": np.asarray([answer["intercept"]], dtype=np.float64)}
            combo = answer["metadata"]
        convergence = _sl_convergence(records, combo.get("converged") if row["status"] == "ok" else None)
        row.update(score_complete=True, prediction_cache_complete=True,
                   oof_complete=row["status"] == "ok", _combiner_fit_count=combo.get("combiner_fit_count", 0),
                   converged=convergence["converged"], _combiner_converged=convergence["combiner_converged"],
                   _base_nonconverged=convergence["base_nonconverged"])
        metadata = {**combo, **convergence, "task": asdict(task), "status": row["status"], "reason": row["error"],
                    "input_spec_sha256": self._frozen_digest(panel), "pipeline_sha256": self._frozen_digest(panel, variant, phase="sl"),
                    "source_record_hashes": identity["source_record_hashes"], "result": row,
                    "sample_map_refs": records[0].metadata["sample_map_refs"]}
        row["prediction_cache_ref"] = self._append(identity, arrays, metadata, kind="meta_result")
        return row
