"""Multi-panel cells, compact set subtraction, and a bounded durable dispatcher.

This protocol intentionally does not import legacy result identities. Input
metadata and the small frozen plan are checked by the controller, not hashed
again by each worker. One ordinal names one model cell within that plan.
"""
from bisect import bisect_right
from collections import Counter, deque
from contextlib import ExitStack
import csv
import json
import math
import mmap
import os
from pathlib import Path
import struct
import threading
import time
import uuid

from .shared_queue import QueueError, LeaseLostError, atomic_json, canonical, file_lock

FORMAT = "multi-panel-slurm-v1"
KEYS = ("panel_id", "seed", "draw", "N", "K", "model")
R2_COLUMNS = ("r2_holdout", "null_mse_train_N", "r2_holdout_reason")


def read(path):
    return json.loads(Path(path).read_bytes())


def file_metadata(path):
    path = Path(path).resolve()
    s = path.stat()
    return {"path": str(path), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def check_files(files):
    for saved in files:
        if file_metadata(saved["path"]) != saved:
            raise QueueError("Frozen input changed: " + saved["path"])


class SuiteDesign:
    def __init__(self, panels):
        self.panels = panels
        self.names = {}; self.offsets = []; self.shapes = []; self.maps = []
        self.count = 0
        for i, p in enumerate(panels):
            if p["name"] in self.names:
                raise QueueError("Duplicate panel")
            self.names[p["name"]] = i
            dimensions = (tuple(p["K"]), tuple(p["N"]), tuple(map(tuple, p["repeats"])), tuple(p["models"]))
            if any(not v or len(set(v)) != len(v) for v in dimensions):
                raise QueueError("Empty or duplicate design dimension")
            self.offsets.append(self.count)
            shape = tuple(map(len, dimensions)); self.shapes.append(shape)
            self.count += math.prod(shape)
            self.maps.append(tuple({v: j for j, v in enumerate(d)} for d in dimensions))
        if not panels or self.count >= 2**32:
            raise QueueError("Suite must contain 1..2^32-1 cells")

    def task(self, ordinal):
        if type(ordinal) is not int or not 0 <= ordinal < self.count:
            raise QueueError("Ordinal outside design")
        i = bisect_right(self.offsets, ordinal) - 1
        k, n, r, m = self.shapes[i]
        q, mi = divmod(ordinal - self.offsets[i], m)
        q, ri = divmod(q, r); ki, ni = divmod(q, n)
        p = self.panels[i]; seed, draw = p["repeats"][ri]
        return dict(panel_id=p["name"], seed=seed, draw=draw, N=p["N"][ni], K=p["K"][ki], model=p["models"][mi])

    def ordinal(self, row):
        try:
            i = self.names[row["panel_id"]]
            if any(type(row[k]) is not int for k in ("seed", "draw", "N", "K")):
                raise QueueError("Cell integer key has wrong type")
            ki, ni, ri, mi = (d[v] for d, v in zip(self.maps[i],
                (row["K"], row["N"], (row["seed"], row["draw"]), row["model"])))
            _, n, r, m = self.shapes[i]
            return self.offsets[i] + ((ki * n + ni) * r + ri) * m + mi
        except (KeyError, TypeError) as exc:
            raise QueueError("Result outside frozen panel design") from exc


def valid_result(row, panel):
    status = row.get("status")
    if status == "failed":
        return False
    if row.get("dataset") != panel["dataset"] or row.get("algorithm_version") != panel["algorithm_version"]:
        raise QueueError("Result scientific identity changed")
    if status == "skipped":
        if not row.get("error"):
            raise QueueError("Skipped result needs a reason")
        return True
    if status != "ok":
        raise QueueError("Unknown result status")
    required = ("mse", "rmse", "mae") if panel["task"] == "regression" else ("brier", "accuracy")
    for name in required:
        value = row.get(name)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise QueueError("Invalid required metric: " + name)
    if panel["task"] == "classification":
        for name in ("brier", "accuracy", "roc_auc", "pr_auc", "balanced_accuracy", "f1"):
            if row.get(name) is not None and not 0 <= row[name] <= 1:
                raise QueueError("Classification metric outside [0,1]: " + name)
    null, r2 = row.get("null_mse_train_N"), row.get("r2_holdout")
    if null is None or not math.isfinite(null) or null < 0:
        raise QueueError("Missing/invalid paper R2 denominator")
    if null > 0:
        error = row["mse" if panel["task"] == "regression" else "brier"]
        if r2 is None or not math.isfinite(r2) or not math.isclose(r2, 1 - error / null, rel_tol=1e-10, abs_tol=1e-10):
            raise QueueError("Paper R2 does not match saved prediction error")
    elif r2 is not None or row.get("r2_holdout_reason") != "zero_baseline_error":
        raise QueueError("Zero R2 denominator must be explicit")
    return True


def scan(plan, rounds, accept=None):
    """One pass over durable result cells; duplicate accepted keys count once."""
    design = SuiteDesign(plan["panels"])
    bits = bytearray((design.count + 7) // 8)
    counts = [Counter() for _ in design.panels]
    done = 0
    with ExitStack() as locks:
        for directory in map(Path, rounds):
            locks.enter_context(file_lock(directory / "dispatcher.lock"))
            manifest = read(directory / "manifest.json")
            if manifest["plan_id"] != plan["plan_id"]:
                raise QueueError("Round belongs to another suite")
            path = directory / "results.jsonl"
            if not path.exists():
                continue
            before = file_metadata(path)
            with path.open("rb") as f:
                while line := f.readline(2 * 1024 * 1024 + 1):
                    if len(line) > 2 * 1024 * 1024:
                        raise QueueError("Oversized result record")
                    if not line.endswith(b"\n"):
                        break  # A killed writer's uncommitted final record.
                    entry = json.loads(line); row = entry["result"]
                    ordinal = design.ordinal(row)
                    if entry["queue_id"] != manifest["queue_id"] or entry["ordinal"] != ordinal:
                        raise QueueError("Result provenance/ordinal mismatch")
                    pi = design.names[row["panel_id"]]
                    if not valid_result(row, design.panels[pi]):
                        continue
                    byte, shift = divmod(ordinal, 8)
                    if bits[byte] & (1 << shift):
                        continue
                    bits[byte] |= 1 << shift; done += 1
                    counts[pi][row["status"]] += 1
                    counts[pi]["nonconverged"] += row.get("converged") is False
                    if accept is not None:
                        accept(pi, ordinal, row)
            if file_metadata(path) != before:
                raise QueueError("Results still being written; stop workers before resume")
    return design, bits, {"done": done, "remaining": design.count - done,
                          "panels": [dict(c) for c in counts]}


def prepare_round(plan, rounds, directory):
    """D minus completed cells, streamed into a uint32 file ordered per panel."""
    import numpy as np
    from .scheduler_cost import CostEstimator
    directory = Path(directory)
    if (directory / "prepared.json").exists():
        saved = read(directory / "prepared.json")
        if saved["sources"] == list(map(str, rounds)) and saved["plan_id"] == plan["plan_id"]:
            check_files(saved["source_files"])
            manifest = read(directory / "manifest.json")
            metadata = file_metadata(directory / "remaining.u32")
            if [metadata["size"], metadata["mtime_ns"]] != manifest["index_stat"]:
                raise QueueError("Prepared pending index changed")
            return saved
        raise QueueError("Prepared round sources changed")
    design, bits, report = scan(plan, rounds)
    if not report["remaining"]:
        return report
    if directory.exists():
        raise QueueError("Unpublished round directory; select a fresh round directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = directory.with_name("." + directory.name + "-" + uuid.uuid4().hex)
    stage.mkdir()
    done = np.frombuffer(bits, dtype=np.uint8); estimator = CostEstimator()
    ranges = []; cursor = 0
    with (stage / "remaining.u32").open("xb") as f:
        for pi, p in enumerate(design.panels):
            start = cursor
            groups = sorted(((-estimator.estimate(model, n, k), ki, ni, mi)
                for ki, k in enumerate(p["K"]) for ni, n in enumerate(p["N"])
                for mi, model in enumerate(p["models"])))
            _, ns, repeats, models = design.shapes[pi]
            for _, ki, ni, mi in groups:
                base = design.offsets[pi] + (ki * ns + ni) * repeats * models + mi
                for begin in range(0, repeats, 65536):
                    ids = base + np.arange(begin, min(begin + 65536, repeats), dtype=np.uint32) * models
                    pending = ids[((done[ids // 8] >> (ids % 8)) & 1) == 0]
                    f.write(pending.astype("<u4", copy=False).tobytes()); cursor += len(pending)
            ranges.append([start, cursor])
        f.flush(); os.fsync(f.fileno())
    if cursor != report["remaining"]:
        raise QueueError("Set subtraction count mismatch")
    manifest = dict(format=FORMAT, queue_id=uuid.uuid4().hex, plan_id=plan["plan_id"],
                    plan=str(Path(plan["launch"]["output"]) / "plan.json"),
                    count=cursor, ranges=ranges, lease_seconds=300., max_attempts=5,
                    workers=plan["allocation"]["workers"])
    metadata = file_metadata(stage / "remaining.u32")
    manifest["index_stat"] = [metadata["size"], metadata["mtime_ns"]]
    atomic_json(stage / "manifest.json", manifest)
    report.update(plan_id=plan["plan_id"], sources=list(map(str, rounds)),
                  source_files=[file_metadata(Path(r) / "results.jsonl") for r in rounds if (Path(r) / "results.jsonl").exists()])
    atomic_json(stage / "prepared.json", report)
    os.replace(stage, directory)
    return report


class SuiteDispatcher:
    """One allocation. Only leases and one recent acknowledgement per slot in RAM."""
    def __init__(self, root, *, clock=time.time):
        self.root = Path(root); self.clock = clock
        self.mutex = threading.RLock(); self.submit_mutex = threading.Lock()
        self.closed = self.poisoned = self.paused = False
        self._stack = ExitStack()
        try:
            self._stack.enter_context(file_lock(self.root / "dispatcher.lock"))
            self.manifest = read(self.root / "manifest.json")
            self.plan = read(self.manifest["plan"])
            if self.plan["plan_id"] != self.manifest["plan_id"]:
                raise QueueError("Queue plan changed")
            self.queue_id = self.manifest["queue_id"]; self.epoch = uuid.uuid4().hex
            self.design = SuiteDesign(self.plan["panels"])
            f = self._stack.enter_context((self.root / "remaining.u32").open("rb"))
            stat = os.fstat(f.fileno())
            if stat.st_size != self.manifest["count"] * 4 or [stat.st_size, stat.st_mtime_ns] != self.manifest["index_stat"]:
                raise QueueError("Pending index length mismatch")
            self.order = self._stack.enter_context(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
            self.journal = self._stack.enter_context((self.root / "results.jsonl").open("xb", buffering=0))
            self.cursors = [a for a, b in self.manifest["ranges"]]
            self.ends = [b for a, b in self.manifest["ranges"]]
            self.started = [0] * len(self.cursors)
            self.active = {}; self.by_worker = {}; self.last = {}; self.retry = deque()
            self.done = self.failed = self.exhausted = self.expired_leases = 0
            self.panel_counts = [Counter() for _ in self.cursors]
            self.next_reap = 0.
        except BaseException:
            self._stack.close(); raise

    def _worker(self, worker):
        # The slot prefix bounds both worker registration and ACK cache memory.
        if not isinstance(worker, str):
            raise QueueError("Invalid worker identity")
        try:
            slot = int(worker.split(":", 1)[0])
        except ValueError as exc:
            raise QueueError("Invalid worker slot") from exc
        if not 0 <= slot < self.manifest["workers"]:
            raise QueueError("Worker outside allocation")
        return slot

    def _reap(self):
        now = self.clock()
        if now < self.next_reap:
            return
        self.next_reap = now + 1.
        for task_id, row in list(self.active.items()):
            if row["expiry"] > now or row.get("committing"):
                continue
            del self.active[task_id]; self.by_worker.pop(row["slot"], None)
            self.expired_leases += 1
            if row["attempt"] >= self.manifest["max_attempts"]:
                self.exhausted += 1
            else:
                self.retry.append((row["ordinal"], row["attempt"] + 1))

    def _check(self, task_id, token, worker):
        row = self.active.get(task_id)
        if row is None or row["token"] != token or row["worker"] != worker or (row["expiry"] <= self.clock() and not row.get("committing")):
            raise LeaseLostError("Stale or reassigned cell lease")
        return row

    def _lease(self, row):
        return {k: row[k] for k in ("id", "token", "task", "expiry", "attempt")} | {"state": "leased", "queue_id": self.queue_id}

    def claim(self, worker, *, cached_cells=()):
        slot = self._worker(worker)
        with self.mutex:
            if self.closed or self.poisoned:
                raise QueueError("Dispatcher stopped")
            self._reap()
            if slot in self.by_worker:
                row = self.active[self.by_worker[slot]]
                if row["worker"] != worker:
                    return {"state": "wait"}
                return self._lease(row)
            previous = self.last.pop(slot, None)
            if self.paused:
                return {"state": "paused"}
            if self.retry:
                ordinal, attempt = self.retry.popleft()
            else:
                available = [i for i, end in enumerate(self.ends) if self.cursors[i] < end]
                if not available:
                    return {"state": "wait" if self.active else "blocked" if self.failed or self.exhausted else "complete"}
                untouched = [i for i in available if not self.started[i]]
                preferred = previous["panel"] if previous else None
                pi = min(untouched) if untouched else preferred if preferred in available else min(available, key=lambda i: self.started[i])
                ordinal = struct.unpack_from("<I", self.order, self.cursors[pi] * 4)[0]
                self.cursors[pi] += 1; self.started[pi] += 1; attempt = 1
            task = self.design.task(ordinal); task_id = str(ordinal)
            row = dict(id=task_id, ordinal=ordinal, task=task, worker=worker, slot=slot,
                       attempt=attempt, token=uuid.uuid4().hex, expiry=self.clock() + self.manifest["lease_seconds"])
            self.active[task_id] = row; self.by_worker[slot] = task_id
            return self._lease(row)

    def heartbeat(self, task_id, token, worker):
        with self.mutex:
            row = self._check(task_id, token, worker)
            row["expiry"] = self.clock() + self.manifest["lease_seconds"]
            return {"expiry": row["expiry"]}

    def submit(self, task_id, token, worker, result):
        slot = self._worker(worker)
        with self.submit_mutex:
            with self.mutex:
                if self.poisoned or self.closed:
                    raise QueueError("Dispatcher stopped")
                last = self.last.get(slot)
                if last and last["id"] == task_id and last["token"] == token and last["worker"] == worker:
                    if last["result"] != result:
                        raise QueueError("Conflicting duplicate acknowledgement")
                    return {"accepted": True, "duplicate": True}
                row = self._check(task_id, token, worker)
                if self.design.ordinal(result) != row["ordinal"]:
                    raise QueueError("Result cell does not match lease")
                pi = self.design.names[result["panel_id"]]
                valid = valid_result(result, self.design.panels[pi])
                raw = canonical(dict(queue_id=self.queue_id, ordinal=row["ordinal"], result=result)) + b"\n"
                row["committing"] = True
            try:
                pending = memoryview(raw)
                while pending:
                    written = self.journal.write(pending)
                    if not written:
                        raise OSError("Short result write")
                    pending = pending[written:]
                os.fsync(self.journal.fileno())
            except BaseException:
                self.poisoned = True; raise
            with self.mutex:
                del self.active[task_id]; del self.by_worker[slot]
                self.last[slot] = dict(id=task_id, token=token, worker=worker, result=result, panel=pi)
                self.done += valid; self.failed += not valid
                self.panel_counts[pi][result["status"]] += 1
                return {"accepted": True, "duplicate": False}

    def stats(self):
        with self.mutex:
            return dict(total=self.manifest["count"], done=self.done, failed=self.failed,
                        exhausted=self.exhausted, leased=len(self.active),
                        pending=sum(b - a for a, b in zip(self.cursors, self.ends)) + len(self.retry),
                        expired_leases=self.expired_leases, panels=[dict(c) for c in self.panel_counts])

    def close(self):
        with self.submit_mutex, self.mutex:
            self.closed = True; self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def finalize(plan, rounds):
    """Publication pass: project rows and verify exact unique coverage together."""
    root = Path(plan["launch"]["output"])
    if (root / "verified.json").exists():
        if read(root / "verified.json")["plan_id"] != plan["plan_id"]:
            raise QueueError("Published plan changed")
        cleanup(plan)
        return read(root / "verified.json")
    temporary = []; writers = []; headers = []
    with ExitStack() as stack:
        for p in plan["panels"]:
            folder = root / "panels" / p["name"]; folder.mkdir(parents=True, exist_ok=True)
            path = folder / ("final-" + uuid.uuid4().hex + ".tmp")
            f = stack.enter_context(path.open("x", newline="", encoding="utf-8"))
            header = tuple(dict.fromkeys(("panel_id", *p["public_columns"], *R2_COLUMNS)))
            writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n"); writer.writeheader()
            writers.append(writer); headers.append(header); temporary.append((path, f))
        def emit(pi, ordinal, row):
            # Same strict projection as the engine; no numerical imports needed.
            writers[pi].writerow({column: row[column] for column in headers[pi]})
        design, _, report = scan(plan, rounds, emit)
        if report["remaining"]:
            raise QueueError("Incomplete suite; final results not published")
        for path, f in temporary:
            f.flush(); os.fsync(f.fileno())
    summary = []
    for pi, p in enumerate(plan["panels"]):
        path, _ = temporary[pi]; final = path.with_name("final.csv")
        os.replace(path, final)
        item = dict(panel=p["name"], rows=math.prod(design.shapes[pi]),
                    **report["panels"][pi], final_csv=str(final))
        atomic_json(final.with_name("verified.json"), dict(complete=True, plan_id=plan["plan_id"], final_file=file_metadata(final), **item))
        summary.append(item)
    receipt = dict(complete=True, plan_id=plan["plan_id"], rows=design.count, panels=summary,
                   resources=plan["allocation"], checkpoint_retention=plan["launch"]["checkpoint_retention"],
                   verification="unique cell coverage and metrics; no full-file checksum pass")
    atomic_json(root / "summary.json", receipt)
    atomic_json(root / "verified.json", receipt)
    cleanup(plan)
    return receipt


def cleanup(plan):
    if plan["launch"]["checkpoint_retention"] != "delete":
        return
    import shutil
    root = Path(plan["launch"]["output"]).resolve()
    if not read(root / "verified.json").get("complete"):
        raise QueueError("Cannot delete before complete publication")
    for p in plan["panels"]:
        folder = root / "panels" / p["name"]
        receipt = read(folder / "verified.json")
        if not (folder / "final.csv").is_file() or receipt["plan_id"] != plan["plan_id"]:
            raise QueueError("Cannot delete: final result missing")
        check_files([receipt["final_file"]])
    target = root / "rounds"
    if target.is_symlink() or target.resolve() != root / "rounds":
        raise QueueError("Checkpoint path escapes run directory")
    if target.exists():
        # Worker logs and allocation evidence live inside rounds while running.
        # Retain them outside the checkpoint tree before deleting that tree.
        for directory in target.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                raise QueueError("Unexpected checkpoint directory")
            archive = root / "logs" / directory.name
            archive.mkdir(parents=True, exist_ok=True)
            sources = [directory / "manifest.json", directory / "prepared.json"]
            control = directory / "control"
            if control.exists():
                sources += [p for p in control.iterdir() if p.is_file() and p.suffix in (".json", ".out", ".err")]
            for source in sources:
                if source.exists():
                    destination = archive / source.name
                    shutil.copy2(source, destination)
                    with destination.open("r+b") as f:
                        os.fsync(f.fileno())
        shutil.rmtree(target)
    atomic_json(root / "checkpoint-archive.json", dict(complete=True, policy="delete", plan_id=plan["plan_id"]))
