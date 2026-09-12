"""Opt-in single-model dispatcher. One owner, durable journal, local SQLite index.

No Slurm submission or modification of the legacy engine occurs in this module.
The journal/immutable task manifest may live on shared storage; the rebuildable
SQLite index must live on the dispatcher's local scratch, never be shared by workers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Iterable


class QueueError(ValueError):
    pass


class LeaseLostError(QueueError):
    """Authoritative ownership loss; never a data/identity validation failure."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as f:
        f.write(canonical(value) + b"\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path); sync_directory(path.parent)


@contextmanager
def file_lock(path):
    """Nonblocking OS lock, including Windows tests. No stale PID lock files."""
    handle = Path(path).open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise QueueError(f"Owner already holds {path}") from exc
    try:
        yield
    finally:
        handle.close()


@dataclass(frozen=True)
class ModelTask:
    seed: int
    draw: int
    N: int
    K: int
    model: str

    def __post_init__(self):
        for name in ("seed", "draw", "N", "K"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in {"N", "K"} else 0):
                raise QueueError(f"Invalid {name}")
        if not isinstance(self.model, str) or not self.model:
            raise QueueError("Invalid model")

    @property
    def id(self):
        return digest(asdict(self))

    @property
    def cell(self):
        return digest([self.seed, self.draw, self.N, self.K])


class Dispatcher:
    """Thread-safe service object. Results are committed before acknowledgment.

    Leases are invalidated on every dispatcher restart (new epoch), fencing old
    workers. A repeated already-accepted submission with the same token/payload
    is acknowledged, including after restart. Numerical failures are terminal
    failures; infrastructure lease expiry has a bounded retry budget.
    """
    @staticmethod
    def create(root, tasks: Iterable[tuple[ModelTask, float]], *, identity: dict,
               lease_seconds=120., max_attempts=5):
        root = Path(root)
        if root.exists():
            raise QueueError("Use a new output directory")
        if not math.isfinite(lease_seconds) or lease_seconds <= 0 or max_attempts < 1:
            raise QueueError("Invalid lease policy")
        canonical(identity)
        root.mkdir(parents=True)
        count = 0
        with (root / "tasks.jsonl").open("xb") as f:
            for task, cost in tasks:
                if not math.isfinite(cost) or cost <= 0:
                    raise QueueError("Cost must be positive and finite")
                f.write(canonical({"task": asdict(task), "cost": float(cost)}) + b"\n")
                count += 1
            f.flush(); os.fsync(f.fileno())
        if not count:
            raise QueueError("Empty task design")
        manifest = {"format": "single-model-queue-v1", "identity": identity,
                    "tasks_sha256": file_digest(root / "tasks.jsonl"), "count": count,
                    "lease_seconds": float(lease_seconds), "max_attempts": int(max_attempts)}
        atomic_json(root / "manifest.json", manifest)
        atomic_json(root / "queue-id.json", {"queue_id": digest(manifest)})
        return digest(manifest)

    def __init__(self, root, *, scratch, clock=time.time):
        self.root = Path(root).resolve()
        self.clock = clock
        self.mutex = threading.RLock()
        self._owner = file_lock(self.root / "dispatcher.lock")
        self._owner.__enter__()
        self.closed = False
        self.poisoned = False
        self.db = None
        self.journal = None
        self.db_path = None
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_bytes())
            self.queue_id = digest(self.manifest)
            if json.loads((self.root / "queue-id.json").read_bytes())["queue_id"] != self.queue_id:
                raise QueueError("Manifest changed")
            if file_digest(self.root / "tasks.jsonl") != self.manifest["tasks_sha256"]:
                raise QueueError("Task manifest changed")
            scratch = Path(scratch).resolve()
            if scratch == self.root or self.root in scratch.parents:
                raise QueueError("SQLite scratch must be separate from durable output")
            scratch.mkdir(parents=True, exist_ok=True)
            self.db_path = scratch / ("queue-" + uuid.uuid4().hex + ".sqlite")
            self.db = sqlite3.connect(self.db_path, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.db.executescript('''
                PRAGMA journal_mode=MEMORY;
                PRAGMA synchronous=OFF;
                CREATE TABLE tasks(id TEXT PRIMARY KEY, cell TEXT, task TEXT, cost REAL,
                  state TEXT NOT NULL DEFAULT 'pending', attempt INTEGER NOT NULL DEFAULT 0,
                  worker TEXT, token TEXT, expiry REAL, result TEXT, accepted_token TEXT,
                  origin TEXT) WITHOUT ROWID;
                CREATE INDEX ready ON tasks(state,cost DESC,id);
                CREATE INDEX affinity ON tasks(state,cell,cost DESC);
                CREATE INDEX leases ON tasks(state,expiry);
                CREATE UNIQUE INDEX busy_worker ON tasks(worker) WHERE state='leased';
            ''')
            count = 0
            with (self.root / "tasks.jsonl").open("rb") as f:
                for line in f:
                    entry = json.loads(line); task = ModelTask(**entry["task"])
                    try:
                        self.db.execute("INSERT INTO tasks(id,cell,task,cost) VALUES(?,?,?,?)",
                                        (task.id, task.cell, canonical(asdict(task)).decode(), entry["cost"]))
                    except sqlite3.IntegrityError as exc:
                        raise QueueError("Duplicate logical task") from exc
                    count += 1
                    if count % 4096 == 0:
                        self.db.commit()
            self.db.commit()
            if count != self.manifest["count"]:
                raise QueueError("Task count changed")
            self.counts = {"pending": count}
            self.sequence = 0; self.previous = "0" * 64; self.paused = False
            path = self.root / "events.jsonl"
            path.touch(exist_ok=True)
            with path.open("r+b") as f:
                while True:
                    offset = f.tell(); line = f.readline(2 * 1024 * 1024 + 1)
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        if len(line) > 2 * 1024 * 1024:
                            raise QueueError("Oversized journal record")
                        f.truncate(offset); f.flush(); os.fsync(f.fileno()); break
                    frame = json.loads(line)
                    body = frame["body"]
                    if (frame["sha256"] != digest(body) or body["previous"] != self.previous
                            or body["sequence"] != self.sequence or body["queue_id"] != self.queue_id):
                        raise QueueError("Journal integrity failure")
                    self._apply(body["event"])
                    self.previous = frame["sha256"]; self.sequence += 1
            self.journal = path.open("ab", buffering=0)
            self.epoch = uuid.uuid4().hex
            self._commit({"kind": "restart", "epoch": self.epoch})
        except BaseException:
            self.close()
            raise

    def _apply(self, event):
        kind = event["kind"]
        old_state = self._get(event["id"])["state"] if "id" in event else None
        if kind == "restart":
            self.db.execute("UPDATE tasks SET state='pending',worker=NULL,token=NULL,expiry=NULL WHERE state='leased'")
            self.db.execute("UPDATE tasks SET state='exhausted' WHERE state='pending' AND attempt>=?", (self.manifest["max_attempts"],))
        elif kind == "lease":
            self.db.execute("UPDATE tasks SET state='leased',attempt=attempt+1,worker=?,token=?,expiry=? WHERE id=?",
                            (event["worker"], event["token"], event["expiry"], event["id"]))
        elif kind == "heartbeat":
            self.db.execute("UPDATE tasks SET expiry=? WHERE id=?", (event["expiry"], event["id"]))
        elif kind == "expire":
            self.db.execute("UPDATE tasks SET state=CASE WHEN attempt>=? THEN 'exhausted' ELSE 'pending' END,worker=NULL,token=NULL,expiry=NULL WHERE id=?",
                            (self.manifest["max_attempts"], event["id"]))
        elif kind in {"result", "import"}:
            result = event["result"]
            state = "failed" if result["status"] == "failed" else "done"
            self.db.execute("UPDATE tasks SET state=?,result=?,accepted_token=?,origin=?,worker=NULL,token=NULL,expiry=NULL WHERE id=?",
                            (state, canonical(result).decode(), event.get("token"), canonical(event.get("origin", {"queue_id": self.queue_id})).decode(), event["id"]))
        elif kind == "pause":
            self.paused = event["paused"]
        else:
            raise QueueError("Unknown journal event")
        self.db.commit()
        if kind == "restart":
            self.counts = {row[0]: row[1] for row in self.db.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state")}
        elif old_state is not None:
            new_state = self._get(event["id"])["state"]
            if old_state != new_state:
                self.counts[old_state] = self.counts.get(old_state, 0) - 1
                self.counts[new_state] = self.counts.get(new_state, 0) + 1

    def _commit(self, event):
        if self.closed or self.poisoned:
            raise QueueError("Dispatcher closed or requires restart")
        body = {"sequence": self.sequence, "previous": self.previous,
                "queue_id": self.queue_id, "event": event}
        frame = {"body": body, "sha256": digest(body)}
        data = canonical(frame) + b"\n"
        if len(data) > 2 * 1024 * 1024:
            raise QueueError("Result exceeds journal record limit")
        try:
            remaining = memoryview(data)
            while remaining:
                wrote = self.journal.write(remaining)
                if not wrote:
                    raise OSError("Short journal write")
                remaining = remaining[wrote:]
            os.fsync(self.journal.fileno())
            self._apply(event)
            self.previous = frame["sha256"]; self.sequence += 1
        except BaseException:
            self.poisoned = True
            raise

    def _reap(self):
        # Stream bounded batches; never materialize all expired production tasks.
        while True:
            rows = self.db.execute("SELECT id FROM tasks WHERE state='leased' AND expiry<=? LIMIT 256", (self.clock(),)).fetchall()
            if not rows:
                return
            for row in rows:
                self._commit({"kind": "expire", "id": row["id"]})

    def _get(self, task_id):
        row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise QueueError("Unknown task")
        return row

    def _check_lease(self, task_id, token, worker):
        row = self._get(task_id)
        if row["state"] != "leased" or row["token"] != token or row["worker"] != worker or row["expiry"] <= self.clock():
            raise LeaseLostError("Lease is stale, expired or owned by another worker")
        return row

    def claim(self, worker, *, cached_cells=()):
        if not isinstance(worker, str) or not worker or len(worker) > 256:
            raise QueueError("Invalid worker")
        with self.mutex:
            self._reap()
            active = self.db.execute("SELECT * FROM tasks WHERE state='leased' AND worker=?", (worker,)).fetchone()
            if active is not None:
                return self._lease_payload(active)  # idempotent lost claim reply
            if self.paused:
                return {"state": "paused"}
            candidates = self.db.execute("SELECT * FROM tasks WHERE state='pending' ORDER BY cost DESC,id LIMIT 64").fetchall()
            if not candidates:
                stats = self.stats()
                return {"state": "wait" if stats.get("leased", 0) else ("blocked" if stats.get("failed", 0) or stats.get("exhausted", 0) else "complete")}
            cells = set(list(cached_cells)[:32])
            top = candidates[0]
            row = next((r for r in candidates if r["cell"] in cells and r["cost"] >= top["cost"] * .5), top)
            self._commit({"kind": "lease", "id": row["id"], "worker": worker,
                          "token": self.epoch + ":" + uuid.uuid4().hex,
                          "expiry": self.clock() + self.manifest["lease_seconds"]})
            return self._lease_payload(self._get(row["id"]))

    def _lease_payload(self, row):
        return {"state": "task", "id": row["id"], "task": json.loads(row["task"]),
                "cell": row["cell"], "token": row["token"], "expiry": row["expiry"],
                "attempt": row["attempt"], "queue_id": self.queue_id}

    def heartbeat(self, task_id, token, worker):
        with self.mutex:
            self._check_lease(task_id, token, worker)
            expiry = self.clock() + self.manifest["lease_seconds"]
            self._commit({"kind": "heartbeat", "id": task_id, "expiry": expiry})
            return {"expiry": expiry}

    def validate_result(self, task_id, result):
        row = self._get(task_id)
        expected = json.loads(row["task"])
        if any(str(result.get(k)) != str(v) for k, v in expected.items()):
            raise QueueError("Result key does not match task")
        if result.get("status") not in {"ok", "skipped", "failed"}:
            raise QueueError("Invalid result status")
        # JSON canonicalization refuses nonfinite numeric payloads. Legacy CSV
        # metric strings receive additional semantic validation in the importer.
        canonical(result)
        return row

    def submit(self, task_id, token, worker, result):
        with self.mutex:
            row = self.validate_result(task_id, result)
            if row["state"] in {"done", "failed"}:
                if row["accepted_token"] == token and row["result"] == canonical(result).decode():
                    return {"accepted": True, "duplicate": True}
                if row["accepted_token"] != token:
                    raise LeaseLostError("Completed task belongs to another lease")
                raise QueueError("Conflicting or stale completed submission")
            self._check_lease(task_id, token, worker)
            self._commit({"kind": "result", "id": task_id, "token": token, "result": result})
            return {"accepted": True, "duplicate": False}

    def import_result(self, task_id, result, origin):
        """Internal trusted migration API; deliberately absent from worker RPC."""
        with self.mutex:
            row = self.validate_result(task_id, result)
            if result["status"] == "failed":
                return False  # Keep old numerical failures executable.
            if row["state"] == "done" and row["result"] == canonical(result).decode() and row["origin"] == canonical(origin).decode():
                return False
            if row["state"] != "pending" or row["attempt"]:
                raise QueueError("Migration requires untouched pending tasks")
            self._commit({"kind": "import", "id": task_id, "result": result, "origin": origin})
            return True

    def pause(self, paused=True):
        with self.mutex:
            self._commit({"kind": "pause", "paused": bool(paused)})
            return self.stats()

    def stats(self):
        with self.mutex:
            return {**self.counts, "total": self.manifest["count"], "paused": self.paused, "queue_id": self.queue_id}

    def export_results(self, path):
        with self.mutex:
            with Path(path).open("x", encoding="utf-8") as f:
                for row in self.db.execute("SELECT result,origin FROM tasks WHERE state IN ('done','failed') ORDER BY id"):
                    f.write(json.dumps({"result": json.loads(row[0]), "origin": json.loads(row[1])}, allow_nan=False) + "\n")
                f.flush(); os.fsync(f.fileno())

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.journal:
            self.journal.close()
        if self.db:
            self.db.close()
        if self.db_path:
            self.db_path.unlink(missing_ok=True)
        self._owner.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
