"""Readiness gates: real service CLI, stale epochs, failed replay and timeouts."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import pytest

from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask, QueueError, atomic_json
from aleatoric_nk_grid.queue_service import make_server
from aleatoric_nk_grid.queue_readiness import (
    DEFAULT_WAIT_SECONDS, check_ready, publish_ready, remove_owned_ready, wait_ready,
)


TOKEN = "readiness-test-token-only-12345678"


def queue(tmp_path):
    root = tmp_path / "queue"
    Dispatcher.create(root, [(ModelTask(1, 0, 10, 1, "ols"), 1)], identity={"probe": "readiness"})
    return root


def test_wait_exceeds_measured_replay_and_never_accepts_missing_file(tmp_path):
    assert DEFAULT_WAIT_SECONDS > 2494.464
    with pytest.raises(TimeoutError, match="do not admit workers"):
        wait_ready(tmp_path / "absent", queue_id="q", generation="g", token=TOKEN,
                   timeout=.08, poll_seconds=.01)


def test_live_identity_and_stale_generation_epoch_rejection(tmp_path):
    root = queue(tmp_path); path = root / "ready.json"
    with Dispatcher(root, scratch=tmp_path / "scratch") as dispatcher:
        server = make_server(dispatcher, token=TOKEN)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        try:
            ready = publish_ready(path, dispatcher, host="127.0.0.1", port=server.server_port,
                                  tls=False, generation="round-1-attempt-1")
            arguments = dict(queue_id=dispatcher.queue_id, generation="round-1-attempt-1", token=TOKEN)
            assert check_ready(path, **arguments) == ready
            for field in ("queue_id", "generation", "epoch"):
                atomic_json(path, {**ready, field: "stale"})
                with pytest.raises(QueueError):
                    check_ready(path, **arguments)
            atomic_json(path, {**ready, "url": "http://compute-node:8765"})
            with pytest.raises(QueueError, match="HTTPS"):
                check_ready(path, **arguments)
            atomic_json(path, ready)
            with pytest.raises(QueueError, match="Unauthorized"):
                check_ready(path, **{**arguments, "token": "bad-token"})
            successor = {**ready, "epoch": "successor"}
            atomic_json(path, successor)
            remove_owned_ready(path, ready)
            assert json.loads(path.read_bytes()) == successor
            remove_owned_ready(path, successor)
            assert not path.exists()
        finally:
            server.shutdown(); server.server_close(); thread.join()


def service_command(root, token, ready, scratch):
    return [sys.executable, "-m", "aleatoric_nk_grid.queue_service", str(root),
            "--scratch", str(scratch), "--token-file", str(token), "--port", "0",
            "--ready-file", str(ready), "--generation", "round1-attempt1"]


def test_real_service_cli_publishes_verified_ready_and_kill_leaves_no_false_ready(tmp_path):
    root = queue(tmp_path); ready = root / "ready.json"
    token = tmp_path / "token"; token.write_text(TOKEN)
    queue_id = json.loads((root / "queue-id.json").read_bytes())["queue_id"]
    with (tmp_path / "service.log").open("w") as log:
        child = subprocess.Popen(service_command(root, token, ready, tmp_path / "scratch"),
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            value = wait_ready(ready, queue_id=queue_id, generation="round1-attempt1", token=TOKEN,
                               timeout=8, poll_seconds=.02)
            result = subprocess.run([sys.executable, "-m", "aleatoric_nk_grid.queue_readiness",
                str(ready), "--queue-id", queue_id, "--generation", "round1-attempt1",
                "--token-file", str(token), "--timeout", "2"],
                check=True, capture_output=True, text=True, timeout=5)
            assert json.loads(result.stdout) == value
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    assert ready.exists()  # Real process death can leave a stale file.
    with pytest.raises(TimeoutError):
        wait_ready(ready, queue_id=queue_id, generation="round1-attempt1", token=TOKEN,
                   timeout=.12, poll_seconds=.01)


def test_replay_failure_never_publishes_ready(tmp_path):
    root = queue(tmp_path); ready = root / "ready.json"
    token = tmp_path / "token"; token.write_text(TOKEN)
    with Dispatcher(root, scratch=tmp_path / "scratch") as dispatcher:
        dispatcher.claim("first")
    # A complete malformed event is corruption, not a repairable truncated tail.
    with (root / "events.jsonl").open("ab") as handle:
        handle.write(b'{"not_a_valid_event":true}\n')
    completed = subprocess.run(service_command(root, token, ready, tmp_path / "scratch"),
                               capture_output=True, text=True, timeout=8)
    assert completed.returncode != 0
    assert not ready.exists()


def test_wait_retries_until_atomic_publication(tmp_path):
    root = queue(tmp_path); ready = root / "ready.json"
    with Dispatcher(root, scratch=tmp_path / "scratch") as dispatcher:
        server = make_server(dispatcher, token=TOKEN)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        publisher = threading.Timer(.1, publish_ready, args=(ready, dispatcher),
            kwargs=dict(host="127.0.0.1", port=server.server_port, tls=False, generation="later"))
        publisher.start()
        try:
            value = wait_ready(ready, queue_id=dispatcher.queue_id, generation="later", token=TOKEN,
                               timeout=3, poll_seconds=.02)
            assert value["epoch"] == dispatcher.epoch
        finally:
            publisher.join(); server.shutdown(); server.server_close(); thread.join()
