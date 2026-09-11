"""Wait for a replayed dispatcher and verify its live authenticated identity.

This provides a launch prerequisite, not Slurm submission or controller fencing.
The caller must use a unique generation and recheck job state under its own lock.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from .queue_service import Client
from .shared_queue import QueueError, atomic_json, sync_directory


DEFAULT_WAIT_SECONDS = 5400.  # Measured full replay takes about 2,494 seconds.


def validate_endpoint(url):
    value = urlsplit(url)
    if (value.scheme not in ("http", "https") or not value.hostname or
            value.username or value.password or value.query or value.fragment or value.path):
        raise QueueError("Invalid dispatcher readiness URL")
    if value.scheme == "http" and value.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise QueueError("Non-loopback readiness requires HTTPS")
    if not value.port:
        raise QueueError("Readiness endpoint requires an explicit port")
    return url


def publish_ready(path, dispatcher, *, host, port, tls, generation):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", generation):
        raise QueueError("Invalid launch generation")
    # Publish only with a constructed Dispatcher: its full replay has completed.
    endpoint_host = "[" + host + "]" if ":" in host else host
    url = validate_endpoint(f'{"https" if tls else "http"}://{endpoint_host}:{port}')
    value = {"format": "dispatcher-ready-v1", "queue_id": dispatcher.queue_id,
             "epoch": dispatcher.epoch, "generation": generation, "url": url}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    return value


def remove_owned_ready(path, expected):
    path = Path(path)
    try:
        if json.loads(path.read_bytes()) == expected:
            path.unlink(); sync_directory(path.parent)
    except FileNotFoundError:
        pass


def check_ready(path, *, queue_id, generation, token, ca_file=None, timeout=5):
    value = json.loads(Path(path).read_bytes())
    if (value.get("format") != "dispatcher-ready-v1" or value.get("queue_id") != queue_id or
            value.get("generation") != generation or not isinstance(value.get("epoch"), str) or
            not value["epoch"]):
        raise QueueError("Dispatcher readiness identity or generation mismatch")
    client = Client(validate_endpoint(value["url"]), token, queue_id, ca_file=ca_file, timeout=timeout)
    live = client.call("identity")
    if live != {"queue_id": queue_id, "epoch": value["epoch"]}:
        raise QueueError("Dispatcher readiness file is stale")
    # Detect replacement between reading the file and completing the RPC.
    if json.loads(Path(path).read_bytes()) != value:
        raise QueueError("Dispatcher readiness changed during verification")
    return value


def wait_ready(path, *, queue_id, generation, token, ca_file=None,
               timeout=DEFAULT_WAIT_SECONDS, poll_seconds=2.):
    if not all(math.isfinite(x) and x > 0 for x in (timeout, poll_seconds)):
        raise ValueError("Readiness timeout and poll interval must be positive and finite")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Dispatcher did not become ready; do not admit workers")
        try:
            return check_ready(path, queue_id=queue_id, generation=generation, token=token,
                               ca_file=ca_file, timeout=min(5., remaining))
        except OSError:
            # Missing file/unavailable endpoint is never proof of readiness.
            pass
        time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ready_file", type=Path)
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--ca-file", type=Path)
    parser.add_argument("--timeout", type=float, default=DEFAULT_WAIT_SECONDS)
    args = parser.parse_args()
    value = wait_ready(args.ready_file, queue_id=args.queue_id, generation=args.generation,
                       token=args.token_file.read_text().strip(), ca_file=args.ca_file, timeout=args.timeout)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
