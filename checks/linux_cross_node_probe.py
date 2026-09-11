"""Two-node synthetic TLS, ownership and crash/spool recovery acceptance probe.

Run one Slurm task on each of two nodes. Only a new validation-only queue is
created. This does not wire production Slurm continuation or run model training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error

from aleatoric_nk_grid.shared_queue import (
    Dispatcher, ModelTask, QueueError, atomic_json, file_digest,
)
from aleatoric_nk_grid.queue_service import Client, execute_worker


def read(path):
    return json.loads(path.read_bytes())


def now():
    return datetime.now(timezone.utc).isoformat()


def wait_file(root, name, timeout=180):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        errors = list(root.glob('error-rank-*.json'))
        if errors:
            raise RuntimeError('Peer failed: ' + str(read(errors[0])))
        if (root / name).exists():
            return read(root / name)
        time.sleep(.1)
    raise TimeoutError('Waiting for ' + name)


def rejected(action, fragment):
    try:
        action()
    except QueueError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError('Expected queue rejection: ' + fragment)


def tls_rejected(client):
    try:
        client.call('stats')
    except urllib.error.URLError as exc:
        assert isinstance(exc.reason, ssl.SSLCertVerificationError), repr(exc)
    else:
        raise AssertionError('Invalid TLS identity was accepted')


def result(task):
    return {**task, 'status': 'ok', 'mse': .25, 'rmse': .5, 'mae': .4,
            'validation_only': True, 'synthetic_payload': 'x' * 1024}


def controller(root, scratch):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(root, 0o700)
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    token = secrets.token_hex(32)
    private = root / 'credentials'
    private.mkdir(mode=0o700)
    token_file = private / 'token'
    token_file.write_text(token)
    os.chmod(token_file, 0o600)
    key = private / 'server.key'
    cert = private / 'server.crt'
    config = private / 'openssl.cnf'
    config.write_text('[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ext\n'
        '[dn]\nCN=' + hostname + '\n[ext]\nsubjectAltName=DNS:' + hostname + ',DNS:' + fqdn
        + '\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\n'
        + 'extendedKeyUsage=serverAuth\n')
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
        '-days', '1', '-keyout', str(key), '-out', str(cert), '-config', str(config)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.chmod(key, 0o600)
    queue_root = root / 'queue'
    queue_id = Dispatcher.create(queue_root,
        [(ModelTask(12345, 0, 122, 47, model), cost)
         for model, cost in [('ols', 3), ('ridge', 2), ('lasso', 1)]],
        identity={'validation_only': True, 'synthetic_results': True,
                  'probe': 'two-node-tls-crash-spool', 'job': os.environ['SLURM_JOB_ID']},
        lease_seconds=300)
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', 0))
        port = sock.getsockname()[1]
    url = 'https://' + hostname + ':' + str(port)
    client = Client(url, token, queue_id, ca_file=str(cert))
    process = None
    log = (root / 'service.log').open('ab')
    report = dict(started_at_utc=now(), validation_only=True, synthetic_results=True,
        controller_hostname=hostname, job=os.environ['SLURM_JOB_ID'], queue_id=queue_id,
        probe_sha256=file_digest(Path(__file__)),
        source_sha256={name: file_digest(Path(__import__('aleatoric_nk_grid').__file__).parent / name)
                       for name in ('shared_queue.py', 'queue_service.py')},
        ca_sha256=file_digest(cert), secret_permissions='directory 0700, token and key 0600',
        limitations='Three synthetic keys on two nodes. Real queue-service CLI and worker loop, '
                    'not production Slurm continuation/controller integration, full journal volume, '
                    'worker process crash recovery or model throughput.')

    def start_server():
        p = subprocess.Popen([sys.executable, '-m', 'aleatoric_nk_grid.queue_service',
            str(queue_root), '--scratch', str(scratch), '--token-file', str(token_file),
            '--host', '0.0.0.0', '--port', str(port), '--tls-cert', str(cert), '--tls-key', str(key)],
            stdout=log, stderr=log)
        deadline = time.monotonic() + 40
        try:
            while time.monotonic() < deadline:
                if p.poll() is not None:
                    raise RuntimeError('Fixture service exited: ' + str(p.returncode))
                try:
                    client.call('stats')
                    return p
                except (OSError, urllib.error.URLError):
                    time.sleep(.1)
            raise TimeoutError('Fixture service startup')
        except BaseException:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
            raise

    try:
        process = start_server()
        atomic_json(root / 'ready-first.json', {'url': url, 'queue_id': queue_id, 'hostname': hostname,
                                               'ip': socket.gethostbyname(hostname)})
        phase1 = wait_file(root, 'client-first.json')
        assert phase1['client_hostname'] != hostname, phase1
        assert phase1['spool_retained'] and phase1['cross_node_owner_rejected']
        before = client.call('stats')
        assert before.get('done') == 1 and before.get('leased') == 1 and before.get('pending') == 1
        # Kill only the child process serving this brand-new synthetic fixture.
        process.kill()
        assert process.wait(timeout=10) == -9
        process = None
        began = time.monotonic()
        process = start_server()
        restarted = client.call('stats')
        assert restarted.get('done') == 1 and restarted.get('pending') == 2 and not restarted.get('leased', 0)
        report['service_crash_restart_seconds'] = time.monotonic() - began
        atomic_json(root / 'ready-restarted.json', {'ready': True})
        phase2 = wait_file(root, 'client-complete.json')
        final = client.call('stats')
        assert final.get('done') == final['total'] == 3 and not final.get('leased', 0)
        process.terminate()
        process.wait(timeout=10)
        process = None
        with Dispatcher(queue_root, scratch=scratch) as queue:
            assert queue.stats().get('done') == 3
            queue.export_results(root / 'synthetic-results.jsonl')
        exported = [json.loads(line)['result'] for line in (root / 'synthetic-results.jsonl').read_text().splitlines()]
        assert len(exported) == len({ModelTask(**{k: r[k] for k in ('seed', 'draw', 'N', 'K', 'model')}).id for r in exported}) == 3
        assert all(r['validation_only'] for r in exported)
        report.update(phase1, **phase2, passed=True, completed_at_utc=now(), final_valid_synthetic_keys=3,
                      synthetic_export_sha256=file_digest(root / 'synthetic-results.jsonl'),
                      journal_bytes=(queue_root / 'events.jsonl').stat().st_size)
        key.unlink()
        token_file.unlink()
        report['private_key_and_token_removed'] = True
        atomic_json(root / 'report.json', report)
        print(json.dumps(report), flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log.close()


def worker(root, scratch):
    ready = wait_file(root, 'ready-first.json')
    assert socket.gethostname() != ready['hostname'], 'Slurm must place ranks on distinct nodes'
    token = (root / 'credentials/token').read_text()
    cert = str(root / 'credentials/server.crt')
    client = Client(ready['url'], token, ready['queue_id'], ca_file=cert)
    assert client.call('stats')['total'] == 3
    tls_rejected(Client(ready['url'], token, ready['queue_id']))
    tls_rejected(Client(ready['url'].replace(ready['hostname'], ready['ip']), token, ready['queue_id'], ca_file=cert))
    rejected(lambda: Client(ready['url'], '0' * 64, ready['queue_id'], ca_file=cert).call('stats'), 'Unauthorized')
    rejected(lambda: Client(ready['url'], token, 'incorrect-queue', ca_file=cert).call('stats'), 'Wrong queue identity')

    def competing_owner():
        with Dispatcher(root / 'queue', scratch=scratch):
            raise AssertionError('Two nodes acquired ownership of one queue')
    rejected(competing_owner, 'Owner already holds')

    initial = client.call('claim', worker='cross-node-initial')
    client.call('heartbeat', task_id=initial['id'], token=initial['token'], worker='cross-node-initial')
    initial_result = result(initial['task'])
    submission = dict(task_id=initial['id'], token=initial['token'], worker='cross-node-initial', result=initial_result)
    assert not client.call('submit', **submission)['duplicate']
    assert client.call('submit', **submission)['duplicate']

    class SubmitOutage:
        queue_id = client.queue_id
        def call(self, operation, **arguments):
            if operation == 'submit':
                raise OSError('Injected submit outage before delivery; synthetic fixture only')
            return client.call(operation, **arguments)

    spool = root / 'worker-spool'
    interrupted = execute_worker(SubmitOutage(), 'cross-node-spool', result, spool=spool,
                                 heartbeat_seconds=.2, idle_seconds=.1, deadline_seconds=2)
    assert interrupted['state'] == 'unacknowledged' and interrupted['accepted'] == 0, interrupted
    receipt_path = Path(interrupted['receipt'])
    saved = read(receipt_path)
    orphan = client.call('claim', worker='cross-node-spool')
    assert saved['token'] == orphan['token'] and saved['result'] == result(orphan['task'])
    atomic_json(root / 'client-first.json', dict(client_hostname=socket.gethostname(),
        trusted_cross_node_tls=True, untrusted_ca_rejected=True, wrong_hostname_rejected=True,
        wrong_token_rejected=True, wrong_queue_rejected=True, cross_node_owner_rejected=True,
        spool_retained=True, interrupted_worker_state=interrupted['state']))
    wait_file(root, 'ready-restarted.json')
    rejected(lambda: client.call('heartbeat', task_id=orphan['id'], token=orphan['token'], worker='cross-node-spool'), 'Lease is stale')
    rejected(lambda: client.call('submit', task_id=orphan['id'], token=orphan['token'], worker='cross-node-spool', result=saved['result']), 'Lease is stale')
    assert client.call('submit', **submission)['duplicate']
    executed = Counter()

    def recompute(task):
        executed[ModelTask(**task).id] += 1
        return result(task)

    resumed = execute_worker(client, 'cross-node-spool', recompute, spool=spool,
                             heartbeat_seconds=.2, idle_seconds=.1, deadline_seconds=30)
    assert resumed == {'state': 'complete', 'accepted': 2}, resumed
    assert len(executed) == 2 and all(v == 1 for v in executed.values()) and executed[orphan['id']] == 1
    assert initial['id'] not in executed and not list(spool.glob('*.json'))
    atomic_json(root / 'client-complete.json', dict(stale_heartbeat_rejected=True, stale_submit_rejected=True,
        accepted_submission_idempotent_after_restart=True, only_two_pending_keys_recomputed=True,
        stale_spooled_result_recomputed=True, spool_empty_after_acknowledgments=True,
        resumed_worker_state=resumed['state'], resumed_accepted=resumed['accepted']))


def main(args):
    assert sys.platform == 'linux'
    assert os.environ.get('SLURM_NTASKS') == '2'
    rank = int(os.environ['SLURM_PROCID'])
    assert rank in (0, 1)
    os.umask(0o077)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    assert args.output.resolve().is_relative_to(Path('/valhalla'))
    scratch = Path('/dev/shm') / ('ffc-cross-node-' + os.environ['SLURM_JOB_ID'] + '-rank-' + str(rank))
    scratch.mkdir(exist_ok=False)
    try:
        (controller if rank == 0 else worker)(args.output, scratch)
    except BaseException as exc:
        if args.output.exists():
            atomic_json(args.output / ('error-rank-' + str(rank) + '.json'),
                        {'rank': rank, 'error': repr(exc), 'time': now()})
        raise
    finally:
        assert scratch.parent == Path('/dev/shm') and scratch.name.startswith('ffc-cross-node-')
        shutil.rmtree(scratch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
