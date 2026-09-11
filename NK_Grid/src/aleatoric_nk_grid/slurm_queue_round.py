"""One Slurm allocation: replayed TLS dispatcher plus cross-node worker steps.

The allocation owns every step, so controller loss also ends its workers. A new
allocation reuses durable spools and rebuilds the journal under an exclusive
round lock; it never launches alongside an existing controller. No future rounds
are submitted here. A failed/interrupted round is recoverable from this root.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import uuid

from .shared_queue import Dispatcher, QueueError, atomic_json, file_lock
from .queue_readiness import wait_ready
from .queue_service import Client, worker_slot


def read(path):
    return json.loads(Path(path).read_bytes())


def worker(args):
    ready = read(args.launch)
    root = Path(ready['queue'])
    slot = Path(ready['control']) / 'spools' / os.environ['SLURM_PROCID']
    info = wait_ready(ready['ready_file'], queue_id=ready['queue_id'], generation=ready['generation'],
                      token=Path(ready['token_file']).read_text().strip(), ca_file=ready['ca_file'])
    command = [sys.executable, '-m', 'aleatoric_nk_grid.single_model_worker', 'run', str(root),
               '--url', info['url'], '--repo-root', ready['repo'], '--token-file', ready['token_file'],
               '--ca-file', ready['ca_file'], '--spool', str(slot), '--max-seconds', '172800']
    # Exec ensures Slurm signals reach the actual numerical worker process.
    os.execv(sys.executable, command)


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise QueueError('A Slurm allocation is required')
    if args.workers < 1 or int(os.environ['SLURM_NTASKS']) < args.workers + 1:
        raise QueueError('Reserve one task for the dispatcher in addition to workers')
    args.control.mkdir(parents=True, exist_ok=True)
    os.chmod(args.control, 0o700)
    with file_lock(args.control / 'round.lock'):
        manifest = read(args.queue / 'manifest.json')
        qid = read(args.queue / 'queue-id.json')['queue_id']
        commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
        if manifest['identity']['cell_spec']['git_commit'] != commit:
            raise QueueError('Run the exact frozen queue source commit')
        if subprocess.check_output(['git', '-C', str(args.repo), 'status', '--porcelain'], text=True).strip():
            raise QueueError('Source checkout must be clean')
        generation = uuid.uuid4().hex
        attempt = args.control / generation
        attempt.mkdir(mode=0o700)
        token = attempt / 'token'
        token.write_text(secrets.token_hex(32)); token.chmod(0o600)
        hostname = socket.gethostname()
        cert, key = attempt / 'ca.crt', attempt / 'server.key'
        cnf = attempt / 'openssl.cnf'
        cnf.write_text('[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ext\n'
            '[dn]\nCN=' + hostname + '\n[ext]\nsubjectAltName=DNS:' + hostname + ',DNS:' + socket.getfqdn()
            + '\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\n'
            + 'extendedKeyUsage=serverAuth\n')
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '7',
                        '-keyout', str(key), '-out', str(cert), '-config', str(cnf)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        key.chmod(0o600)
        scratch = Path('/dev/shm') / ('gpa-round-' + os.environ['SLURM_JOB_ID'])
        scratch.mkdir(exist_ok=False)
        if os.statvfs(scratch).f_bavail * os.statvfs(scratch).f_frsize < args.scratch_gib * 1024**3:
            raise QueueError('Insufficient node-local scratch for measured replay volume')
        launch = dict(queue=str(args.queue.resolve()), queue_id=qid, repo=str(args.repo.resolve()),
            control=str(args.control.resolve()), generation=generation, token_file=str(token.resolve()),
            ca_file=str(cert.resolve()), ready_file=str((attempt / 'ready.json').resolve()),
            job_id=os.environ['SLURM_JOB_ID'], source_commit=commit, workers=args.workers)
        atomic_json(attempt / 'launch.json', launch)
        atomic_json(args.control / 'latest.json', launch)
        service = workers = None
        log = (attempt / 'service.log').open('ab')
        def interrupted(signum, frame):
            raise InterruptedError('Slurm controller interrupted: ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            service = subprocess.Popen([sys.executable, '-m', 'aleatoric_nk_grid.queue_service', str(args.queue),
                '--scratch', str(scratch), '--host', '0.0.0.0', '--port', '0', '--token-file', str(token),
                '--tls-cert', str(cert), '--tls-key', str(key), '--ready-file', launch['ready_file'],
                '--generation', generation, '--advertise-host', hostname], stdout=log, stderr=log)
            info = wait_ready(launch['ready_file'], queue_id=qid, generation=generation,
                              token=token.read_text(), ca_file=str(cert))
            client = Client(info['url'], token.read_text(), qid, ca_file=str(cert))
            atomic_json(attempt / 'admission.json', {'ready': info, 'stats': client.call('stats')})
            # Step tasks consume their own reserved cores; controller uses the spare.
            workers = subprocess.Popen(['srun', '--ntasks=' + str(args.workers), '--cpus-per-task=1',
                '--ntasks-per-core=1', '--kill-on-bad-exit=1', '--output=' + str(attempt / 'worker-%t.out'),
                '--error=' + str(attempt / 'worker-%t.err'), sys.executable, '-m',
                'aleatoric_nk_grid.slurm_queue_round', 'worker', '--launch', str(attempt / 'launch.json')])
            while workers.poll() is None:
                if service.poll() is not None:
                    raise QueueError('Dispatcher exited; worker step must stop before restart')
                atomic_json(attempt / 'progress.json', {'stats': client.call('stats'), 'observed_at': time.time()})
                time.sleep(15)
            if workers.returncode != 0:
                raise QueueError('Worker step failed; retain journal and spools for recovery')
            stats = client.call('stats')
            atomic_json(attempt / 'finished.json', {'stats': stats, 'worker_exit': workers.returncode})
        finally:
            if workers is not None and workers.poll() is None:
                workers.terminate()
                try:
                    workers.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    workers.kill(); workers.wait(timeout=30)
            if service is not None and service.poll() is None:
                service.terminate(); service.wait(timeout=60)
            log.close()
        # Taking ownership offline fences the prior epoch and exports exact results.
        with Dispatcher(args.queue, scratch=scratch) as queue:
            stats = queue.stats()
            queue.export_results(attempt / 'results.jsonl')
            atomic_json(args.control / 'round-result.json', {'stats': stats, 'generation': generation,
                'results': str(attempt / 'results.jsonl'), 'complete': stats.get('done', 0) == stats['total']})
        print(json.dumps(stats), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    for name in ('queue', 'repo', 'control'):
        r.add_argument('--' + name, type=Path, required=True)
    r.add_argument('--workers', type=int, required=True)
    r.add_argument('--scratch-gib', type=int, default=20)
    w = sub.add_parser('worker'); w.add_argument('--launch', type=Path, required=True)
    a = p.parse_args()
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    (run if a.command == 'run' else worker)(a)


if __name__ == '__main__':
    main()
