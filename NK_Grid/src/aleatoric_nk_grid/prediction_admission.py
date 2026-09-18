"""Live Lustre soft-quota admission and shared pending-space reservations.

This is a controller operation, never a per-cell/worker quota query. No data is
deleted to make a reservation fit. Existing runs retain their ledger entries
until their own verified completion releases the unwritten allowance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from .shared_queue import QueueError, atomic_json, digest, file_lock


def _query(args):
    return subprocess.check_output(args, text=True, timeout=30)


def read_live_quota(cache_root, *, query=_query):
    root = Path(cache_root).resolve()
    parts = root.parts
    try:
        at = parts.index('projects')
        project = Path(*parts[:at + 2])
    except (ValueError, IndexError) as exc:
        raise QueueError('Prediction quota admission requires an explicit Lustre project path') from exc
    project_line = query(['lfs', 'project', '-d', str(project)]).strip().splitlines()
    if len(project_line) != 1:
        raise QueueError('Cannot determine unique Lustre project identity')
    identifier = project_line[0].split()[0]
    if not identifier.isdigit():
        raise QueueError('Invalid Lustre project ID')
    mount = '/' + parts[1]
    raw = query(['lfs', 'quota', '-p', identifier, mount])
    candidates = [line.split() for line in raw.splitlines() if line.strip().startswith(mount + ' ')]
    if len(candidates) != 1 or len(candidates[0]) < 9:
        raise QueueError('Lustre quota response lacks aggregate bytes/files')
    values = candidates[0]
    def number(index):
        return int(values[index].rstrip('*'))
    result = {'project_root': str(project), 'project_id': int(identifier),
              'used_bytes': number(1) * 1024, 'soft_quota_bytes': number(2) * 1024,
              'used_files': number(5), 'soft_quota_files': number(6),
              'filesystem_free_bytes': shutil.disk_usage(root).free,
              'queried_at': datetime.now(timezone.utc).isoformat(), 'raw': raw}
    if result['soft_quota_bytes'] <= 0 or result['soft_quota_files'] <= 0:
        raise QueueError('Finite project soft quota is required')
    return result


def cache_disk_usage(root):
    """Inspect only this run's known shard/index directories, not the project."""
    root = Path(root)
    total = count = 0
    for directory in [root, *(root / name for name in ('shards', 'sample-maps', 'indexes', 'meta-results', 'locations'))]:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file():
                total += path.stat().st_size
                count += 1
    return total, count


def auxiliary_disk_usage(output):
    output = Path(output)
    paths = {output / 'base-records.sqlite'}
    if (output / 'recovery-indexes').exists():
        paths.update((output / 'recovery-indexes').glob('*'))
    for parent in (output / 'rounds', output):
        if parent.exists():
            paths.update(parent.glob('*/recovery.sqlite'))
    return sum(p.stat().st_size for p in paths if p.is_file())


def estimate_plan_storage(plan):
    """Uncompressed predictions plus conservative ordered-map/metadata bounds."""
    workflow = plan['prediction_workflow']
    estimates = plan.get('prediction_dimensions', {})
    total = cells = 0
    for panel in workflow['panels']:
        dims = estimates.get(panel['panel_id'], estimates)
        if 'n_test_max' not in dims:
            raise QueueError('Freeze prediction dimensions before storage admission')
        spec = panel['cell_spec']
        repeats = len(spec['resolved_repeat_plan'])
        ks, ns = spec['resolved_k_grid'], spec['resolved_n_grid']
        models, variants = len(panel['pipelines']), len(panel['variants'])
        holdout = int(dims['n_test_max'])
        feature_bytes = int(dims.get('feature_map_bytes', max(ks) * 256))
        id_bytes = int(dims.get('max_id_width', 128))
        shared = workflow.get('storage', {}).get('layout') == 'shared-v1'
        for n in ns:
            prediction = 8 * (models * (n + holdout) + variants * holdout)
            maps = 0 if shared else models * ((n + holdout) * (id_bytes + 8 + 32) + feature_bytes + 8 * n)
            metadata = 65536 * (models + variants)
            total += repeats * len(ks) * (prediction + maps + metadata)
        if shared:
            fold_rules = len({p['oof_folds'] for p in panel['pipelines']})
            total += repeats * (fold_rules * sum(ns) * (id_bytes + 8 + 32 + 8)
                + holdout * (id_bytes + 8 + 32) + len(ks) * feature_bytes
                + (fold_rules * len(ns) + len(ks) + 1) * 8192)
        cells += repeats * len(ns) * len(ks) * (models + variants)
    # Indexes and zlib framing can exceed raw bytes on small records.
    return {'final_bytes_upper': int(total * 1.15), 'model_cells': cells}


def check_plan_storage(plan, allocated_workers, *, query=_query):
    workflow = plan['prediction_workflow']
    cache = Path(workflow['cache_root'])
    storage = workflow.get('storage', {})
    estimated = estimate_plan_storage(plan)
    cap = int(storage.get('max_bytes', estimated['final_bytes_upper']))
    if cap < estimated['final_bytes_upper']:
        raise QueueError('Configured prediction storage budget is below the frozen uncompressed upper bound')
    # Independent fold checkpoints: an 8 MiB writer tail plus one maximum
    # complete task per worker. No whole-cache duplicate is reserved.
    temporary = int(storage.get('temporary_max_bytes', allocated_workers * 80 * 1024**2))
    minimum_temp = allocated_workers * 80 * 1024**2
    if temporary < minimum_temp:
        raise QueueError('Temporary fold-cache budget cannot cover allocated workers')
    quota = read_live_quota(cache, query=query)
    project = Path(quota['project_root'])
    ledger_path = project / '.prediction-cache-reservations.json'
    actual, files = cache_disk_usage(cache)
    actual += auxiliary_disk_usage(workflow['output_root'])
    if actual > cap:
        raise QueueError('Prediction cache exceeds its frozen storage budget')
    file_cap = int(storage.get('max_files', max(4096, (cap // (128 * 1024**2) + allocated_workers * 4 + 1) * 4)))
    if files > file_cap:
        raise QueueError('Prediction cache exceeds its frozen file-count budget')
    key = digest({'output_root': workflow['output_root'], 'workflow': workflow})
    with file_lock(project / '.prediction-cache-reservations.lock'):
        ledger = json.loads(ledger_path.read_bytes()) if ledger_path.exists() else {'format': 1, 'reservations': {}}
        reservations = ledger['reservations']
        old = reservations.get(key)
        if old is not None and old['cap_bytes'] != cap:
            raise QueueError('Frozen storage reservation changed')
        own = {'output_root': workflow['output_root'], 'cap_bytes': cap,
               'pending_bytes': max(0, cap - actual), 'pending_files': max(0, file_cap - files),
               'temporary_bytes': temporary, 'updated_at': quota['queried_at']}
        if (Path(workflow['output_root']) / 'verified.json').exists():
            receipt = json.loads((Path(workflow['output_root']) / 'verified.json').read_bytes())
            if receipt.get('complete') is True and receipt.get('plan_sha256') == digest(plan):
                own.update(pending_bytes=0, pending_files=0, temporary_bytes=0)
        projected = {**reservations, key: own}
        pending = sum(r['pending_bytes'] + r['temporary_bytes'] for r in projected.values())
        pending_files = sum(r['pending_files'] for r in projected.values())
        reserve = int(float(storage.get('quota_reserve_gb', 500)) * 10**9)
        file_reserve = int(storage.get('file_reserve', 1_000_000))
        if quota['used_bytes'] + pending + reserve > quota['soft_quota_bytes']:
            raise QueueError('Live soft quota cannot admit all pending cache reservations and project reserve')
        if pending + reserve > quota['filesystem_free_bytes']:
            raise QueueError('Filesystem free space is below pending reservations plus reserve')
        if quota['used_files'] + pending_files + file_reserve > quota['soft_quota_files']:
            raise QueueError('Live file-count quota cannot admit prediction cache reservations')
        ledger['reservations'] = projected
        atomic_json(ledger_path, ledger)
    report = {**quota, **estimated, 'reservation': own, 'all_pending_bytes': pending,
              'reserve_bytes': reserve, 'accepted': True, 'actual_cache_bytes': actual}
    atomic_json(Path(workflow['output_root']) / 'storage-admission.json', report)
    return report


def release_plan_storage(plan):
    """Release this verified run's unwritten allowance without new admission.

    Used after final publication and safely replayable after controller loss.
    Already-written data remains charged by Lustre. No quota query, reservation
    increase, file deletion, or other run's entry is involved.
    """
    workflow = plan['prediction_workflow']
    output = Path(workflow['output_root']).resolve()
    receipt_path = output / 'verified.json'
    if not receipt_path.is_file():
        raise QueueError('Storage release requires a published final verification receipt')
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt.get('complete') is not True or receipt.get('plan_sha256') != digest(plan)
            or receipt.get('prediction_cache_complete') is not True
            or receipt.get('score_complete') is not True):
        raise QueueError('Storage release requires this exact fully verified prediction plan')
    parts = Path(workflow['cache_root']).resolve().parts
    try:
        at = parts.index('projects')
        project = Path(*parts[:at + 2])
    except (ValueError, IndexError) as exc:
        raise QueueError('Storage release requires the frozen Lustre project path') from exc
    ledger_path = project / '.prediction-cache-reservations.json'
    key = digest({'output_root': workflow['output_root'], 'workflow': workflow})
    with file_lock(project / '.prediction-cache-reservations.lock'):
        if not ledger_path.is_file():
            raise QueueError('Verified run storage reservation ledger is missing')
        ledger = json.loads(ledger_path.read_bytes())
        reservations = ledger['reservations']
        if key not in reservations or reservations[key].get('output_root') != workflow['output_root']:
            raise QueueError('Verified run storage reservation is missing or belongs to another output')
        old = reservations[key]
        verified_sha256 = digest(receipt)
        if old.get('released_verified_sha256') not in (None, verified_sha256):
            raise QueueError('Storage release verification receipt changed')
        released = {**old, 'pending_bytes': 0, 'pending_files': 0, 'temporary_bytes': 0,
                    'released_verified_sha256': verified_sha256,
                    'released_at': old.get('released_at') or datetime.now(timezone.utc).isoformat()}
        if released != old:
            reservations[key] = released
            atomic_json(ledger_path, ledger)
    report = {'released': True, 'plan_sha256': digest(plan), 'reservation_key': key,
              'project_root': str(project), 'reservation': released}
    atomic_json(output / 'storage-release.json', report)
    return report
