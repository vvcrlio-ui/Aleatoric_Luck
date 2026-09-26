"""Plan only missing model keys; stream the final standard CSV without origins.

The sealed old bundle remains the immutable base, rather than importing millions
of successes into the dispatcher's fsynced journal. A compact bitset indexes the
design; scratch SQLite detects conflicting duplicates without an in-memory set.
Only the final ready.json receipt publishes a usable plan. Partial directories
from interrupted preparation must not be used as queues.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from contextlib import closing

from .shared_queue import Dispatcher, ModelTask, QueueError, atomic_json, canonical, digest, file_digest
from .result_migration import assert_compatible_specs, validate_scientific_result


class Design:
    def __init__(self, spec):
        self.spec = spec
        self.ns = tuple(spec['resolved_n_grid'])
        self.ks = tuple(spec['resolved_k_grid'])
        self.repeats = tuple(tuple(x) for x in spec['resolved_repeat_plan'])
        self.models = tuple(spec['models'])
        dimensions = (self.ks, self.ns, self.repeats, self.models)
        if any(not v or len(set(v)) != len(v) for v in dimensions):
            raise QueueError('Design dimensions must be nonempty and unique')
        self.maps = tuple({v: i for i, v in enumerate(values)} for values in dimensions)
        self.count = len(self.ks) * len(self.ns) * len(self.repeats) * len(self.models)
        self.bits = bytearray((self.count + 7) // 8)

    def ordinal(self, row):
        values = []
        for name in ('seed', 'draw', 'N', 'K'):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise QueueError('Invalid integer task key')
            try:
                values.append(int(value))
            except ValueError as exc:
                raise QueueError('Invalid integer task key') from exc
        task = ModelTask(*values, row.get('model'))
        try:
            k, n, r, m = (index[value] for index, value in zip(self.maps,
                (task.K, task.N, (task.seed, task.draw), task.model)))
        except KeyError as exc:
            raise QueueError('Result outside frozen design') from exc
        return ((k * len(self.ns) + n) * len(self.repeats) + r) * len(self.models) + m

    def contains(self, ordinal):
        return bool(self.bits[ordinal // 8] & (1 << (ordinal % 8)))

    def mark(self, ordinal):
        self.bits[ordinal // 8] |= 1 << (ordinal % 8)


def checked_bundle(bundle, expected_sha256, new_spec, certificate):
    bundle = Path(bundle)
    if file_digest(bundle / 'manifest.json') != expected_sha256:
        raise QueueError('Unexpected export manifest')
    value = json.loads((bundle / 'manifest.json').read_bytes())
    if value.get('format') != 'sealed-legacy-export-v1' or value.get('sealed') is not True:
        raise QueueError('Source must be a sealed legacy export')
    if certificate.get('policy') != 'exact-scheduler-plus-conditional-gesvd-v1' or not certificate.get('files'):
        raise QueueError('Source compatibility certificate required')
    assert_compatible_specs(value['cell_spec'], new_spec, certificate=certificate)
    if file_digest(bundle / 'results.jsonl') != value['results_sha256']:
        raise QueueError('Export rows changed')
    columns = value.get('public_columns', ())
    if not columns or len(set(columns)) != len(columns):
        raise QueueError('Original public schema required')
    return value


def scan_old(bundle, manifest, design, db, accept=lambda row: None):
    count = failed = unique = duplicates = 0
    with (Path(bundle) / 'results.jsonl').open('rb') as f:
        for line in f:
            value = json.loads(line); row = value['result']; origin = value['origin']
            ordinal = design.ordinal(row)
            if origin.get('analysis_id') != manifest['source_analysis_id'] or origin.get('payload_sha256') != digest(row):
                raise QueueError('Legacy result provenance mismatch')
            if set(row) != set(manifest['public_columns']):
                raise QueueError('Old row does not match original public schema')
            count += 1
            if not validate_scientific_result(row, task_kind='regression'):
                failed += 1; continue
            sha = bytes.fromhex(digest(row))
            if design.contains(ordinal):
                if db.execute('SELECT sha FROM seen WHERE id=?', (ordinal,)).fetchone()[0] != sha:
                    raise QueueError('Conflicting old results')
                duplicates += 1; continue
            db.execute('INSERT INTO seen VALUES (?,?)', (ordinal, sha))
            design.mark(ordinal); unique += 1; accept(row)
            if unique % 8192 == 0:
                db.commit()
    db.commit()
    if count != manifest['rows']:
        raise QueueError('Export row count mismatch')
    return dict(source_records=count, old_valid_unique=unique, old_failed_records=failed,
                identical_old_duplicates=duplicates, pending=design.count - unique)


def scratch_db(directory):
    db = sqlite3.connect(str(Path(directory) / 'dedup.sqlite'))
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; '
                    'CREATE TABLE seen(id INTEGER PRIMARY KEY,sha BLOB NOT NULL)')
    return db


def prepare(bundle, output, *, new_spec, certificate, expected_manifest_sha256, scratch, cost_profile=None):
    from .single_model_worker import iter_model_tasks
    output = Path(output)
    if output.exists():
        raise QueueError('Use a new resume directory')
    manifest = checked_bundle(bundle, expected_manifest_sha256, new_spec, certificate)
    design = Design(new_spec)
    output.mkdir(parents=True)
    Path(scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='resume-', dir=scratch) as temporary:
        with closing(scratch_db(temporary)) as db:
            report = scan_old(bundle, manifest, design, db)
    # Identity embeds complete inputs, not mutable filenames.
    identity = {'cell_spec': new_spec, 'compatibility_certificate_sha256': digest(certificate),
                'base_export_manifest_sha256': expected_manifest_sha256,
                'public_columns': manifest['public_columns'], 'cost_profile': cost_profile}
    queue_id = None
    if report['pending']:
        def remaining():
            for task, cost in iter_model_tasks(new_spec, profile=cost_profile):
                if not design.contains(design.ordinal(task.__dict__)):
                    yield task, cost
        queue_id = Dispatcher.create(output / 'queue', remaining(), identity=identity)
    atomic_json(output / 'certificate.json', certificate)
    atomic_json(output / 'ready.json', {'format': 'pending-resume-v1', 'identity': identity,
        'base_bundle': str(Path(bundle).resolve()), 'queue_id': queue_id,
        'expected_total': design.count, **report})
    return json.loads((output / 'ready.json').read_bytes())


def merge(output, *, resumed, new_results=None, scratch):
    """Called offline with queue ownership held after workers have finished.

    CSV is published only after exact full design coverage, disjoint old/new
    keys and valid metrics. The accompanying receipt binds all input hashes.
    Caller obtains new_results with Dispatcher.export_results under its lock.
    """
    output, resumed = Path(output), Path(resumed)
    if output.exists() or output.with_suffix(output.suffix + '.manifest.json').exists():
        raise QueueError('Use a new final output path')
    receipt = json.loads((resumed / 'ready.json').read_bytes())
    if receipt.get('format') != 'pending-resume-v1':
        raise QueueError('Unknown resume plan')
    identity = receipt['identity']
    certificate = json.loads((resumed / 'certificate.json').read_bytes())
    if digest(certificate) != identity['compatibility_certificate_sha256']:
        raise QueueError('Compatibility certificate changed')
    bundle = Path(receipt['base_bundle'])
    manifest = checked_bundle(bundle, identity['base_export_manifest_sha256'], identity['cell_spec'], certificate)
    if identity['public_columns'] != manifest['public_columns']:
        raise QueueError('Public schema changed')
    design = Design(identity['cell_spec'])
    if design.count != receipt['expected_total']:
        raise QueueError('Expected design changed')
    Path(scratch).mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output.with_name(output.name + '.incomplete')
    versions = {}; new_count = 0
    new_sha = file_digest(new_results) if new_results is not None else None
    try:
        with temporary_csv.open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=manifest['public_columns'], lineterminator='\n')
            writer.writeheader()
            def accept(row):
                if any(name not in row for name in manifest['public_columns']):
                    raise QueueError('Missing public result columns')
                writer.writerow({key: row[key] for key in manifest['public_columns']})
                version = str(row.get('algorithm_version', ''))
                versions[version] = versions.get(version, 0) + 1
            with tempfile.TemporaryDirectory(prefix='merge-', dir=scratch) as temporary:
                with closing(scratch_db(temporary)) as db:
                    report = scan_old(bundle, manifest, design, db, accept)
            if new_results is not None:
                with Path(new_results).open('rb') as new_file:
                    for line in new_file:
                        value = json.loads(line); row = value['result']
                        ordinal = design.ordinal(row)
                        if value['origin'].get('queue_id') != receipt['queue_id'] or receipt['queue_id'] is None:
                            raise QueueError('Wrong resumed queue origin')
                        if design.contains(ordinal):
                            raise QueueError('Duplicate or overlapping resumed result')
                        if not validate_scientific_result(row, task_kind='regression'):
                            raise QueueError('Unresolved numerical failure')
                        if row.get('algorithm_version') != identity['cell_spec']['algorithm_version']:
                            raise QueueError('Incorrect resumed algorithm version')
                        design.mark(ordinal); accept(row); new_count += 1
            if report['old_valid_unique'] != receipt['old_valid_unique']:
                raise QueueError('Old completed coverage changed')
            if report['old_valid_unique'] + new_count != design.count:
                raise QueueError('Final result coverage incomplete')
            f.flush(); os.fsync(f.fileno())
        if new_results is not None and file_digest(new_results) != new_sha:
            raise QueueError('New results changed during merge')
        report.update(expected_total=design.count, new_valid_unique=new_count,
                      csv_sha256=file_digest(temporary_csv), public_columns=manifest['public_columns'],
                      added_source_columns=False, algorithm_versions=versions,
                      resume_receipt_sha256=file_digest(resumed / 'ready.json'),
                      base_export_manifest_sha256=identity['base_export_manifest_sha256'],
                      new_results_sha256=new_sha, validated_complete=True)
        os.replace(temporary_csv, output)
        atomic_json(output.with_suffix(output.suffix + '.manifest.json'), report)
        return report
    except BaseException:
        temporary_csv.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('prepare')
    plan.add_argument('--bundle', type=Path, required=True)
    plan.add_argument('--manifest-sha256', required=True)
    plan.add_argument('--spec', type=Path, required=True)
    plan.add_argument('--certificate', type=Path, required=True)
    plan.add_argument('--cost-profile', type=Path)
    final = sub.add_parser('merge')
    final.add_argument('--resumed', type=Path, required=True)
    final.add_argument('--new-results', type=Path)
    for command in (plan, final):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(args.bundle, args.output, new_spec=json.loads(args.spec.read_bytes()),
            certificate=json.loads(args.certificate.read_bytes()), expected_manifest_sha256=args.manifest_sha256,
            scratch=args.scratch, cost_profile=json.loads(args.cost_profile.read_bytes()) if args.cost_profile else None)
    else:
        value = merge(args.output, resumed=args.resumed, new_results=args.new_results, scratch=args.scratch)
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
