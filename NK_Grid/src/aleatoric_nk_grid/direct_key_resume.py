"""One audited WAL scan -> completed bitset -> missing model tasks.

Old results stay in place. No full-design SQL index or old-result export is
needed to admit new work. Final CSV assembly rereads the hash-bound old WALs.
"""
import argparse
from array import array
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .pending_resume import Design
from .result_migration import source_compatibility, NEW_ALGORITHM, validate_scientific_result
from .shared_queue import Dispatcher, ModelTask, QueueError, atomic_json, digest, file_digest
from .scheduler_cost import CostEstimator


def add_key(bits, ordinal, count):
    if not 0 <= ordinal < count:
        raise QueueError('Completed key outside design')
    i, shift = divmod(ordinal, 8)
    mask = 1 << shift
    if bits[i] & mask:
        raise QueueError('Duplicate completed key across audited workers')
    bits[i] |= mask


def missing_tasks(spec, bits, estimator):
    ordinal = 0
    for k in spec['resolved_k_grid']:
        for n in spec['resolved_n_grid']:
            for seed, draw in spec['resolved_repeat_plan']:
                for model in spec['models']:
                    if not bits[ordinal // 8] & (1 << (ordinal % 8)):
                        yield ModelTask(seed, draw, n, k, model), estimator.estimate(model, n, k)
                    ordinal += 1


def prepare(a):
    report = json.loads((a.audit/'report.json').read_bytes())
    identity = json.loads((a.audit/'scientific-identity.json').read_bytes())
    old_spec = identity['cell_spec']; design = Design(old_spec)
    if not report['all_prefixes_checked'] or report['workers_checked'] != 698:
        raise QueueError('All original worker WALs must be audited')
    if report['totals']['aborted'] or report['totals']['uncommitted_tail_bytes']:
        raise QueueError('Incomplete or aborted WAL protocol')
    a.output.mkdir(parents=True, exist_ok=False)
    workers = [json.loads(line) for line in (a.audit/'workers.jsonl').read_bytes().splitlines()]
    valid = 0
    for worker in workers:
        if not worker['full_captured_prefix_checked'] or not worker.get('wal_sha256'):
            raise QueueError('Worker audit failed')
        if any('exception' in e or 'aborted' in e for e in worker['errors']):
            raise QueueError('Worker protocol failure')
        path = Path(worker['wal_path'])
        if path.stat().st_size != worker['captured_bytes']:
            raise QueueError('Stopped WAL size changed')
        values = array('I')
        with (a.audit/('keys-'+str(worker['worker'])+'.bin')).open('rb') as f:
            values.frombytes(f.read())
        if len(values) != worker['valid_results']:
            raise QueueError('Worker key count differs from valid rows')
        for ordinal in values:
            add_key(design.bits, ordinal, design.count)
        valid += len(values)
    if valid != 14994428 or design.count != 18000000 or report['totals']['failed_results'] != 19:
        raise QueueError('Stopped-run counts differ from accepted full audit')
    certificate = source_compatibility(a.old, a.repo)
    spec = {**old_spec, 'algorithm_version': NEW_ALGORITHM,
        'git_commit': subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
        'model_params_sha256': file_digest(a.repo/'FFCWS/model_params.yaml')}
    relative = Path('runs/discoverer-gpa-timing-balanced-20260909/prepared')
    if not (a.repo/relative).exists():
        (a.repo/relative).parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(a.old/relative,a.repo/relative)
    from .execution_contract import CellExecutionSpec
    CellExecutionSpec.from_payload(spec).resolve_inputs(repo_root=a.repo)
    bits_path = a.output/'completed.bits'; bits_path.write_bytes(design.bits)
    profile = json.loads((a.repo/'NK_Grid/scheduler_profiles/ffc_gpa_discoverer_20260911.json').read_bytes())
    manifest = {'format':'direct-wal-keys-v1','old_spec':old_spec,'cell_spec':spec,
        'public_columns':identity['public_columns'],'certificate':certificate,
        'audit_sha256':file_digest(a.audit/'report.json'),'workers':workers,
        'old_valid_unique':valid,'pending':design.count-valid,'expected_total':design.count,
        'bits_sha256':file_digest(bits_path),'cost_profile':profile}
    atomic_json(a.output/'base-manifest.json',manifest)
    queue_identity = {'cell_spec':spec,'direct_base_sha256':file_digest(a.output/'base-manifest.json'),
                      'public_columns':identity['public_columns'],'cost_profile':profile}
    queue_id = Dispatcher.create(a.output/'queue',missing_tasks(spec,design.bits,CostEstimator(profile=profile)),identity=queue_identity)
    if json.loads((a.output/'queue/manifest.json').read_bytes())['count'] != design.count-valid:
        raise QueueError('Missing task count differs from bitset complement')
    atomic_json(a.output/'ready.json',{'format':'direct-key-resume-v1','queue_id':queue_id,
        'base_manifest_sha256':file_digest(a.output/'base-manifest.json'),
        'old_valid_unique':valid,'pending':design.count-valid,'expected_total':design.count})
    # Small intermediate per-worker ordinal lists have been compacted into 2.25 MB.
    for worker in workers:
        (a.audit/('keys-'+str(worker['worker'])+'.bin')).unlink()


def merge(a):
    from . import worker_event_wal as wal
    from .nk_grid import project_public_result
    # Streaming frame reader, avoiding the legacy in-memory scan cache.
    import sys
    sys.path.insert(0,str(a.old/'runs/ffc-gpa-production-20260909'))
    from monitor import frame
    ready = json.loads((a.output/'ready.json').read_bytes())
    manifest_path = a.output/'base-manifest.json'
    if file_digest(manifest_path) != ready['base_manifest_sha256']:
        raise QueueError('Base manifest changed')
    manifest = json.loads(manifest_path.read_bytes())
    design = Design(manifest['old_spec'])
    expected_bits = (a.output/'completed.bits').read_bytes()
    if hashlib.sha256(expected_bits).hexdigest() != manifest['bits_sha256']:
        raise QueueError('Completed key bitset changed')
    final = a.output/'ffc_median_mode_gpa.csv'
    if final.exists(): raise QueueError('Final output already exists')
    temp = a.output/'ffc_median_mode_gpa.incomplete.csv'
    old_count = new_count = 0
    with temp.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=manifest['public_columns'],lineterminator='\n');writer.writeheader()
        for item in manifest['workers']:
            path=Path(item['wal_path'])
            # Hash every byte as it is decoded, without a separate verification pass.
            class Reader:
                def __init__(self,handle): self.handle=handle;self.sha=hashlib.sha256()
                def read(self,n=-1):
                    data=self.handle.read(n);self.sha.update(data);return data
                def tell(self): return self.handle.tell()
                def seek(self,n): return self.handle.seek(n)
            with path.open('rb') as raw:
                reader=Reader(raw)
                while True:
                    record=frame(reader)
                    if record is None: break
                    if record[0]['event_type']!='TASK_RESULT': continue
                    columns,rows=wal.decode_public_rows(record[1])
                    if list(columns)!=manifest['public_columns']: raise QueueError('Old columns changed')
                    for row in rows:
                        ordinal=design.ordinal(row)
                        if not expected_bits[ordinal//8] & (1<<(ordinal%8)): continue
                        if not validate_scientific_result(row,task_kind='regression'): raise QueueError('Invalid accepted old row')
                        add_key(design.bits,ordinal,design.count);writer.writerow(row);old_count+=1
                if reader.sha.hexdigest()!=item['wal_sha256'] or reader.tell()!=item['captured_bytes']:
                    raise QueueError('Old WAL changed after the completed-key scan')
        if old_count!=ready['old_valid_unique']: raise QueueError('Old coverage mismatch')
        with a.new_results.open('rb') as new:
            for line in new:
                envelope=json.loads(line);row=envelope['result']
                if envelope['origin'].get('queue_id')!=ready['queue_id']: raise QueueError('Wrong new queue')
                if not validate_scientific_result(row,task_kind='regression') or row['algorithm_version']!=manifest['cell_spec']['algorithm_version']:
                    raise QueueError('Invalid new row')
                add_key(design.bits,design.ordinal(row),design.count)
                writer.writerow(project_public_result(row,header=manifest['public_columns']));new_count+=1
        if old_count+new_count!=design.count: raise QueueError('Final design incomplete')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,final)
    atomic_json(a.output/'ffc_median_mode_gpa.csv.manifest.json',{'validated_complete':True,
        'old_valid_unique':old_count,'new_valid_unique':new_count,'expected_total':design.count,
        'csv_sha256':file_digest(final),'base_manifest_sha256':ready['base_manifest_sha256'],
        'new_results_sha256':file_digest(a.new_results),'added_source_columns':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','merge'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--repo',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--new-results',type=Path)
    a=p.parse_args();(prepare if a.command=='prepare' else merge)(a)

if __name__=='__main__': main()
