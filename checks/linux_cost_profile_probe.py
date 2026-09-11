"""Native Discoverer single-model durations for a bounded GPA ordering profile.

Four cells, two reverse-order repeats, original model/CV budgets and native
isolation. No shared input/outer-preprocessing cache, no production edits.
"""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import resource
import socket
import statistics
import sys
import time

from threadpoolctl import threadpool_limits
from aleatoric_nk_grid import nk_grid as nk
from aleatoric_nk_grid.config import NKGridConfig
from aleatoric_nk_grid.scheduler_cost import CostEstimator
from aleatoric_nk_grid.shared_queue import atomic_json, canonical, file_digest
from aleatoric_nk_grid.single_model_worker import json_result


MODELS = ('ols', 'ridge', 'lasso', 'random_forest', 'shallow_neural_network',
          'extra_trees', 'super_learner', 'xgboost', 'lightgbm')
CELLS = ((122, 47), (122, 400), (907, 261), (1165, 3400))


def now():
    return datetime.now(timezone.utc).isoformat()


def main(args):
    assert sys.platform == 'linux'
    assert args.output.resolve().is_relative_to(Path('/valhalla'))
    args.output.mkdir(parents=True, exist_ok=False)
    rows_path = args.output / 'measurements.jsonl'
    model_params = args.repo / 'FFCWS/model_params.yaml'
    report = dict(started_at_utc=now(), validation_only=True, synthetic_results=False,
        source_commit=args.source_commit, probe_sha256=file_digest(Path(__file__)),
        model_params_sha256=file_digest(model_params), schema_sha256=file_digest(args.schema),
        source_sha256={p.name: file_digest(p) for p in Path(nk.__file__).parent.glob('*.py')},
        hostname=socket.gethostname(), platform=platform.platform(),
        job=os.environ.get('SLURM_JOB_ID'), models=list(MODELS), cells=list(CELLS),
        seed=12345, draw=0, repeats=2, native_subprocess_models=sorted(nk.SERIAL_OUTER_MODELS),
        worker_threads=1, cross_task_cache=False,
        limitations='One server worker, four GPA cells, one seed/draw, two reverse-order repeats. '
                    'Ordering anchors only, not a whole-grid or 698-worker speedup claim. '
                    'No cache contention test, Slurm continuation integration or production results.')
    atomic_json(args.output / 'protocol.json', report)
    config = NKGridConfig(schema=args.schema, out=args.output / 'unused.csv', outcome='gpa',
        models=MODELS, seed=12345, test_size=.2, n_seeds=1, n_draws=1,
        n_sizes_n=3, n_sizes_k=4, max_n=1165, max_k=3400, batch_size=1, n_jobs=1,
        n_grid=(122, 907, 1165), k_grid=(47, 261, 400, 3400), model_params=model_params)
    references = {}
    groups = defaultdict(list)
    expanded_by_k = defaultdict(list)
    records = 0
    began = time.monotonic()
    try:
        with threadpool_limits(1), nk.NKGridExecutionSession.open_from_config(config) as session:
            report['session_startup_seconds'] = time.monotonic() - began
            report['algorithm_version'] = session.algorithm_version
            header = nk.public_result_columns('regression')
            # Warm the same native execution path without changing isolation or budgets.
            for model in MODELS:
                warm = session.run_cell_group(seed=12345, draw=0, n_samples=122,
                                              k_features=47, models=(model,))[0]
                assert warm['status'] == 'ok', (model, warm.get('error'))
            atomic_json(args.output / 'progress.json', {**report, 'phase': 'measuring', 'completed_fits': 0})
            design = [(n, k, m) for n, k in CELLS for m in MODELS]
            with rows_path.open('xb') as output:
                for repeat in range(2):
                    for n, k, model in (design if repeat == 0 else list(reversed(design))):
                        started = time.perf_counter()
                        row = session.run_cell_group(seed=12345, draw=0, n_samples=n,
                                                     k_features=k, models=(model,))[0]
                        elapsed = time.perf_counter() - started
                        assert row['status'] == 'ok', (n, k, model, row.get('error'))
                        assert math.isfinite(elapsed) and elapsed > 0
                        public = json_result(nk.project_public_result(row, header=header))
                        key = (n, k, model)
                        if key in references:
                            assert canonical(public) == canonical(references[key]), ('Repeat public result changed', key)
                        else:
                            references[key] = public
                        expanded = int(row['K_expanded'])
                        record = dict(repeat=repeat, N=n, K=k, model=model, K_expanded=expanded,
                            seconds=elapsed, fit_seconds=json_result(row.get('_fit_seconds')),
                            preprocess_seconds=json_result(row.get('_preprocess_seconds')),
                            slice_seconds=json_result(row.get('_slice_seconds')), public_result=public)
                        output.write(canonical(record) + b'\n')
                        output.flush()
                        os.fsync(output.fileno())
                        groups[(model, n, k, expanded)].append(elapsed)
                        expanded_by_k[str(k)].append(expanded)
                        records += 1
                        progress = dict(phase='measuring', completed_fits=records, total_fits=72,
                                        last_key=list(key), last_seconds=elapsed, updated_at_utc=now())
                        atomic_json(args.output / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
        assert records == 72 and len(references) == len(groups) == 36
        samples = [dict(model=m, N=n, K=k, K_expanded=p, seconds=statistics.median(values),
                        observations=len(values), seconds_min=min(values), seconds_max=max(values))
                   for (m, n, k, p), values in sorted(groups.items())]
        assert all(sample['observations'] == 2 for sample in samples)
        profile = dict(format='model-cell-cost-v1', evidence={
            'kind': 'Discoverer native single-model full-budget elapsed durations',
            'job': report['job'], 'source_commit': args.source_commit,
            'measurements_sha256': file_digest(rows_path),
            'algorithm_version': report['algorithm_version'], 'seed': 12345, 'draw': 0,
            'native_subprocess_models': report['native_subprocess_models'],
            'limitations': report['limitations'],
            'expanded_by_k_method': 'Median observed expanded width at each measured K; unmeasured K uses raw K.'},
            expanded_by_k={k: statistics.median(v) for k, v in expanded_by_k.items()}, samples=samples)
        estimator = CostEstimator(profile=profile)
        estimates = [estimator.estimate(model, n, k) for model in MODELS for n, k in CELLS]
        assert all(math.isfinite(value) and value > 0 for value in estimates)
        atomic_json(args.output / 'cost-profile.json', profile)
        report.update(passed=True, completed_at_utc=now(), measured_fits=records, warmup_fits=9,
            profile_samples=len(samples), repeat_public_results_equal=True,
            profile_sha256=file_digest(args.output / 'cost-profile.json'),
            measurements_sha256=file_digest(rows_path), elapsed_seconds=time.monotonic() - began,
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            native_children_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
            memory_caveat='Separate process/child high-water marks, not simultaneous aggregate RSS.')
        atomic_json(args.output / 'report.json', report)
        atomic_json(args.output / 'progress.json', {**report, 'phase': 'complete'})
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        atomic_json(args.output / 'progress.json', {**report, 'phase': 'failed', 'passed': False,
                    'completed_fits': records, 'error': repr(exc), 'updated_at_utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'schema', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    main(parser.parse_args())
