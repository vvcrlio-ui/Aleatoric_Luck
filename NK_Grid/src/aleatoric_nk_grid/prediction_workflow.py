"""Frozen, plan-wide base prediction -> verification -> SL workflow.

The direct protocol-2 dispatcher remains the transport. This module only adds
scientific task identities, phase queues and a durable all-panel barrier. It
never calls a model's fit method and never repairs missing caches by training.
"""
from array import array
from bisect import bisect_right
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import uuid

from .pending_resume import Design
from .shared_queue import (ModelTask, QueueError, atomic_json, canonical, digest,
                           file_digest, file_lock, transport_manifest)
from .result_migration import validate_scientific_result

FORMAT = 'prediction-workflow-v1'
STATES = ('PLANNED', 'BASE_RUNNING', 'BASE_VERIFYING', 'SL_READY', 'SL_RUNNING',
          'FINAL_VERIFYING', 'COMPLETE', 'REPAIR_REQUIRED')


def read(path):
    return json.loads(Path(path).read_bytes())


def enabled(plan):
    return plan.get('prediction_workflow') is not None


def reject_unphased_cache(spec):
    """Legacy queue/planning/publication cannot honor required prediction data.

    Numerical session capture is a separate API and does not pass this queue
    boundary. Never silently turn an enabled persistence contract into scores.
    """
    cache = spec.get('prediction_cache') or {}
    execution = spec.get('execution') or {}
    if (not isinstance(cache, dict) or not isinstance(execution, dict)
            or cache.get('mode', 'off') != 'off' or cache.get('required') is True
            or execution.get('workflow') == 'base_then_sl'):
        raise QueueError('Enabled prediction cache requires a complete base_then_sl queue workflow; legacy score-only execution is unsupported')


def contract_from_config(config, plan):
    """Single-panel launcher adapter; joint plans concatenate explicit panels."""
    from .prediction_contract import normalize_prediction_options
    cache, execution = normalize_prediction_options(getattr(config, 'prediction_cache', None),
                                                     getattr(config, 'execution', None))
    if execution.get('workflow') != 'base_then_sl':
        if cache:
            raise QueueError('Cluster prediction cache requires base_then_sl; capture is a numerical-session API only')
        return None
    if cache.get('base_library') != 'standalone8-v1':
        raise QueueError('Shared two-phase training currently requires explicit standalone8-v1; legacy SL recipes are separate')
    base_models = ('ols', 'ridge', 'lasso', 'random_forest', 'extra_trees', 'lightgbm', 'xgboost', 'shallow_neural_network')
    requested = set(plan['cell_spec']['models']) - {'super_learner'}
    if requested != set(base_models):
        raise QueueError('standalone8-v1 requires the complete frozen eight-model library')
    pipelines = [{'pipeline_id': 'standalone8-v1/' + model, 'model': model,
                  'oof_folds': cache.get('oof_folds', 5), 'base_library_id': 'standalone8-v1',
                  'model_params': plan['cell_spec']['resolved_model_params'][model]} for model in base_models]
    variants = cache.get('variants')
    if variants is None:
        sl_params = plan['cell_spec']['resolved_model_params']['super_learner']
        default_combiner = ({'rule': 'nnls-intercept-v1'} if plan['task_kind'] == 'regression' else
            {'rule': 'logistic-v1', 'C': sl_params.get('C', 1.), 'max_iter': sl_params.get('max_iter', 500),
             'random_state_rule': 'cell-model-seed'})
        # OLS is excluded from the combination, not from the library: it keeps its
        # own independent column and OOF. Underdetermined OLS predicts far outside
        # the label range (GPA labels 1.0-4.0 against OLS predictions -779.8..760.9),
        # and NNLS already gives it zero weight in 65.8% of the measured pairs.
        # Evidence: docs/sl7-no-ols-60nodes-20260918 (20 seeds x 3 outcomes x 6 scales).
        variants = [{'variant_id': 'standalone8-sl7-v1', 'pipeline_ids': ['standalone8-v1/' + model
                     for model in ('ridge', 'lasso', 'random_forest', 'extra_trees',
                                   'xgboost', 'lightgbm', 'shallow_neural_network')],
                     'missing_policy': 'skip', 'combiner': default_combiner}]
    # Off by default: the original four-model SL is reproduced only when a run
    # explicitly asks for that control. It is four extra pipelines of full+OOF
    # training, not a disk write, and it cannot be added to a sealed cache later.
    if cache.get('store_reported_sl_holdout', False):
        task_kind = plan['task_kind']; formal_id = 'reported-sl4-' + task_kind + '-v1'
        params = plan['cell_spec']['resolved_model_params']['super_learner']
        if params.get('passthrough') or (task_kind == 'regression' and not params.get('positive', True)):
            raise QueueError('Formal cache-only SL supports frozen passthrough=false and positive regression')
        entries = ([('ridge', 'ridge'), ('extra_trees', 'extra_trees'), ('lightgbm', 'lightgbm'),
                    ('shallow_nn', 'shallow_neural_network')] if task_kind == 'regression' else
                   [('logistic', 'ols'), ('lightgbm', 'lightgbm'), ('extra_trees', 'extra_trees'),
                    ('shallow_nn', 'shallow_neural_network')])
        pipelines += [{'pipeline_id': formal_id + '/' + internal, 'model': model,
                       'oof_folds': params.get('cv', 5), 'base_library_id': formal_id,
                       'formal_params': params} for internal, model in entries]
        combiner = ({'rule': 'nnls-intercept-v1'} if task_kind == 'regression' else
                    {'rule': 'logistic-v1', 'C': params.get('C', 1.), 'max_iter': params.get('max_iter', 500),
                     'random_state_rule': 'cell-model-seed'})
        variants = [*variants, {'variant_id': formal_id, 'pipeline_ids': [formal_id + '/' + internal for internal, _ in entries],
                                'missing_policy': 'skip', 'combiner': combiner, 'reported_sl': True}]
    root = Path(plan['launch']['output']).resolve()
    contract = {'format': FORMAT, 'workflow': 'base_then_sl', 'barrier_scope': execution.get('barrier_scope'),
        'protocol_version': execution.get('protocol_version'), 'base_library_id': cache['base_library'],
        'cache_mode': cache.get('mode'), 'required': cache.get('required'), 'output_root': str(root),
        'cache_root': str(root / 'prediction-cache'),
        'storage': {key: cache[key] for key in ('shard_target_mib', 'quota_reserve_gb', 'file_reserve',
            'max_bytes', 'max_files', 'temporary_max_bytes', 'compression_level',
            'layout', 'compaction_max_bytes') if key in cache},
        'phase_round_limits': execution.get('phase_round_limits', {'base': plan['launch']['cluster']['rounds'], 'sl': 1}),
        'sl_resources': execution.get('sl_resources', {'worker_cap': 8, 'io_concurrency': 4,
            'memory': '2G', 'time_limit': '00:30:00'}),
        'panels': [{'panel_id': plan['launch']['panel'], 'cell_spec': plan['cell_spec'],
                    'task_kind': plan['task_kind'], 'pipelines': pipelines, 'variants': variants}]}
    if 'base_round_time_limits' in execution:
        contract['base_round_time_limits'] = execution['base_round_time_limits']
        validate_round_time_limits(contract, global_time_limit=plan['launch']['cluster']['time_limit'])
    return validate_contract(contract)


def joint_contract(contracts, *, output_root, phase_round_limits, sl_resources):
    """One scheduler/receipt for a joint submission, never independent barriers."""
    contracts = [validate_contract(c) for c in contracts]
    if not contracts: raise QueueError('Joint submission needs explicit panel contracts')
    first = contracts[0]
    for c in contracts[1:]:
        if any(c[k] != first[k] for k in ('base_library_id', 'cache_mode', 'required', 'protocol_version')):
            raise QueueError('Joint submission library/cache/protocol contracts differ')
        if c.get('base_round_time_limits') != first.get('base_round_time_limits'):
            raise QueueError('Joint panels require the same frozen base round time limits')
    root = Path(output_root).resolve()
    return validate_contract({**first, 'output_root': str(root), 'cache_root': str(root / 'prediction-cache'),
        'panels': [p for c in contracts for p in c['panels']],
        'phase_round_limits': phase_round_limits, 'sl_resources': sl_resources})


def validate_contract(contract):
    if (not isinstance(contract, dict) or contract.get('format') != FORMAT
            or contract.get('workflow') != 'base_then_sl'
            or contract.get('barrier_scope') != 'submission_plan'
            or contract.get('protocol_version') != 2):
        raise QueueError('Prediction workflow requires the frozen protocol-2 submission-plan barrier')
    if not isinstance(contract.get('base_library_id'), str) or not contract['base_library_id']:
        raise QueueError('Frozen base library identity is required')
    if contract.get('cache_mode') != 'holdout_oof' or contract.get('required') is not True:
        raise QueueError('Two-phase SL requires complete OOF and holdout caches')
    panels = contract.get('panels')
    if not isinstance(panels, list) or not panels:
        raise QueueError('Submission-plan panels must be explicit')
    seen = set()
    for panel in panels:
        panel_id = panel.get('panel_id')
        if not isinstance(panel_id, str) or not panel_id or panel_id in seen:
            raise QueueError('Submission-plan panel IDs must be nonempty and unique')
        seen.add(panel_id)
        if panel.get('task_kind') not in ('regression', 'classification'):
            raise QueueError('Unknown panel task kind')
        Design(panel['cell_spec'])
        pipelines = panel.get('pipelines', [])
        identifiers = [p.get('pipeline_id') for p in pipelines]
        if (not identifiers or len(set(identifiers)) != len(identifiers)
                or any(not isinstance(v, str) or not v for v in identifiers)):
            raise QueueError('Base pipeline IDs must be nonempty and unique')
        for pipeline in pipelines:
            if pipeline.get('model') == 'super_learner' or not pipeline.get('model'):
                raise QueueError('Base phase cannot contain full super_learner.fit tasks')
            if type(pipeline.get('oof_folds')) is not int or pipeline['oof_folds'] < 2:
                raise QueueError('Each base pipeline freezes its OOF folds')
        variants = panel.get('variants', [])
        ids = [v.get('variant_id') for v in variants]
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(v, str) or not v for v in ids):
            raise QueueError('SL variant IDs must be nonempty and unique')
        for variant in variants:
            columns = variant.get('pipeline_ids', [])
            if not columns or len(columns) != len(set(columns)) or set(columns) - set(identifiers):
                raise QueueError('SL variant must name exact, ordered base pipeline columns')
            if variant.get('missing_policy') not in ('fail', 'skip'):
                raise QueueError('SL variant must freeze its missing-column policy')
    limits = contract.get('phase_round_limits', {})
    if set(limits) != {'base', 'sl'} or any(type(v) is not int or v < 1 for v in limits.values()):
        raise QueueError('Freeze separate positive base and SL round limits')
    validate_round_time_limits(contract)
    resources = contract.get('sl_resources', {})
    for key in ('worker_cap', 'io_concurrency'):
        if type(resources.get(key)) is not int or resources[key] < 1:
            raise QueueError('SL resources require explicit worker and I/O concurrency caps')
    if not resources.get('memory') or not resources.get('time_limit'):
        raise QueueError('SL resources require separate memory and wall-time requests')
    if not isinstance(contract.get('cache_root'), str) or not contract['cache_root']:
        raise QueueError('Cache root must be explicit and independent of checkpoints')
    if not isinstance(contract.get('output_root'), str) or not contract['output_root']:
        raise QueueError('Workflow output root is required')
    return contract


def _round_time_seconds(value):
    if not isinstance(value, str):
        raise QueueError('Base round time limits must use HH:MM:SS or D-HH:MM:SS')
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)', value)
    if not match:
        raise QueueError('Base round time limits must use HH:MM:SS or D-HH:MM:SS')
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0 or (match.group(1) is not None and hours >= 24):
        raise QueueError('Base round time limits must be positive valid Slurm durations')
    return total


def validate_round_time_limits(contract, *, global_time_limit=None):
    """Optional frozen first-short/later-long allocation durations.

    This does not add rounds or alter CPU budgets. Live resource admission still
    clips each request to current account/QoS/partition time limits.
    """
    if 'base_round_time_limits' not in contract:
        return None
    values = contract['base_round_time_limits']
    if (not isinstance(values, list) or len(values) != contract.get('phase_round_limits', {}).get('base')):
        raise QueueError('base_round_time_limits must have one entry per frozen base round')
    seconds = [_round_time_seconds(value) for value in values]
    if global_time_limit is not None:
        maximum = _round_time_seconds(global_time_limit)
        if any(value > maximum for value in seconds):
            raise QueueError('Base round time limit exceeds the frozen global time limit')
    return values


@dataclass(frozen=True)
class PredictionTask(ModelTask):
    phase: str
    panel_id: str
    pipeline_id: str
    variant_id: str
    base_library_id: str

    def __post_init__(self):
        super().__post_init__()
        if self.phase not in ('base', 'sl') or not self.panel_id or not self.base_library_id:
            raise QueueError('Invalid prediction task identity')
        if ((self.phase == 'base' and (not self.pipeline_id or self.variant_id or self.model == 'super_learner'))
                or (self.phase == 'sl' and (self.pipeline_id or not self.variant_id or self.model != 'super_learner'))):
            raise QueueError('Prediction task phase/pipeline/variant mismatch')

    @property
    def cell(self):
        return digest([self.panel_id, self.seed, self.draw, self.N, self.K])


class PredictionDesign:
    """Compact concatenation of all panels; no per-cell Python task table."""
    def __init__(self, contract, phase):
        self.contract = validate_contract(contract)
        if phase not in ('base', 'sl'): raise QueueError('Unknown prediction phase')
        self.phase = phase
        self.panels = {}; self.sections = []; self.starts = []; self.count = 0
        for panel in contract['panels']:
            definitions = panel['pipelines' if phase == 'base' else 'variants']
            field = 'pipeline_id' if phase == 'base' else 'variant_id'
            spec = {**panel['cell_spec'], 'models': [d[field] for d in definitions]}
            design = Design(spec)
            self.starts.append(self.count)
            entry = (self.count, design, panel, {d[field]: d for d in definitions})
            self.sections.append(entry); self.panels[panel['panel_id']] = entry
            self.count += design.count
        if self.count >= 2**32: raise QueueError('Prediction phase exceeds uint32 ordinal range')
        self.bits = bytearray((self.count + 7) // 8)

    def contains(self, ordinal):
        return bool(self.bits[ordinal // 8] & (1 << (ordinal % 8)))

    def mark(self, ordinal):
        self.bits[ordinal // 8] |= 1 << (ordinal % 8)

    def task_at(self, ordinal):
        if type(ordinal) is not int or not 0 <= ordinal < self.count:
            raise QueueError('Prediction ordinal outside design')
        start, design, panel, definitions = self.sections[bisect_right(self.starts, ordinal) - 1]
        group, mi = divmod(ordinal - start, len(design.models))
        group, ri = divmod(group, len(design.repeats))
        ki, ni = divmod(group, len(design.ns))
        identifier = design.models[mi]; definition = definitions[identifier]
        seed, draw = design.repeats[ri]
        return PredictionTask(seed, draw, design.ns[ni], design.ks[ki],
            definition['model'] if self.phase == 'base' else 'super_learner', self.phase,
            panel['panel_id'], identifier if self.phase == 'base' else '',
            identifier if self.phase == 'sl' else '', self.contract['base_library_id'])

    def ordinal(self, row):
        panel_id = row.get('panel_id')
        if panel_id not in self.panels: raise QueueError('Result outside submission-plan panels')
        start, design, _, _ = self.panels[panel_id]
        model_key = row.get('pipeline_id' if self.phase == 'base' else 'variant_id')
        ordinal = start + design.ordinal({**row, 'model': model_key})
        expected = asdict(self.task_at(ordinal))
        if any(str(row.get(k)) != str(v) for k, v in expected.items()):
            raise QueueError('Result phase/pipeline/variant identity changed')
        return ordinal

    def groups(self):
        for start, design, _, _ in self.sections:
            for ki, k in enumerate(design.ks):
                for ni, n in enumerate(design.ns):
                    for mi in range(len(design.models)):
                        base = start + (ki * len(design.ns) + ni) * len(design.repeats) * len(design.models) + mi
                        yield self.task_at(base), range(base, base + len(design.repeats) * len(design.models), len(design.models))


def design_for(identity):
    workflow = identity.get('prediction_workflow')
    if workflow is not None:
        if not isinstance(workflow, dict) or not {'contract', 'phase'} <= workflow.keys():
            raise QueueError('Prediction queue requires a complete frozen phase workflow')
        return PredictionDesign(workflow['contract'], workflow['phase'])
    reject_unphased_cache(identity['cell_spec'])
    return Design(identity['cell_spec'])


def cost_identity(task, contract):
    if not isinstance(task, PredictionTask): return None
    panel = next(p for p in contract['panels'] if p['panel_id'] == task.panel_id)
    definition = next(d for d in panel['pipelines' if task.phase == 'base' else 'variants']
                      if d['pipeline_id' if task.phase == 'base' else 'variant_id']
                      == (task.pipeline_id or task.variant_id))
    result = {'phase': task.phase, 'pipeline_id': task.pipeline_id, 'variant_id': task.variant_id,
            'base_library_id': task.base_library_id, 'oof_folds': definition.get('oof_folds', 0),
            'cache_mode': contract['cache_mode'], 'training_identity': digest(definition),
            'cell_spec_sha256': digest(panel['cell_spec'])}
    if contract.get('storage', {}).get('layout') == 'shared-v1':
        result.pop('cell_spec_sha256')
        result['training_context_sha256'] = cost_training_context(panel['cell_spec'])
    return result


def cost_training_context(spec):
    """Timing compatibility only; never permits cross-run prediction reuse.

    Drop run paths, repeat/grid selection and scheduler/storage settings. Keep
    raw input hashes, schema, every numerical/environment and retry setting.
    N/K and expanded K remain separate measured profile dimensions.
    """
    ignored = {'git_commit', 'require_clean_worktree', 'experiment_id', 'panel_id', 'preset',
               'schema_locator', 'model_params_locator', 'prediction_cache', 'execution',
               'resolved_repeat_plan', 'resolved_n_grid', 'resolved_k_grid', 'execution_groups'}
    value = {k: v for k, v in spec.items() if k not in ignored}
    value['input_provenance'] = {k: {'sha256': v['sha256']} for k, v in spec.get('input_provenance', {}).items()}
    return digest(value)


def phase_identity(plan, phase):
    contract = validate_contract(plan['prediction_workflow'])
    identity = {'cell_spec': plan['cell_spec'], 'plan_sha256': digest(plan),
                'runtime_sha256': plan['runtime_sha256'],
                'prediction_workflow': {'contract': contract, 'phase': phase}}
    if phase == 'sl':
        receipt = verify_base_receipt(plan, verify_files=False)
        identity['prediction_workflow']['base_receipt_sha256'] = digest(receipt)
    return identity


def task_kind(identity, row):
    workflow = identity.get('prediction_workflow')
    if workflow:
        return next(p['task_kind'] for p in workflow['contract']['panels'] if p['panel_id'] == row['panel_id'])
    return identity.get('task_kind', 'regression')


def validate_sample_maps(contract, record, row, *, sealed=False):
    """Verify exact content-addressed order/label maps before score ACK.

    Map identities are derived from the already-frozen ordered-array descriptors.
    The codec checks record and raw-block hashes during this single read; we do
    not hash feature arrays again merely to reconstruct their existing identity.
    """
    import numpy as np
    from .prediction_cache import cache_identity, verify_reference
    refs = record['metadata'].get('sample_map_refs')
    cell = record['identity'].get('cell_identity', {})
    ordered = cell.get('ordered_arrays') if isinstance(cell, dict) else None
    groups = {'training': ('train_ids', 'train_positions', 'y_train', 'oof_fold'),
              'evaluation': ('holdout_ids', 'holdout_positions', 'y_holdout'),
              'features': ('feature_names', 'source_names')}
    if not isinstance(refs, dict) or not set(groups) <= refs.keys():
        raise QueueError('Required training/evaluation/features sample maps missing')
    if not isinstance(ordered, dict):
        raise QueueError('Prediction cache lacks frozen ordered-array descriptors')
    values = {}
    for kind, names in groups.items():
        if not isinstance(refs[kind], dict):
            raise QueueError('Required sample-map reference is malformed')
        required = set(names) - {'oof_fold'}
        if not required <= ordered.keys():
            raise QueueError('Frozen prediction identity lacks required sample/feature arrays')
        present = {name: ordered[name] for name in names if name in ordered}
        expected = {'sample_map_content': cache_identity({'metadata': {'kind': kind}, 'arrays': present})}
        mapping = verify_reference(Path(contract['cache_root']), refs[kind],
                                   expected_identity=expected, require_sealed=sealed)
        if mapping['kind'] != 'sample_map' or mapping['status'] != 'ok':
            raise QueueError('Required sample-map reference has wrong record kind/status')
        if set(mapping['arrays']) != set(present):
            raise QueueError('Sample-map arrays differ from frozen ordered identity')
        for name, value in mapping['arrays'].items():
            descriptor = present[name]
            if (not isinstance(descriptor, dict) or value.ndim != 1 or list(value.shape) != descriptor.get('shape')
                    or value.dtype.str != descriptor.get('dtype')):
                raise QueueError('Sample-map array shape/dtype differs from frozen identity')
            values[name] = value
    n = int(row['N']); holdout = len(values['y_holdout'])
    if any(values[name].shape != (n,) for name in ('train_ids', 'train_positions', 'y_train')):
        raise QueueError('Training sample/label order length differs from N')
    if holdout < 1 or any(values[name].shape != (holdout,) for name in ('holdout_ids', 'holdout_positions')):
        raise QueueError('Evaluation sample/label order lengths differ')
    for name in ('y_train', 'y_holdout'):
        value = values[name]
        if value.dtype.kind != 'f' or value.dtype.itemsize != 8 or not np.isfinite(value).all():
            raise QueueError('Sample-map labels must be finite float64')
    if len(values['source_names']) != int(row['K']) or not len(values['feature_names']):
        raise QueueError('Source/feature order length differs from frozen task')
    if any(values[name].dtype.kind not in ('U', 'S') for name in ('feature_names', 'source_names')):
        raise QueueError('Feature/source names require exact string order')
    if 'oof_fold' in values:
        fold = values['oof_fold']
        if fold.shape != (n,) or fold.dtype.kind != 'i' or fold.dtype.itemsize != 8:
            raise QueueError('Training-map fold assignments require length-N int64')
    prediction = record['arrays'].get('holdout_prediction')
    if prediction is not None and prediction.shape != (holdout,):
        raise QueueError('Holdout prediction length differs from evaluation order')
    return values


def validate_result_cache(identity, row, *, sealed=False):
    """Read durable numeric records before ACK/coverage; never trust flags alone."""
    workflow = identity.get('prediction_workflow')
    if not workflow or row.get('status') == 'failed': return
    from .prediction_cache import verify_reference
    contract = workflow['contract']
    reference = row.get('prediction_cache_ref')
    if not isinstance(reference, dict): raise QueueError('Required prediction cache reference missing')
    record = verify_reference(Path(contract['cache_root']), reference, require_sealed=sealed)
    metadata = record['metadata']
    expected = {key: row.get(key) for key in PredictionTask.__dataclass_fields__}
    stored = metadata.get('task', {})
    if any(str(stored.get(k)) != str(v) for k, v in expected.items()):
        raise QueueError('Prediction cache task identity mismatch')
    if record['status'] not in ({'ok', 'nonconverged'} if row['status'] == 'ok' else {row['status']}):
        raise QueueError('Prediction cache status differs from result')
    panel = next(p for p in contract['panels'] if p['panel_id'] == row['panel_id'])
    definition = next(d for d in panel['pipelines' if workflow['phase'] == 'base' else 'variants']
        if d['pipeline_id' if workflow['phase'] == 'base' else 'variant_id'] == (row['pipeline_id'] or row['variant_id']))
    if (metadata.get('input_spec_sha256') != digest(panel['cell_spec'])
            or metadata.get('pipeline_sha256') != digest(definition)):
        raise QueueError('Prediction cache frozen input/pipeline provenance differs')
    sample_arrays = validate_sample_maps(contract, record, row, sealed=sealed)
    if row['status'] == 'skipped':
        if not metadata.get('reason'): raise QueueError('Cache skip requires a durable reason')
    else:
        arrays = record['arrays']
        legal_oof_skip = (workflow['phase'] == 'base' and metadata.get('oof_status') == 'skipped'
            and bool(metadata.get('oof_reason')) and all(v['missing_policy'] == 'skip'
                for v in panel['variants'] if row['pipeline_id'] in v['pipeline_ids']))
        required = ('oof_prediction', 'holdout_prediction') if workflow['phase'] == 'base' and not legal_oof_skip else ('holdout_prediction',)
        if any(name not in arrays for name in required): raise QueueError('Required prediction array missing')
        if workflow['phase'] == 'base' and not legal_oof_skip:
            import numpy as np
            if arrays['oof_prediction'].shape != (row['N'],):
                raise QueueError('OOF coverage length differs from training size')
            fold = arrays.get('oof_fold'); mapped_fold = sample_arrays.get('oof_fold')
            if (fold is None or mapped_fold is None or fold.shape != (row['N'],)
                    or fold.dtype.kind != 'i' or fold.dtype.itemsize != 8
                    or not np.array_equal(fold, mapped_fold)):
                raise QueueError('OOF fold assignments differ from the exact training sample map')
            labels = np.unique(fold)
            if (len(labels) < 2 or len(labels) > definition['oof_folds']
                    or not np.array_equal(labels, np.arange(len(labels)))):
                raise QueueError('OOF folds do not provide complete nonnegative fold coverage')
    return record


def scan(plan, phase, rounds, accept=lambda row: None, *, sealed=False):
    """Stopped generations only; all panel identities share one coverage bitmap."""
    from contextlib import ExitStack
    identity = phase_identity(plan, phase)
    design = PredictionDesign(plan['prediction_workflow'], phase); sources = []
    with ExitStack() as locks:
        for directory in map(Path, rounds):
            locks.enter_context(file_lock(directory / 'dispatcher.lock'))
            manifest = read(directory / 'manifest.json'); qid = digest(manifest)
            if (manifest['identity'] != identity or manifest['sources'] != sources
                    or read(directory / 'queue-id.json')['queue_id'] != qid
                    or file_digest(directory / 'remaining.u32') != manifest['remaining_sha256']):
                raise QueueError('Prediction round identity changed')
            order = array('I'); order.frombytes((directory / 'remaining.u32').read_bytes())
            if sys.byteorder != 'little': order.byteswap()
            if len(order) != manifest['count']: raise QueueError('Prediction round count changed')
            members = bytearray(len(design.bits))
            for ordinal in order:
                if ordinal >= design.count or design.contains(ordinal) or members[ordinal // 8] & (1 << (ordinal % 8)):
                    raise QueueError('Duplicate/outside prediction round task')
                members[ordinal // 8] |= 1 << (ordinal % 8)
            path = directory / 'results.jsonl'
            if not path.exists():
                sources.append({'root': str(directory), 'queue_id': qid, 'results_sha256': None}); continue
            size = path.stat().st_size
            with path.open('rb') as handle:
                while line := handle.readline(2 * 1024 * 1024 + 1):
                    if len(line) > 2 * 1024 * 1024: raise QueueError('Oversized prediction result')
                    if not line.endswith(b'\n'): break
                    entry = json.loads(line); row = entry['result']; ordinal = design.ordinal(row)
                    if (entry['origin']['queue_id'] != qid or entry['task_id'] != design.task_at(ordinal).id
                            or not members[ordinal // 8] & (1 << (ordinal % 8))):
                        raise QueueError('Prediction result provenance mismatch')
                    if not validate_scientific_result(row, task_kind=task_kind(identity, row)): continue
                    panel_spec = design.panels[row['panel_id']][2]['cell_spec']
                    if row.get('algorithm_version') != panel_spec['algorithm_version']:
                        raise QueueError('Prediction algorithm changed')
                    validate_result_cache(identity, row, sealed=sealed)
                    if design.contains(ordinal): raise QueueError('Duplicate prediction success')
                    design.mark(ordinal); accept(row)
            if path.stat().st_size != size: raise QueueError('Stopped prediction journal changed')
            sources.append({'root': str(directory), 'queue_id': qid, 'results_sha256': file_digest(path)})
    return design, sources


def prepare_round(plan, phase, rounds, directory, *, cost_profile=None):
    from .scheduler_cost import CostEstimator
    design, sources = scan(plan, phase, rounds)
    done = sum(b.bit_count() for b in design.bits)
    if done == design.count: return {'done': done, 'remaining': 0, 'phase': phase}
    directory = Path(directory); identity = phase_identity(plan, phase)
    estimator = CostEstimator(profile=cost_profile); ordered = []; coverage = []
    work = 0.; fully_priced = True
    for task, ordinals in design.groups():
        remaining = sum(not design.contains(i) for i in ordinals)
        if not remaining: continue
        ci = cost_identity(task, design.contract)
        mean = estimator.mean_seconds(task.model, task.N, task.K, identity=ci)
        price = estimator.batch_seconds(task.model, task.N, task.K, identity=ci)
        coverage.append({'phase': phase, 'panel_id': task.panel_id, 'pipeline_id': task.pipeline_id,
                         'variant_id': task.variant_id, 'N': task.N, 'K': task.K, 'priced': price is not None})
        if mean is None: fully_priced = False
        else: work += mean * remaining
        # Unknown SL work has no historical super_learner 40 weight.
        ordering = mean if mean is not None else (1 if phase == 'sl' else task.N * task.K ** .5)
        ordered.append((ordering, ordinals))
    report = {'done': done, 'remaining': design.count - done, 'phase': phase,
              'work_seconds': work if fully_priced else None,
              'cost_profile_coverage': {'total_groups': len(coverage),
                  'priced_groups': sum(g['priced'] for g in coverage), 'groups': coverage},
              'cost_profile_sha256': digest(cost_profile) if cost_profile is not None else None}
    if directory.exists():
        saved = read(directory / 'prepared.json'); manifest = read(directory / 'manifest.json')
        if (manifest['identity'] != identity or manifest['sources'] != sources
                or saved['done'] != done or saved['remaining'] != report['remaining']
                or read(directory / 'queue-id.json')['queue_id'] != digest(manifest)
                or file_digest(directory / 'remaining.u32') != manifest['remaining_sha256']):
            raise QueueError('Prepared prediction round changed')
        recovery = manifest.get('prediction_recovery')
        if recovery and recovery.get('format') == 'prediction-recovery-v1' and (Path(recovery['path']).resolve() != (directory / 'recovery.sqlite').resolve()
                         or recovery['sha256'] != file_digest(directory / 'recovery.sqlite')):
            raise QueueError('Prepared prediction recovery index changed')
        return {**report, 'queue_id': digest(manifest)}
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name('.' + directory.name + '-' + uuid.uuid4().hex); temporary.mkdir()
    with (temporary / 'remaining.u32').open('xb') as handle:
        for _, ordinals in sorted(ordered, key=lambda item: -item[0]):
            chunk = array('I', (i for i in ordinals if not design.contains(i)))
            if sys.byteorder != 'little': chunk.byteswap()
            handle.write(chunk.tobytes())
        handle.flush(); os.fsync(handle.fileno())
    manifest = {'format': 'direct-success-bitmap-v1', 'identity': identity,
        'count': report['remaining'], **transport_manifest(), 'max_attempts': 5,
        'sources': sources, 'remaining_sha256': file_digest(temporary / 'remaining.u32'), 'fresh': True}
    manifest['prediction_recovery'] = build_recovery_index(plan, temporary / 'recovery.sqlite',
                                                          published_path=directory / 'recovery.sqlite')
    report['queue_id'] = digest(manifest)
    atomic_json(temporary / 'manifest.json', manifest)
    atomic_json(temporary / 'queue-id.json', {'queue_id': report['queue_id']})
    atomic_json(temporary / 'prepared.json', report); temporary.replace(directory)
    return report


def build_recovery_index(plan, path, *, published_path=None):
    """Freeze exact orphan-record lookups for reassignment to a NEW worker.

    Called by the controller only after every earlier allocation is terminal.
    Sealed shard indexes, not prediction arrays, are streamed into a controller-
    owned SQLite database. Only unindexed abandoned shard prefixes are decoded
    during repair. The database remains immutable for the whole next allocation.
    """
    from .prediction_cache import seal_stopped_writers, safe_cache_path
    if plan['prediction_workflow'].get('storage', {}).get('layout') == 'shared-v1':
        from .prediction_recovery import build_incremental
        return build_incremental(plan)
    contract = plan['prediction_workflow']; path = Path(path)
    roots = {'main': Path(contract['cache_root']),
             'fold': Path(contract['output_root']) / 'prediction-training-checkpoints'}
    count = 0
    contains_legacy_fold_identities = False
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('CREATE TABLE records (root_kind TEXT NOT NULL, identity TEXT NOT NULL, '
            'content_sha256 TEXT, reference TEXT NOT NULL, reference_sha256 TEXT NOT NULL, '
            'PRIMARY KEY(root_kind,identity,reference_sha256)) WITHOUT ROWID')
        for kind, root in roots.items():
            if not root.exists(): continue
            seal_stopped_writers(root, writer_revoked=True)
            # Each index covers a bounded shard. No list of all records/files is
            # materialized and no expensive full-data hash is repeated here.
            for index_path in (root / 'indexes').glob('*.json'):
                index = read(index_path)
                if not index.get('sealed'): raise QueueError('Stopped prediction shard index is not sealed')
                relative = index.get('path', '')
                if not relative.startswith(('shards/', 'meta-results/')): continue
                source = safe_cache_path(root, relative)
                if source.stat().st_size != index.get('bytes'):
                    raise QueueError('Stopped prediction shard size changed')
                for ref in index.get('records', []):
                    if ref.get('path') != relative or not ref.get('identity') or not ref.get('sha256'):
                        raise QueueError('Malformed stopped prediction recovery index')
                    if kind == 'fold' and ref.get('fold_identity_version') != 2:
                        contains_legacy_fold_identities = True
                    # The consumer validates frame integrity and content equality
                    # for its exact identity before reusing anything. Normal old
                    # index records need not carry an already-decoded content hash.
                    connection.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)',
                        (kind, ref['identity'], ref.get('content_sha256'), canonical(ref).decode(), ref['sha256']))
                    count += 1
        connection.commit()
    with path.open('r+b') as handle: os.fsync(handle.fileno())
    return {'format': 'prediction-recovery-v1', 'path': str(Path(published_path or path).resolve()),
            'sha256': file_digest(path), 'plan_sha256': digest(plan),
            'workflow_sha256': digest(contract), 'records': count,
            'contains_legacy_fold_identities': contains_legacy_fold_identities}


def verify_base_receipt(plan, *, verify_files=True):
    """Confirm the sealed barrier. File bytes are reread only when asked.

    The receipt's own identity is cheap and always checked. Rehashing the task
    index and every cache index is not: at production repeat counts the index
    alone is tens of gigabytes, and the controller calls this on every advance.
    Those files are sealed under an exclusive lock and cannot change while the
    run owns them, so callers pass ``verify_files`` for the transitions that
    actually need it - sealing, controller restart, and consuming a cache this
    run did not produce.
    """
    path = Path(plan['launch']['output']) / 'base-verified.json'
    if not path.exists(): raise QueueError('Base verification barrier is not sealed')
    receipt = read(path)
    if (receipt.get('plan_sha256') != digest(plan) or receipt.get('complete') is not True
            or receipt.get('workflow_sha256') != digest(plan['prediction_workflow'])):
        raise QueueError('Base verification receipt identity changed')
    if verify_files and receipt.get('records_index_sha256') != file_digest(Path(plan['launch']['output']) / 'base-records.sqlite'):
        raise QueueError('Verified base task-reference index changed; explicit repair required')
    from .prediction_evidence import verify_cache_evidence
    for item in receipt.get('cache_indexes', []) if verify_files else ():
        verify_cache_evidence(Path(plan['prediction_workflow']['cache_root']), item)
    return receipt


def seal_base(plan, rounds):
    from .prediction_cache import seal_stopped_writers, encode_index_json
    root = Path(plan['launch']['output']); path = root / 'base-verified.json'
    if path.exists(): return verify_base_receipt(plan)
    if plan['prediction_workflow'].get('storage', {}).get('layout') == 'shared-v1':
        from .prediction_layout import compact_stopped
        cache = Path(plan['prediction_workflow']['cache_root'])
        seal_stopped_writers(cache, writer_revoked=True)
        storage = plan['prediction_workflow']['storage']
        compact_stopped(cache, temporary_byte_limit=min(storage.get('compaction_max_bytes', 512 * 1024**2),
            storage.get('temporary_max_bytes', 512 * 1024**2)), readers_drained=True, directories=('shards',),
            target_bytes=storage.get('shard_target_mib', 128) * 1024**2)
    temporary = root / ('base-records-' + uuid.uuid4().hex + '.sqlite.tmp')
    statuses = Counter(); record_hash = __import__('hashlib').sha256()
    design = PredictionDesign(plan['prediction_workflow'], 'base')
    with closing(sqlite3.connect(temporary)) as connection:
        connection.execute('PRAGMA synchronous=FULL')
        shared = plan['prediction_workflow'].get('storage', {}).get('layout') == 'shared-v1'
        # Large compressed rows belong in a rowid table: WITHOUT ROWID spills
        # payloads around half a SQLite page and can increase space substantially.
        connection.execute('CREATE TABLE records(task_id TEXT PRIMARY KEY, reference TEXT NOT NULL, row BLOB NOT NULL)')
        def accept(row):
            statuses[row['status']] += 1
            reference = row['prediction_cache_ref']
            record_hash.update(canonical([design.task_at(design.ordinal(row)).id, reference['sha256']]) + b'\n')
            connection.execute('INSERT INTO records VALUES (?,?,?)',
                (design.task_at(design.ordinal(row)).id, canonical(reference).decode(),
                 encode_index_json({k: v for k, v in row.items() if k != 'prediction_cache_ref'}) if shared else canonical(row).decode()))
        completed, sources = scan(plan, 'base', rounds, accept)
        count = sum(b.bit_count() for b in completed.bits)
        if count != design.count: raise QueueError('All-panel base cache coverage incomplete')
        connection.commit()
        def refs():
            for (raw,) in connection.execute('SELECT reference FROM records ORDER BY task_id'):
                yield json.loads(raw)
        # Only accepted records and their maps are touched. Never recursively
        # scan TB of cache or truncate a writer that retains its exclusive lock.
        seal_stopped_writers(Path(plan['prediction_workflow']['cache_root']),
                             writer_revoked=True, references=refs())
        scan(plan, 'base', rounds, sealed=True)
        from .prediction_cache import verified_index_evidence
        evidence = verified_index_evidence(Path(plan['prediction_workflow']['cache_root']), refs())
        for item in evidence:
            stat = (Path(plan['prediction_workflow']['cache_root']) / item['path']).stat()
            item.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        # A later same-size rewrite must not be able to reuse a sealed timestamp,
        # or the receipt's timestamp fast path would skip its content check.
        if evidence:
            from .prediction_evidence import seal_mtime_barrier
            seal_mtime_barrier(Path(plan['prediction_workflow']['cache_root']),
                               max(item['mtime_ns'] for item in evidence))
    with temporary.open('r+b') as handle: os.fsync(handle.fileno())
    os.replace(temporary, root / 'base-records.sqlite')
    # Durable accepted rows are the authority. Unsubmitted tails cannot satisfy
    # coverage; generation leases cannot survive stopped, exclusively locked rounds.
    receipt = {'format': FORMAT, 'complete': True, 'plan_sha256': digest(plan),
        'workflow_sha256': digest(plan['prediction_workflow']), 'rows': count, 'expected_rows': design.count,
        'panels': [p['panel_id'] for p in plan['prediction_workflow']['panels']],
        'statuses': dict(statuses), 'sources': sources, 'cache_indexes': evidence,
        'source_record_hashes_sha256': record_hash.hexdigest(),
        'records_index_sha256': file_digest(root / 'base-records.sqlite'),
        'score_complete': True, 'prediction_cache_complete': True, 'oof_complete': True,
        'unfinished_leases': 0, 'unsubmitted_required_results': 0, 'unresolved_failures': 0}
    atomic_json(path, receipt)
    return receipt


def base_record_for_task(contract, task, *, connection=None):
    """O(log cells) lookup against the sealed, controller-owned reference index.

    The worker must call verify_base_receipt once before opening SL input. This
    helper does no scan, write, or fallback training.
    """
    if not isinstance(task, PredictionTask): task = PredictionTask(**task)
    if task.phase != 'base': raise QueueError('Base index requires an exact base task')
    root = Path(contract['output_root'])
    path = root / 'base-records.sqlite'
    uri = path.resolve().as_uri() + '?mode=ro&immutable=1'
    if connection is None:
        with closing(sqlite3.connect(uri, uri=True)) as owned:
            return base_record_for_task(contract, task, connection=owned)
    else:
        found = connection.execute('SELECT reference,row FROM records WHERE task_id=?', (task.id,)).fetchone()
    if found is None: raise QueueError('Required base record absent; explicit repair required')
    from .prediction_cache import decode_index_json
    reference, row = json.loads(found[0]), decode_index_json(found[1])
    row['prediction_cache_ref'] = reference
    return {'reference': reference, 'row': row}


def accepted_references(rounds):
    """Stream already-validated stopped journals with bounded record memory."""
    for directory in map(Path, rounds):
        source = directory / 'results.jsonl'
        if not source.exists(): continue
        with source.open('rb') as handle:
            while line := handle.readline(2 * 1024 * 1024 + 1):
                if len(line) > 2 * 1024 * 1024: raise QueueError('Oversized prediction result')
                if not line.endswith(b'\n'): break
                row = json.loads(line)['result']
                if row['status'] in ('ok', 'skipped'): yield row['prediction_cache_ref']


def finalize(plan, base_rounds, sl_rounds):
    from .prediction_cache import seal_stopped_writers
    root = Path(plan['launch']['output']); receipt_path = root / 'verified.json'
    base = verify_base_receipt(plan)
    if receipt_path.exists():
        receipt = read(receipt_path)
        if (receipt['plan_sha256'] != digest(plan) or receipt['base_verified_sha256'] != digest(base)
                or receipt['final_csv_sha256'] != file_digest(root / 'final.csv')):
            raise QueueError('Final prediction workflow publication changed')
        return receipt
    # Seal only source shards referenced by stopped SL journals, not unrelated writers.
    scan(plan, 'sl', sl_rounds)
    seal_stopped_writers(Path(plan['prediction_workflow']['cache_root']), writer_revoked=True,
                         references=accepted_references(sl_rounds))
    if plan['prediction_workflow'].get('storage', {}).get('layout') == 'shared-v1':
        from .prediction_layout import compact_stopped
        storage = plan['prediction_workflow']['storage']
        compact_stopped(plan['prediction_workflow']['cache_root'], temporary_byte_limit=min(
            storage.get('compaction_max_bytes', 512 * 1024**2), storage.get('temporary_max_bytes', 512 * 1024**2)),
            readers_drained=True, directories=('meta-results',), target_bytes=storage.get('shard_target_mib', 128) * 1024**2)
    columns = list(dict.fromkeys(['panel_id', 'phase', 'pipeline_id', 'variant_id', 'base_library_id'] + plan['public_columns']))
    temporary = root / ('final-' + uuid.uuid4().hex + '.tmp'); counts = {}
    with temporary.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n'); writer.writeheader()
        for phase, rounds in (('base', base_rounds), ('sl', sl_rounds)):
            design, _ = scan(plan, phase, rounds, lambda row: writer.writerow({k: row.get(k, '') for k in columns}), sealed=True)
            count = sum(b.bit_count() for b in design.bits)
            if count != design.count: raise QueueError('Prediction workflow final coverage incomplete')
            counts[phase] = count
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, root / 'final.csv')
    receipt = {'format': FORMAT, 'complete': True, 'phase_rows': counts, 'rows': sum(counts.values()),
        'plan_sha256': digest(plan), 'base_verified_sha256': digest(base),
        'final_csv_sha256': file_digest(root / 'final.csv'), 'score_complete': True,
        'prediction_cache_complete': True, 'oof_complete': True}
    atomic_json(receipt_path, receipt)
    # Phase journals hold provenance/refs needed for recovery and stay independent
    # of caches. New-mode checkpoint pruning must never remove prediction-cache.
    return receipt
