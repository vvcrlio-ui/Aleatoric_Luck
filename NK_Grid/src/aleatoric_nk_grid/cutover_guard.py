"""Read-only cutover decision. This module never stops or cancels a job.

The operator must collect a fresh exhaustive job observation while holding the
old run's existing control lock and recheck it immediately before any mutation.
Passing this predicate alone is not an atomic Slurm transaction or authorization
to signal an active worker.
"""
from datetime import datetime, timedelta


def blockers(state, coverage, jobs, *, now):
    reasons = []
    if now.utcoffset() is None:
        raise ValueError('Timezone-aware current time required')
    from zoneinfo import ZoneInfo
    if now.astimezone(ZoneInfo(state['timezone'])).date().isoformat() < state['not_before_local_date']:
        reasons.append('not_before_date')
    if state.get('deployment_ready') is not True:
        reasons.append('deployment_not_ready')
    if state.get('production_switched'):
        reasons.append('already_switched')
    for label, observation in (('coverage',coverage),('jobs',jobs)):
        if observation.get('run_id') != state['old_run_id'] or observation.get('analysis_id') != state['old_analysis_id']:
            reasons.append(label + '_identity')
        try:
            captured = datetime.fromisoformat(observation['captured_at'])
            maximum_age = 600 if label == 'coverage' else 120
            if captured.utcoffset() is None or not timedelta(0) <= now-captured <= timedelta(seconds=maximum_age):
                reasons.append(label + '_stale')
        except (ValueError,TypeError,KeyError):
            reasons.append(label + '_time_unknown')
    expected = state['expected_per_booster']
    if (coverage.get('verified_full_unique_design') is not True or
        any(coverage.get('valid_by_model',{}).get(m) != expected for m in ('xgboost','lightgbm')) or
        any(coverage.get('failed_by_model',{}).get(m, -1) != 0 for m in ('xgboost','lightgbm'))):
        reasons.append('boosting_incomplete_or_unverified')
    if (jobs.get('query_ok') is not True or jobs.get('exhaustive_old_run_scope') is not True or
        jobs.get('control_lock_held') is not True or not isinstance(jobs.get('jobs'), list)):
        reasons.append('job_observation_uncertain')
    else:
        terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
        for job in jobs['jobs']:
            status = job.get('state')
            if status == 'PENDING' and job.get('never_started') is True:
                continue
            if status not in terminal:
                reasons.append('active_or_unknown_job:' + str(job.get('job_id')))
    return reasons
