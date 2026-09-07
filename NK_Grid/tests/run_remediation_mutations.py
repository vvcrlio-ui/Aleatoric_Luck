"""Controlled temporary-source mutations, no edits to production or user data."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'NK_Grid/src/aleatoric_nk_grid'
mutations = [
 ('group_count', 'preprocessing.py', 'units = tuple(\n        SamplingUnit(name=source, groups=tuple(source_groups))', 'return tuple(SamplingUnit(name=g.name, groups=(g,)) for g in groups)\n    units = tuple(\n        SamplingUnit(name=source, groups=tuple(source_groups))', 'test_remediation_contracts.py', 'a1'),
 ('retain_fold_models', 'model_registry.py', 'curves.append(history["valid"]["rmse"])', 'self.retained_ = getattr(self, "retained_", []) + [booster]\n                curves.append(history["valid"]["rmse"])', 'test_remediation_boosting.py', 'b4'),
 ('average_fold_minima', 'model_registry.py', 'self.best_rounds_ = _select_cv_round(self.cv_curve_)', 'self.best_rounds_ = int(np.mean(np.argmin(np.asarray(curves), axis=1) + 1))', 'test_remediation_boosting.py', 'b1'),
 ('fixed_200', 'model_registry.py', 'self.batch_size = len(y) if policy == "full" else policy', 'self.batch_size = 200', 'test_remediation_batch.py', 'c1'),
 ('clone_loses_batch', 'model_registry.py', '    def fit(self, X, y, sample_weight=None):', '    def get_params(self, deep=True):\n        params = super().get_params(deep=deep)\n        params.pop("batch_size", None)\n        return params\n\n    def fit(self, X, y, sample_weight=None):', 'test_remediation_batch.py', 'c2'),
 ('full_N_oof', 'model_registry.py', '        self.model_ = StackingRegressor(\n            estimators=estimators,\n            final_estimator=LinearRegression(positive=self.positive),\n            cv=cv,', '        estimators = [(name, clone(estimator).fit(X, y)) for name, estimator in estimators]\n        self.model_ = StackingRegressor(\n            estimators=estimators,\n            final_estimator=LinearRegression(positive=self.positive),\n            cv="prefit",', 'test_remediation_folds.py', 'super_oof'),
 ('full_N_fill', 'fold_local.py', 'y = np.asarray(y)\n        self.alphas_', 'X = self.preprocessor.fit_transform(X)\n        y = np.asarray(y)\n        self.alphas_', 'test_remediation_folds.py', 'd6'),
 ('swapped_denominator', 'evaluation.py', '"skill_train_mean": 1 - mse / null if null > 0 else np.nan', '"skill_train_mean": 1 - mse / variance if variance > 0 else np.nan', 'test_remediation_metrics.py', 'e1'),
 ('clip_negative_score', 'evaluation.py', '"r2_test_mean": 1 - mse / variance if variance > 0 else np.nan', '"r2_test_mean": max(0, 1 - mse / variance) if variance > 0 else np.nan', 'test_remediation_metrics.py', 'e1'),
]
records=[]
mutations.append(('full_N_scale', 'fold_local.py',
    'process = make_pipeline(clone(self.preprocessor), StandardScaler()).fit(X.iloc[train])',
    'process = make_pipeline(clone(self.preprocessor), StandardScaler()).fit(X.iloc[train])\n            process.named_steps["standardscaler"].fit(clone(self.preprocessor).fit_transform(X))',
    'test_remediation_folds.py', 'd6'))
mutations.append(('skip_content_verification', 'execution_contract.py',
    '    if actual != expected_sha256:', '    if False:',
    'test_remediation_environment.py', 'f2'))
mutations.append(('count_only_completion', 'flat_task_table.py',
    'def _queue_missing_model_keys(connection: sqlite3.Connection) -> int:',
    'def _queue_missing_model_keys(connection: sqlite3.Connection) -> int:\n    return max(0, connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0] - connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0])',
    'test_remediation_environment.py', 'f6'))
mutations.append(('remove_execution_capacity', 'nk_grid.py',
    '        validate_size_grid((n_samples,), "N", len(indexes.train_index))', '',
    'test_remediation_contracts.py', 'portable_actual_method_gate'))
for name, filename, old, new, test, selection in mutations:
    with tempfile.TemporaryDirectory(prefix='nk-remediation-mutant-') as temporary:
        package=Path(temporary)/'aleatoric_nk_grid'; shutil.copytree(SOURCE,package,ignore=shutil.ignore_patterns('__pycache__'))
        path=package/filename; text=path.read_text(); assert old in text, name
        path.write_text(text.replace(old,new,1))
        bootstrap=Path(temporary)/'run.py'
        bootstrap.write_text('import sys,types,pytest\nfrom pathlib import Path\np=types.ModuleType("aleatoric_nk_grid");p.__path__=[str(Path(__file__).parent/"aleatoric_nk_grid")];sys.modules["aleatoric_nk_grid"]=p\nraise SystemExit(pytest.main(sys.argv[1:]))\n')
        result=subprocess.run([sys.executable,str(bootstrap),'-q',str(ROOT/'NK_Grid/tests'/test),'-k',selection],capture_output=True,text=True)
        # Collection/import errors do not count as killing a mutant.
        killed=result.returncode == 1 and ' failed' in result.stdout
        records.append({'mutation':name,'killed':killed,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
print(json.dumps(records,indent=2))
raise SystemExit(0 if all(r['killed'] for r in records) else 1)
