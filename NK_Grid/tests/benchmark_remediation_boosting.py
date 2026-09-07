"""One fresh native process per implementation. Use /usr/bin/time -v on Linux.

Input NPZ must contain already prepared float64 X/y under the M2 legacy
preprocessing definition. Synthetic input is never labelled real GPA/OOM.
Run baseline and M2 separately under identical cgroup/CPU limits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import platform

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--commit',required=True)
parser.add_argument('--model',choices=['xgboost','lightgbm'],required=True)
parser.add_argument('--input',type=Path)
parser.add_argument('--rows',type=int,default=201)
parser.add_argument('--columns',type=int,default=16085)
parser.add_argument('--seed',type=int,default=12345)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()

def peak_rss_bytes():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counter = Counters(); counter.cb = ctypes.sizeof(counter)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not query(current(),ctypes.byref(counter),counter.cb):
            raise ctypes.WinError()
        return int(counter.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)

source=subprocess.check_output(['git','show',f'{args.commit}:NK_Grid/src/aleatoric_nk_grid/model_registry.py'],cwd=ROOT)
module=types.ModuleType('native_benchmark_registry');module.__file__=str(ROOT/'NK_Grid/src/aleatoric_nk_grid/model_registry.py')
sys.modules[module.__name__]=module
exec(compile(source,module.__file__,'exec'),module.__dict__)
if args.input:
    data=np.load(args.input,allow_pickle=False);X,y=data['X'],data['y'];label='supplied NPZ'
else:
    rng=np.random.default_rng(971);X=rng.normal(size=(args.rows,args.columns));y=rng.normal(size=args.rows);label='synthetic'
params=yaml.safe_load((ROOT/'NK_Grid/model_params.yaml').read_text())['regression'][args.model]
record={'commit':args.commit,'source_sha256':hashlib.sha256(source).hexdigest(),'input_kind':label,
        'input_array_sha256':hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),'shape':X.shape,
        'dtype':str(X.dtype),'params':params,'seed':args.seed,'threads':1,'platform':platform.platform(),
        'peak_definition':'OS process high-water RSS, cumulative including input loading; not Slurm cgroup or arbitrary child processes'}
library=__import__(args.model)
original_train=library.train
def observed_train(*a,**kw):
    final = not kw.get('evals') and not kw.get('valid_sets')
    if final: record['cv_phase_peak_rss_bytes']=peak_rss_bytes()
    start=time.perf_counter()
    result=original_train(*a,**kw)
    if final:
        record['final_fit_seconds']=time.perf_counter()-start
        record['through_final_fit_peak_rss_bytes']=peak_rss_bytes()
    return result
library.train=observed_train
started=time.perf_counter()
with threadpool_limits(limits=1):
    constructor=module.XGBoostCVRegressor if args.model=='xgboost' else module.LightGBMCVRegressor
    fitted=constructor(seed=args.seed,n_jobs=1,**params).fit(X,y)
    pred=fitted.predict(X)
record.update(seconds=time.perf_counter()-started,best_rounds=int(fitted.best_rounds_),
              prediction_sha256=hashlib.sha256(np.asarray(pred).tobytes()).hexdigest())
record['process_peak_rss_bytes']=peak_rss_bytes()
with args.output.open('x') as handle:json.dump(record,handle,indent=2)
print(json.dumps(record))
