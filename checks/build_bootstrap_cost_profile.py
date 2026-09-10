"""Summarize measured single-model latency; no new numerical fitting runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from aleatoric_nk_grid.shared_queue import atomic_json, file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    data=json.loads(args.results.read_bytes())
    groups=defaultdict(list); expanded={}
    for arm in data:
        if arm['arm']!='global_single':continue
        for item in arm['records']:
            row=item['row']
            if row['status']!='ok':continue
            groups[(row['model'],row['N'],row['K'],row['K_expanded'])].append(item['seconds'])
            expanded[str(row['K'])]=row['K_expanded']
    samples=[dict(model=m,N=n,K=k,K_expanded=p,seconds=statistics.median(seconds),observations=len(seconds))
        for (m,n,k,p),seconds in sorted(groups.items())]
    value={'format':'model-cell-cost-v1','evidence':{
        'kind':'local Windows full-budget nine-model benchmark, global single-model arm only',
        'results_sha256':file_digest(args.results),'seeds':[12345],'draws':[0],
        'limitations':'Four GPA cells and two repeats. Not measured on Discoverer. Extrapolation is an ordering heuristic; no 698-worker speedup claim.'},
        'expanded_by_k':expanded,'samples':samples}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,value)
    print(json.dumps({'samples':len(samples),'models':len({x['model'] for x in samples}),'profile_sha256':file_digest(args.output)}))


if __name__=='__main__':main()
