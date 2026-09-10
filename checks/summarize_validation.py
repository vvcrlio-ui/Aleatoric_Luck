"""Freeze the local development evidence without changing production outputs."""
from pathlib import Path
import json
import shutil
import statistics
import subprocess
import zipfile
from aleatoric_nk_grid.shared_queue import atomic_json, file_digest
from aleatoric_nk_grid.result_migration import source_compatibility


def main():
    source=Path(__file__).resolve().parents[1]; output=source.parent; original=source.parents[2]
    benchmark=output/"gpa-benchmark-01"
    completion=json.loads((benchmark/"completion.json").read_bytes())
    assert completion["complete"] and completion["all_bitwise_equal"] and completion["model_runs"]==168
    runs=json.loads((benchmark/"results.json").read_bytes())
    protocol=json.loads((benchmark/"protocol.json").read_bytes())
    protocol["source_sha256"]={name.replace("\\","/"):sha for name,sha in protocol["source_sha256"].items()}
    rows=[]; baseline=statistics.mean(r["wall_seconds"] for r in runs if r["arm"]=="fixed_groups")
    for arm in ("fixed_groups","queue_single","queue_cache"):
        selected=[r for r in runs if r["arm"]==arm]
        wall=statistics.mean(r["wall_seconds"] for r in selected)
        rows.append({"arm":arm,"mean_seconds":wall,"runs_seconds":[r["wall_seconds"] for r in selected],
            "time_reduction_fraction":1-wall/baseline,"throughput_relative":baseline/wall,
            "task_wall_occupancy":sum(sum(r["worker_busy_seconds"]) for r in selected)/(6*wall),
            "mean_model_cpu_seconds":statistics.mean(r["model_task_cpu_seconds"] for r in selected)})
    # Numerical identity is still exactly the version used in the GPA benchmark.
    for name in ("nk_grid.py","model_registry.py","preprocessing.py","fold_local.py","svd_fallback.py"):
        relative="NK_Grid/src/aleatoric_nk_grid/"+name
        assert file_digest(source/relative)==protocol["source_sha256"][relative]
    certificate=source_compatibility(original,source)
    atomic_json(output/"compatibility-certificate.json",certificate)
    changed=sorted(name for name,sha in protocol["source_sha256"].items() if file_digest(source/name)!=sha)
    log=(output/"checks-final.log").read_text(encoding="utf-8")
    assert "35 passed" in log
    scale=json.loads((output/"queue-scale-100k/report.json").read_bytes())
    assert scale["restart_verified"]
    files=sorted((source/"NK_Grid/src/aleatoric_nk_grid").glob("*.py"))
    files+=sorted((source/"checks").glob("*.py"))
    files += [source/name for name in ("FFCWS/model_params.yaml","SMR/model_params.yaml",
        "NK_Grid/model_params.yaml","SCHEDULER_DEVELOPMENT.md","LOCAL_VALIDATION.md")]
    with zipfile.ZipFile(output/"development-source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(source).as_posix())
    summary={"completion":completion,"gpa_comparison":rows,"local_tests_passed":35,
        "source_base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip(),
        "source_branch":subprocess.check_output(["git","branch","--show-current"],cwd=source,text=True).strip(),
        "changes_since_gpa_benchmark":changed,"numerical_source_unchanged_since_benchmark":True,
        "queue_scale_probe":scale,"production_switched":False,
        "final_source_sha256":{path.relative_to(source).as_posix():file_digest(path) for path in files},
        "evidence_sha256":{name:file_digest(output/name) for name in (
            "gpa-benchmark-01/results.json","gpa-benchmark-01/protocol.json","gpa-benchmark-01/completion.json",
            "queue-scale-100k/report.json","checks-final.log","compatibility-certificate.json","development-source.zip")}}
    atomic_json(output/"validation-summary.json",summary)
    shutil.copyfile(source/"LOCAL_VALIDATION.md",output/"REPORT.md")
    print(json.dumps({"model_runs":168,"passed_checks":35,"numerical_hashes_verified":True,"gpa_comparison":rows},ensure_ascii=False))


if __name__=="__main__":main()
