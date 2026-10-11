# BMRC controller lock probe

Check whether two BMRC nodes exclude each other using the same `fcntl.flock`
call as the controller. This investigates missing successor-controller entries;
it does not establish the cause of the historical incident by itself.

The job requests two nodes, one task and CPU per node, 512 MB per node, and
a three-minute limit on `short` under `mills.prj`, with `skl-compat` and
`Python/3.11.3-GCCcore-12.3.0`. It creates a fresh diagnostic directory inside
the supplied existing experiment output directory. It never opens the real
`.cluster-state.lock` or writes `cluster-state.json`.

## Retrieve and run

From the existing repository checkout, fetch and extract only this script.
This preserves the checkout used for the completed experiment.

```bash
git fetch https://github.com/OxfordDemSci/aleatoric_luck.git 'SMR&FFC'
probe_script=$(mktemp /tmp/bmrc-controller-lock-probe.XXXXXX.sbatch)
git show FETCH_HEAD:runs/bmrc-controller-lock-probe-20260930/bmrc-controller-lock-probe.sbatch > "$probe_script"

# Set this to the existing experiment output directory on BMRC.
run_root=/absolute/path/to/the/experiment/output
sbatch "$probe_script" "$run_root"
```

Record the job ID printed by `sbatch`. The test runs once. No experiment is
resumed. Its two log files appear in the directory where `sbatch` was run.

## Send back

After the job finishes, send both files, including an empty error file:

- `controller-lock-probe-JOBID.out`
- `controller-lock-probe-JOBID.err`

The `.out` file prints the diagnostic directory and, on a completed probe,
both hostnames, the shared inode, and the contender result. Keep that directory
until the result has been reviewed. It contains `holder.json`,
`contender.json`, and the independent `probe.lock` file.

## Interpretation

- `PASS`: the contender was blocked while the holder retained its lock, on
  the tested node pair. This does not rule out a historical lock-file replacement,
  another state writer, or a problem on a different node pair.
- `FAIL`: both nodes acquired an exclusive lock on the same inode while the
  holder still held its lock. The job deliberately exits nonzero.
- A timeout, `INCONCLUSIVE`, module error, or other exception is not a result
  about cross-node lock correctness. Send the logs unchanged.

Local validation covers Bash and embedded Python syntax. Cross-node behavior
must be measured by running the job on BMRC.
