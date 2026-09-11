# Launching and resuming experiments

The launch layer turns a selected panel, model list, and experiment size into a run. It prepares the environment, checks inputs, schedules computation, and records progress. The shared engine handles preprocessing and model training.

Start with a dry-run preview, then use dev for a small trial or timing_full to check out-of-memory (OOM) errors, runtime, and other execution issues across the full N/K range. Use production for the formal repeated experiment. Each stage is started explicitly.

## From panel selection to execution

[experiment.py](experiment.py) reads the selected panel and determines the outcome, models, and N/K design. Once inputs are ready, the code checks the design against the available sample and source counts. Preview mode displays the launch request without reading analysis tables or starting training.

Local runs call the engine directly. Cluster runs define the task table and assign tasks to workers. Each new experiment uses a separate output directory.

## Preparing the environment and data

The launcher creates or checks the Python environment for the selected execution environment. The dataset's adapter prepares the data; the schema specifies analysis-table paths, the outcome, and feature definitions. The launcher reads the panel's schema or a user-supplied schema.

Prepared analysis data can be reused. To update preprocessing, regenerate the data and schema with the adapter before starting an experiment. Some cluster integrations offer automatic data preparation before execution; see the root tutorial for details.

## Scheduling local and cluster computation

Local execution starts training processes directly. On Slurm clusters, job scripts request CPUs, memory, and wall time. Worker processes start when resources are allocated. Larger workloads can run in batches.

The experiment design determines which N/K, seed, draw, and model combinations to evaluate. Resource settings determine how many tasks run concurrently. Cluster-specific integrations handle accounts, partitions, resource limits, and submission; the shared engine implements the training methods.

## Resuming after interruption

A resumed run reuses the original inputs and parameters, reads saved results, and schedules only the remaining tasks.

`complete` means result validation, publication, and the selected checkpoint handling have finished. Training whose results were not saved before interruption is repeated on resume.

Existing environment integrations include BMRC and Discoverer. Other clusters require configuration and submission scripts suited to their environment. See the [root quick start](../README.md) for installation, trial runs, and execution commands.
