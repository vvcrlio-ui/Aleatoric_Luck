# FFC: Fragile Families Challenge predictability

> This is the `FFC/` subtree of the [`Aleatoric_Luck`](../README.md) repository.
> See the repository root README for how this relates to `SMR/` and the other
> predictability papers.

This directory contains two things:

- [`NK_Grid/`](NK_Grid/README.md): our own N x K predictability sweep applied
  to FFC's six outcomes (`gpa`, `grit`, `materialHardship`, `eviction`,
  `layoff`, `jobTraining`), built on the same NK_Grid tool used for SMR. It
  adds background-data cleaning (`src/prepare_ffc_analysis.py`,
  `src/prepare_ffc_nk_inputs.py`) and support for FFC's official fixed
  train/test split (`--test-data`), which the original NK_Grid tool did not
  need for SMR's data.
- [`ffc_replication-master/`](ffc_replication-master/README.md): a vendored,
  read-only copy of the original Fragile Families Challenge paper's own
  replication materials (Salganik et al. 2020, PNAS), sourced from
  https://github.com/atkindel/ffc_replication. This is **not** our code — see
  that directory's README for what it actually contains (mainly: aggregation
  and analysis of the 160+ Challenge teams' already-submitted predictions,
  plus a simple 4-variable naive benchmark). It is kept here for local
  reference when checking what the original paper did and did not do.
  Restricted/large data was intentionally not vendored; fetch it from the
  original repository if needed.

## Data

FFC's raw and cleaned data (`NK_Grid/data/`) and `ffc_replication-master`'s
own restricted data (`data/private/`, `data/derived/`) are gitignored. See
`NK_Grid/plans/` (local-only, not committed) for the cleaning design notes,
and `NK_Grid/README.md` for how to regenerate the cleaned feature matrix from
`background.dta`.

## Quick start

```bash
cd FFC/NK_Grid
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```
