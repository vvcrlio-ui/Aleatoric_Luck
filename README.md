# Aleatoric Luck

Predictability research infrastructure, organized by paper/dataset.

```text
Aleatoric_Luck/
├── FFC/     — Fragile Families Challenge predictability (see FFC/README.md)
└── SMR/     — SMR income predictability + Zheng-Cheng replication (see SMR/README.md)
```

Each top-level directory is self-contained: its own `NK_Grid/` copy (the N x K
predictability sweep tool), its own `data/` (gitignored, not committed), and
its own setup instructions. `NK_Grid` is intentionally **not** shared as a
single root-level package — each paper's copy can diverge (e.g. FFC's copy
adds background-data cleaning and a fixed external test-split mode that SMR
does not need). When a change to `NK_Grid` itself is broadly useful, port it
across copies deliberately rather than assuming they stay in sync.

## Layout

- [`FFC/`](FFC/README.md)
  - `NK_Grid/` — N x K sweep adapted for FFC's six outcomes, official
    train/test split, and cleaned `background.dta` feature matrix.
  - `ffc_replication-master/` — vendored copy of the original paper's own
    replication materials (third-party code, not ours; see its README).
- [`SMR/`](SMR/README.md)
  - `NK_Grid/` — N x K sweep for the SMR hourly-wage / total-income panels.
  - `Zheng_Cheng_Replication/` — the Zheng & Cheng (2025) replication
    workflow. Unrelated to the predictability-sweep work; shares no code
    with `NK_Grid/` beyond copied support modules.
  - `tests/` — covers both subprojects above.

## Data

No raw or derived data is committed anywhere in this repository (`**/data/`
is gitignored repo-wide). Each subproject documents how to obtain or
regenerate its own data in its own README.
