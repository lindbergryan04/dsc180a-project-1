# Task 5: DIII-D surrogates with error bars

Part A of the Tasks 5–7 handout. Surrogates for electron heat flux `Q_e`, ion heat flux
`Q_i`, and electron particle flux `Gamma_e`, trained on both teams' Task 4 runs for DIII-D
shot #128913 (inputs `RLTS_1`, `RLTS_2`, `RLNS_1`, `TAUS_2`, `XNUE`, `RMIN_LOC`). Every score
compares with numerical TGLF, not with CGYRO or experiment.

| File | Purpose |
| --- | --- |
| `data.py` | Combine the team tables (53 negative-heat cases excluded, signed `Gamma_e` kept) and rebuild the frozen 5,131 / 1,100 / 1,100 split |
| `engine.py` | Preprocessing fitted on training rows, five-network ensembles, checkpointing, inference |
| `search.py` | Staged architecture search, shared versus per-flux ensembles (194 ensembles, 970 fits) |
| `report.py` | Summary tables from the frozen final evaluation |
| `pooled_intervals.py` | Pool the three repeats of each selected specialist (15 networks), calibrate 90% intervals, score them |
| `test_task5.py` | Split, preprocessing, restart, and selection-gate tests |
| `search_matrix.csv` | The 114 one-change configurations screened first |
| `split.csv` | Frozen split membership |
| `varied_test.csv` | 240 varied cases at the 12 intermediate radii (seed 180501) with their TGLF answers |
| `results/` | The tables the report cites |
| `task5_results.ipynb` | Read-only view of `results/` |

## Commands

Place the Task 5 course files as described in `DATA.md`. From the repository root:

```sh
t5=notebooks/ryan/task5
python -m unittest discover -s $t5 -p test_task5.py     # seconds
python $t5/search.py all --device cpu --threads 2       # several hours of CPU
python $t5/report.py
python $t5/pooled_intervals.py pool
python $t5/pooled_intervals.py calibrate
python $t5/pooled_intervals.py score
```

Output goes to `notebooks/ryan/runs/task5/` (ignored by Git), and the scripts refuse to write
into this folder. The checkpoints (about 470 MB) are not committed, so every step after the
tests needs the search to be rerun. A fresh run on different hardware can differ numerically.
`search.py` and `engine.py` use `fcntl` and `resource`, so training needs macOS or Linux.

## Protocol

Validation scores chose every configuration, checkpoint, and finalist. The test set and
the 12 profile cases were opened only after the final choice was frozen in
`results/selection.json`. Ensemble predictions average the networks after each one's inverse
target transform. The test cases had been seen in an earlier quick demonstration under another
split, so the 240 varied cases are the cleaner held-out check. Their forecasts and intervals
were frozen (forecast SHA-256 `a18bf8c5…`) before TGLF was run on them. The TGLF answers came
from a local serial GACODE TGLF build (commit `925525eb`); `../task4/tglf_cases.py` regenerates
the input decks.

## Checks run when this code moved into the repository

With the original checkpoints restored into `notebooks/ryan/runs/task5/search/`, these
reproduced the original outputs byte for byte (gzip files compared after decompression):
the split (also covered by the unit tests), every per-run metrics table, `system_summary.csv`,
`paired_bootstrap.csv`, `flux_regime_metrics.csv`, the pooled scores and predictions, the
interval calibration and freeze record (including its checkpoint and forecast hashes), and
the interval scores. `engine.py` is unchanged, so the saved runs keep their identity hash.
