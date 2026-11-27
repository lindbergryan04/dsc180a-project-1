# Task 6: how many TGLF runs, and how large a network

Part B of the Tasks 5–7 handout, for DIII-D shot #128913. It uses the three Task 5
specialists' recipes on a larger cleaned pool (23,694 standard Task 4 runs from both teams).
All errors compare with numerical TGLF.

| File | Purpose |
| --- | --- |
| `prepare.py` | Clean both teams' runs; record the specialists; freeze the 70/15/15 split (16,586 / 3,554 / 3,554), keeping every Task 5 case in its Task 5 partition |
| `learning_curves.py` | 81 five-network ensembles: five training sizes, three draws, three widths; test scored only after the width comparisons are frozen |
| `part_b.py` | 4× width and rerun controls, scoring on more held-out cohorts with 90% error bars, and the analysis tables |
| `specialists.json` | Architecture, transforms, and training settings of the three specialists |
| `split.csv` | Frozen split membership |
| `results/` | The tables the report cites |
| `task6_results.ipynb` | Read-only view of `results/` |

## Commands

Needs the Task 6 Team B table (`DATA.md`) and the Task 5 files. From the repository root:

```sh
t6=notebooks/ryan/task6
python $t6/prepare.py clean                          # writes data/processed/task6/combined_nbasis12.csv
python $t6/prepare.py split                          # compare with split.csv
python $t6/learning_curves.py train --workers 5      # several hours of CPU
python $t6/learning_curves.py summarize
python $t6/part_b.py capacity --workers 6
python $t6/part_b.py score
python $t6/part_b.py analyze
```

`prepare.py specialists` regenerates `specialists.json` from a completed Task 5 search. Output
goes to `notebooks/ryan/runs/task6/` (ignored by Git); the checkpoints are not committed.

## Checks run when this code moved into the repository

`prepare.py` reproduced `combined_nbasis12.csv` and the split byte for byte, and its specialist
record matches the original architecture file. The training settings it records rebuild all 81
learning-curve configurations and all six capacity-run configurations exactly. With the original
checkpoints restored, `learning_curves.py summarize` reproduced the subset plan and every results
file, and `part_b.py score`/`analyze` reproduced all 16 result tables and both compressed tables.

## Scope limits

Everything here is one discharge with fixed geometry per radius. Spreads across training draws
measure sensitivity to the training sample, not uncertainty for one input. Scaling fits are
order-of-magnitude estimates; a few cases hold a large share of the squared error.
