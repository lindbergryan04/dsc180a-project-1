# Task 7: transfer to DIII-D shot 193870

Part C, step 1 of the Tasks 5–7 handout: score the frozen shot-128913 models on a second
discharge without refitting. `transfer.py` scores all 81 Task 6 ensembles and the three pooled
Task 5 specialists on the 13 supplied shot-193870 profile inputs and 517 perturbed cases
(Team B's runs at NBASIS 12/12 with nonnegative heat flux). Part C steps 2–3, widening the
input ranges and adding q and magnetic shear as features, were not done.

| File | Purpose |
| --- | --- |
| `transfer.py` | Checks the shot-193870 runs against the supplied inputs, then scores every frozen model on six cohorts |
| `results/transfer_scores.csv` | RMSE, MAE, bias, and R² for each model and cohort |
| `01_second_discharge_transfer.ipynb` | Read-only view of the scores |

```sh
python notebooks/ryan/task7/transfer.py
```

This needs the Task 5 and Task 6 checkpoints (not committed) and the Task 7 course archive
(`DATA.md`). With the original checkpoints restored, it reproduced
`transfer_scores.csv` and all 44,520 shot-193870 predictions exactly. It also checks that
every Task 6 model's shot-128913 test RMSE matches Task 6.

The six model inputs omit geometry. All eight recorded geometry inputs differ between the two
shots at every radius, so these scores test transfer across geometry the models never saw.
