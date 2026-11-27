# Task 3 results cited in the final report

Frozen outputs of the original Task 3 run of the 15-network ensemble (seeds 42–56,
intervals from calibrated member disagreement), from the same workflow as
`task3_workflow.py`. File names follow that original run; `task3_workflow.py` writes the
out-of-range table as `outside_metrics.csv`.

| File | Contents |
| --- | --- |
| `test_metrics.csv` | Error and 90% interval coverage on the 750 `scan_db` test cases |
| `ood_comparison.csv` | The same on the nine `scan_db_outside` groups, with RMSE and member SD relative to the test set |
| `d3d_paired_predictions.csv` | Predictions and intervals for the 300 perturbed DIII-D (`data_d3d`) cases with their TGLF values |
| `d3d_paired_metrics.csv` | Summary of those 300 cases |

`../../task2/results/test_metrics.csv` holds the Task 2 five-network ensemble and baselines on
the 750 test cases, from the saved models verified in the portable Task 2 package.
