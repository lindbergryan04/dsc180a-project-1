# dsc180a-project-1

DSC 180A Quarter 1 group project.

## Project overview

**Neural Network Surrogates for TGLF: Prediction Accuracy and Uncertainty**

We explore, develop, and compare methods for approximating the ion heat,
electron heat, and electron particle fluxes predicted by TGLF. Our research
question is which modeling approaches and design choices provide useful
predictions under the available data and computational constraints. We assess
prediction error, computational cost, and behavior across plasma conditions.
Uncertainty is part of evaluation, and DIII-D is a case study within the project.

We chose the models and architectures, selected the visualizations, and interpreted the results. We used AI to help implement the models and write project documentation. See [AI_USE.md](AI_USE.md) for details.

## Mentors:
Igor Sfiligoi (SDSC)
Emily Belli (GA)
Jeff Candy (GA)

## Group members:
Ryan Lindberg
Sweekrit Bhatnagar
Joseph Calderon (Joey)
Aryan Nath

## Repository structure

| Path | Contents |
| --- | --- |
| `notebooks/` | Individual experiment notebooks for each member. |
| `data/README.md` | Dataset descriptions, access instructions, file formats, and quality checks. |
| `data/Task1_TGLF_SAT3_dataset/` | Task 1 SAT3 starter dataset. |
| `data/Task2_TGLF_scan/` | Task 2 TGLF parameter-scan data. |
| `data/Task3_TGLF_Uncertainty/` | Task 3 uncertainty and extrapolation datasets. |
| `data/Task4_DIIID_Database/` | Task 4 DIII-D database materials. |
| `emily_slides/` | Mentor slides and the group's Week 5 check-in presentation. |
| `references/` | Research papers used for background and related work. |
| `tasks/` | Mentor-provided instructions for Tasks 1–4. |
| `report/` | Checkpoint report source and PDF, references, figures, review notes, and course template source. |
| `requirements.txt` | Pinned Python dependencies for the project. |
| `.gitignore` | Excludes datasets, local environments, credentials, and generated build files. |

Dataset files under `data/` are stored locally and excluded from Git; only
`data/README.md` is tracked when committed.

## Data

See [data/README.md](data/README.md) for data access and storage instructions.

## Environment

`requirements.txt` pins the scientific packages used by Ryan and Sweekrit.
Both used CPU PyTorch. Sweekrit ran Python 3.12.11 and Ryan ran Python 3.12.12.

Create a Python 3.12 environment from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce results

### Ryan's notebooks (`notebooks/ryan/`)

Ryan’s notebooks contain the exploratory analyses and their saved outputs. To rebuild the reported experiments from the course data, run the commands below from the repository root with the environment above.

Extract the course archives into their respective task directories as described in [data/README.md](data/README.md). Then run the three tasks in order.

```sh
sat3=data/Task1_TGLF_SAT3_dataset/SAT3_dataset/data
scan=data/Task2_TGLF_scan/scan_db/data
uq=data/Task3_TGLF_Uncertainty
runs=notebooks/ryan/runs

python notebooks/ryan/task1/starter.py --sat3-data "$sat3" --output "$runs/task1"
python notebooks/ryan/task2/scan.py --scan-data "$scan" --sat3-data "$sat3" --activations --learning-curves --output "$runs/task2"
python notebooks/ryan/task3/task3_workflow.py --task2 "$runs/task2" --data "$uq" --output "$runs/task3"
```

Task 1 compares the starter models on 43 cases. Task 2 builds the data splits, trains the baselines and five-network ensemble, and evaluates the activation settings and learning curves. Task 3 uses the Task 2 models and trains ten additional members for the uncertainty analysis. For another training run, set `runs` to a fresh directory. Training times and numerical results can vary with the CPU and software environment.

### Sweekrit's notebooks (`notebooks/sbhatnagar/`)

Each task folder holds one notebook, its figures (`figures/`) and its
intermediate CSVs (`data/`, committed to git).

| Task | Notebook | Needs | Writes |
| --- | --- | --- | --- |
| `task1/` | `01_sat3_flux_analysis.ipynb` | Raw SAT3 dataset (see below) | `figures/`, `data/sat3_tglf_table.csv`, `data/sat3_experiments.csv` |
| `task2/` | `01_scandb_prediction_training.ipynb` | Committed `data/scan_db_table.csv` and `data/sat3_table.csv` | `figures/` |
| `task3/` | `01_ensemble_uncertainty.ipynb` | Committed `data/*_table.csv` (scan, out-of-range, DIII-D) | `figures/` |

How to run:

1. Start Jupyter from inside the task folder, for example
   `cd notebooks/sbhatnagar/task2 && jupyter lab`. Paths are relative, and
   `task1/` and `task2/` import `tglf_dataset.py` from their own folder.
2. Run all cells from top to bottom. The tasks are independent, so any order works.
3. Seeds are fixed (`RANDOM_STATE = 42`; ensemble seeds 42-46), torch runs on
   one CPU thread with deterministic kernels, and XGBoost runs on one thread.
   Results should match on any CPU. Training and prediction times (task 2,
   Table 2 and Figure 2) depend on the machine.

Tasks 2 and 3 run from a fresh clone with no extra downloads.

Task 1 reads the raw SAT3 dataset, which is not in git (large; see
[data/README.md](data/README.md) for the source). Put it at
`notebooks/sbhatnagar/task1/data/SAT3_dataset/data/` or set the `SAT3_DIR`
environment variable to its `data` folder before starting Jupyter:

```sh
export SAT3_DIR=/path/to/SAT3_dataset/data      # Windows PowerShell: $env:SAT3_DIR = "C:\path\to\SAT3_dataset\data"
```

Task 2 can also rebuild `data/sat3_table.csv` from the raw dataset (step 4.1)
if you delete that CSV and set `SAT3_DIR`.

### Joey's analysis (`notebooks/jscalderon/`)

Run the commands below from the repository root. Place the four compressed course archives in the directories listed in [data/README.md](data/README.md). Joey’s tasks read `SAT3_dataset.tar.gz`, `scan_db.tar.gz`, `scan_db_outside.tar.gz`, and `data_d3d.tar.gz` directly from those locations.

Use a separate Python 3.12 environment. Joey’s Task 2 results record pandas 3.1.0, while the shared environment uses 3.0.6.

```sh
joey=notebooks/jscalderon
python3.12 -m venv "$joey/.venv"
source "$joey/.venv/bin/activate"
python -m pip install -r "$joey/Task 2/results/requirements.txt" scipy==1.18.1 threadpoolctl==3.7.0 jupyterlab==4.6.4 ipykernel==7.4.0

out=data/generated/joey
mkdir -p "$out"
python "$joey/Task 1/support/analyze_sat3.py"
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 --output=task2 --output-dir="$out" "$joey/Task 2/Task2_Neural_Networks_Calderon.ipynb"
python "$joey/Task 2/support/check_results.py" --notebook "$out/task2.ipynb"
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 --output=task3 --output-dir="$out" "$joey/Task 3/Task3_TGLF_Ensemble.ipynb"
```

Task 4 needs TGLF on `PATH` (GACODE commit `1df9cdce`). Extract `task4_d3d.tar.gz`, copy `make_db.py` into its `tools/` folder, and run from inside `task4_d3d`:

```sh
t4="<repo>/notebooks/jscalderon/Task 4"
python tools/run_tglf_parallel.py d3d_profile -nproc 16
python "$t4/collect_d3d.py" step1 . "$t4/step1_check.csv"
python tools/make_db.py                                  # seed 3, 1,000 cases
python tools/run_tglf_parallel.py d3d_mydb_3/data -nproc 16
python "$t4/collect_d3d.py" table . 3 "$t4/d3d_database.csv"
python "$t4/plot_d3d.py"
```

| Task | What to check |
| --- | --- |
| 1 | Parses 43 SAT3 cases and checks eight isotope triplets. The parsed JSON is in `data/generated/joey/task1/`. The third flux block is momentum (`Pi`), and the fourth is energy exchange (`S`). |
| 2 | Regenerates tables, predictions, figures, and weights in `Task 2/results/`. The expected split is 3,496 training, 750 validation, and 750 test cases, with 43 separate SAT3 cases. Recorded test RMSE is 17.125 for ion heat, 12.700 for electron heat, and 3.453 for electron particle flux. The read-only checker recalculates the metrics from predictions and exits with an error on discrepancies. |
| 3 | Trains its own ensemble and displays results in `data/generated/joey/task3.ipynb`. Expected inputs are 5,000 scan cases, 270 outside-range cases, and 300 perturbed DIII-D cases. Recorded test coverage was 89.7% for electron heat, 89.2% for ion heat, and 93.1% for electron particle flux. |
| 4 | All 13 profile radii match `d3d_profile_answers.dat`. `d3d_database.csv` has 1,000 cases (seed 3) in the same columns as Aryan's; all runs succeeded, 994 have positive heat fluxes, and none duplicate a teammate's case. |

The Task 2 run replaces the saved files in its results directory. To check the existing results alone, run `python "$joey/Task 2/support/check_results.py"`.

### Aryan's Notebooks (`/notebooks/aryan/`)

Each task folder holds one notebook. Figures and intermediate tables are
written inline into the notebook (Tasks 1-3) or saved alongside it (Task 4's
`cases.csv`); none of the four tasks write a `figures/` or `data/` folder.

| Task | Notebook | Needs (raw data) | Produces |
| --- | --- | --- | --- |
| `task1/` | `01_sat3_parameter_scans.ipynb` | The 43-case SAT3 starter dataset, committed in this repo at `data/Task1_TGLF_SAT3_dataset/SAT3_dataset/data/` | `plots/*.png` (7 parameter-scan figures) and `findings_summary.md`, both already in `task1/` (also reproducible by running `generate_plots.py` standalone) |
| `task2/` | `01_tglf_surrogate_models.ipynb` | `scan_db` (5,000 TGLF cases) and the 43-case `SAT3_dataset`, both **outside this repo** (see Data path mapping) | Tuned `HistGradientBoostingRegressor` surrogates for $Q_i$, $Q_e$, $\Gamma_e$, saved to `tglf_surrogate_models.joblib`; inline tables/plots only, nothing else is written to disk |
| `task3/` | `01_surrogate_uncertainty_quantification.ipynb` | `scan_db`, `scan_db_outside` (9 single-parameter out-of-distribution sweeps), and `data_d3d` (DIII-D reference case + 300 perturbed TGLF runs), all **outside this repo** | A 10-member bootstrap ensemble with validation-calibrated 90% prediction intervals, saved to `tglf_uncertainty_surrogate.joblib`; inline tables/plots only |
| `task4/` | `01_diiid_tglf_database.ipynb` | `task4_d3d` (DIII-D discharge #128913 profile cases at 13 radii, plus the case-generation tooling), **outside this repo** | `cases.csv`: the 1,000 generated input cases, committed at `notebooks/aryan/task4/cases.csv`. `d3d_database.csv` (the same 1,000 cases merged with their TGLF fluxes, now that all 1,000 simulations have finished) is committed in this repo at `notebooks/aryan/task4/d3d_database.csv` |

**Model artifacts live outside the repo.** `tglf_surrogate_models.joblib`
(Task 2, ~2.5 MB) and `tglf_uncertainty_surrogate.joblib` (Task 3, ~26 MB) are
written to the notebook's working directory, which is the parent folder
`.../DSC 180A/` that contains this repo clone as a subdirectory — not
anywhere under `dsc180a-project-1/`. Neither file is committed to Git.

#### Data path mapping

Only Task 1 resolves its dataset through an
environment-variable override (`SAT3_DATA_DIR`) plus a repo-relative fallback
(`data/Task1_TGLF_SAT3_dataset/SAT3_dataset/data/`), auto-detected from the
notebook's own location. Tasks 2-4 instead use plain relative paths
(`scan_db/data`, `SAT3_dataset/data`, `data_d3d/d3d_uq/data`,
`task4_d3d/d3d_profile/...`, `task4_d3d/d3d_mydb_4/cases.csv`) with **no env
var override**, and these paths currently resolve correctly only because
`scan_db/`, `scan_db_outside/`, `data_d3d/`, `task4_d3d/`, and `SAT3_dataset/`
already sit as siblings of this repo clone, directly inside
`.../DSC 180A/` (outside the repo, matching `data/README.md`'s note that
shared course data "can also remain outside the repository"). To reproduce
Tasks 2-4 from a fresh clone, place or symlink those five directories next to
(not inside) the cloned repo and start Jupyter from that parent folder, e.g.:

```sh
# from the folder that will contain the repo clone
ln -s /path/to/scan_db scan_db
ln -s /path/to/scan_db_outside scan_db_outside
ln -s /path/to/data_d3d data_d3d
ln -s /path/to/task4_d3d task4_d3d
ln -s /path/to/SAT3_dataset SAT3_dataset
cd dsc180a-project-1 && jupyter lab   # or: jupyter lab from the parent folder itself
```

Tasks 2 and 3 also `import parse_sat3`, a standalone parsing module with the
same `load_all_cases()` function as `src/tglf_parser.py`. That module is
**not currently copied into this repository**; it must be importable (for
example, copied next to the notebook, or placed on `PYTHONPATH`) for those
two notebooks to run from a fresh clone. Task 1 does not have this gap — its
notebook imports `tglf_parser` directly from `src/` with automatic repo-root
detection. Task 4 does not need either parser; it reads `cases.csv` and
`out.tglf.gbflux` files directly.

#### Reproduction commands

Activate the environment as described above
(`python3.12 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt`).
Task 2 additionally needs `xgboost`, already pinned in `requirements.txt`.

```sh
# Task 1: dataset already resolves from the repo, no extra setup needed
cd notebooks/aryan/task1 && jupyter lab   # run 01_sat3_parameter_scans.ipynb top to bottom
# optional override if the dataset lives somewhere else:
export SAT3_DATA_DIR=/path/to/SAT3_dataset/data

# Tasks 2-4: start Jupyter from the parent folder with the symlinks above in place
cd "/path/to/DSC 180A"
jupyter lab   # open and run notebooks/aryan/task{2,3,4}/*.ipynb top to bottom
```

Run all cells top to bottom; the four notebooks are independent of each
other and can be run in any order. Seeds are fixed: Tasks 2 and 3 use a
plain local variable `random_seed = 42` (not an exported constant) for the
70%/15%/15% `scan_db` train/validation/test split, the
`HistGradientBoostingRegressor` fits, and (Task 3) the bootstrap resampling
RNG. Task 4's 1,000-case generation uses `SEED = 4`, but that constant lives
in the external generation script (`task4_d3d/tools/make_db.py`, outside
this repo) that produced the committed `cases.csv`, not in the notebook
itself — the notebook only reads the already-generated `cases.csv`. Results
should match on any CPU; training and prediction times depend on the
machine.

## Report

The current checkpoint draft is [report/report.pdf](report/report.pdf), with
editable source in [report/report.tex](report/report.tex). It includes the
required title, abstract, and introduction, plus documented methods and
preliminary results. See [report/README.md](report/README.md) for build instructions.

## Contributing to the checkpoint

1. Pull the latest `main` before making changes. Put your experiment notebooks
   in your own directory under `notebooks/`.
2. Include the model settings, data splits, preprocessing, metrics, and saved
   results needed to interpret your experiments. Keep data files out of Git.
3. Coordinate edits with the group. Edit `report/report.tex`, add citations to `report/reference.bib` and figures
   to `report/figures/`, then rebuild the PDF using the report's build instructions.
4. Commit and push your own contribution. The TA requires at least one push
   from every team member. Review the combined report before the group submits
   the report PDF and repository link to Gradescope.
