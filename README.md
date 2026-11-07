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

| Task | What to check |
| --- | --- |
| 1 | Parses 43 SAT3 cases and checks eight isotope triplets. The parsed JSON is in `data/generated/joey/task1/`. The third flux block is momentum (`Pi`), and the fourth is energy exchange (`S`). |
| 2 | Regenerates tables, predictions, figures, and weights in `Task 2/results/`. The expected split is 3,496 training, 750 validation, and 750 test cases, with 43 separate SAT3 cases. Recorded test RMSE is 17.125 for ion heat, 12.700 for electron heat, and 3.453 for electron particle flux. The read-only checker recalculates the metrics from predictions and exits with an error on discrepancies. |
| 3 | Trains its own ensemble and displays results in `data/generated/joey/task3.ipynb`. Expected inputs are 5,000 scan cases, 270 outside-range cases, and 300 perturbed DIII-D cases. Recorded test coverage was 89.7% for electron heat, 89.2% for ion heat, and 93.1% for electron particle flux. |

The Task 2 run replaces the saved files in its results directory. To check the existing results alone, run `python "$joey/Task 2/support/check_results.py"`.

### Aryan's Notebooks (`/notebooks/`)

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
