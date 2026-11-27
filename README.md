# dsc180a-project-1

UC San Diego DSC 180 group project: a Quarter 1 report on TGLF surrogates and a
Quarter 2 proposal for a CGYRO correction.

## Project overview

**Quarter 1: Fast Neural Network Surrogates for Fusion Reactor Turbulence Simulations: Accuracy, Error Bars, and Transfer Between DIII-D Plasmas**

We compare methods for approximating the ion heat, electron heat, and electron
particle fluxes that TGLF predicts, and ask which modeling choices give useful
predictions under our data and compute limits. We measure prediction error,
cost, uncertainty, and behavior across plasma conditions, with DIII-D as a case
study.

**Quarter 2: Training a neural network surrogate for nonlinear CGYRO nuclear fusion simulations via existing quasilinear TGLF simulations**

We train a neural-network correction that moves TGLF predictions closer to
about 300 nonlinear CGYRO cases. It uses TGLF fluxes, TGLF spectral features,
and a surrogate pretrained on a large TGLF dataset. See the
[proposal](quarter_2_proposal/Q2_CGYRO_Proposal.pdf).

We chose the models, architectures, and visualizations and interpreted the
results. AI helped implement the models and write documentation; see
[AI_USE.md](AI_USE.md).

## People

Group members (Halıcıoğlu Data Science Institute, UC San Diego): Ryan Lindberg,
Sweekrit Bhatnagar, Joseph Calderon (Joey), and Aryan Nath.

Mentors: Emily Belli and Jeff Candy (General Atomics), and Igor Sfiligoi (San
Diego Supercomputer Center).

## Repository structure

| Path | Contents |
| --- | --- |
| `notebooks/` | One folder per member, with a `taskN/` folder per task. |
| `src/` | Shared helpers: `tglf_parser.py` reads TGLF run folders and `d3d_profile.py` builds DIII-D inputs. |
| `quarter_1_report/` | Quarter 1 report source, PDF, figures, and course style files. |
| `quarter_2_proposal/` | Quarter 2 proposal source and PDF, plus TikZ sources for its two figures. |
| `tasks/` | Mentor instructions for Tasks 1 to 7, in Markdown. |
| `reference_slides/` | Mentor slides and our Week 5 check-in presentation. |
| `DATA.md` | Dataset descriptions, access, file formats, and quality checks. |
| `REFERENCES.md` | Citations and DOI links for the background papers. |
| `INSTRUCTIONS.md` | Quarter 1 deadlines and submission rules. |
| `AI_USE.md` | How we used AI tools. |
| `requirements.txt` | Pinned Python dependencies. |
| `.gitignore` | Excludes datasets, environments, credentials, and build files. |

The course datasets are not in Git. Unpack them into a local, gitignored
`data/` folder as described in [DATA.md](DATA.md).

## Environment

`requirements.txt` pins the packages Ryan and Sweekrit used, both with CPU
PyTorch (Sweekrit on Python 3.12.11, Ryan on 3.12.12). From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce results

### Ryan (`notebooks/ryan/`)

The notebooks hold the exploratory analyses and their saved outputs. To rebuild
the reported experiments, extract the course archives as described in
[DATA.md](DATA.md), then run Tasks 1 to 3 in order from the repository root:

```sh
sat3=data/Task1_TGLF_SAT3_dataset/SAT3_dataset/data
scan=data/Task2_TGLF_scan/scan_db/data
uq=data/Task3_TGLF_Uncertainty
runs=notebooks/ryan/runs

python notebooks/ryan/task1/starter.py --sat3-data "$sat3" --output "$runs/task1"
python notebooks/ryan/task2/scan.py --scan-data "$scan" --sat3-data "$sat3" --activations --learning-curves --output "$runs/task2"
python notebooks/ryan/task3/task3_workflow.py --task2 "$runs/task2" --data "$uq" --output "$runs/task3"
```

Task 1 compares the starter models on 43 cases. Task 2 builds the splits, trains
the baselines and a five-network ensemble, and evaluates activation settings and
learning curves. Task 3 reuses the Task 2 models and trains ten more members for
the uncertainty analysis. Set `runs` to a fresh directory for a new run. Timings
and results can vary with CPU and software.

Tasks 4 to 7 each have a folder with a short README (inputs and checks), the
scripts, the result tables the report cites, and a read-only results notebook.
Run them in order.

| Folder | Task | Main command(s) |
| --- | --- | --- |
| `notebooks/ryan/task4/` | DIII-D #128913 database and team merge | `tglf_cases.py`, `team_database.py` |
| `notebooks/ryan/task5/` | Shared vs per-flux surrogates, pooled specialists, 90% intervals | `search.py all`, `report.py`, `pooled_intervals.py pool/calibrate/score` |
| `notebooks/ryan/task6/` | Learning curves, network width, error by radius and input | `prepare.py clean/split`, `learning_curves.py train/summarize`, `part_b.py capacity/score/analyze` |
| `notebooks/ryan/task7/` | Transfer to DIII-D shot 193870 | `transfer.py` |
| `notebooks/ryan/cgyro_check/` | Surrogate error vs. TGLF's difference from nonlinear CGYRO (43 SAT3 cases) | `cgyro_check.py` |

```sh
python -m unittest discover -s notebooks/ryan/task5 -p test_task5.py
```

The Task 5 to 7 course files go in `data/` ([DATA.md](DATA.md)). Trained models
(about 800 MB) are not committed. Retraining Tasks 5 and 6 takes several CPU
hours and writes to `notebooks/ryan/runs/`. Training uses `fcntl`, so it needs
macOS or Linux. TGLF runs need a GACODE install (we used DSMLP).

### Sweekrit (`notebooks/sbhatnagar/`)

Each task folder holds one notebook, its `figures/`, and its intermediate CSVs
in `data/` (committed). `tglf_dataset.py`, which reads TGLF run folders into a
table, sits one level up in `notebooks/sbhatnagar/` and is shared by the tasks.

| Task | Notebook | Needs | Writes |
| --- | --- | --- | --- |
| `task1/` | `01_sat3_flux_analysis.ipynb` | Raw SAT3 dataset (see below) | `figures/`, `data/sat3_tglf_table.csv`, `data/sat3_experiments.csv` |
| `task2/` | `01_scandb_prediction_training.ipynb` | Committed `data/scan_db_table.csv` and `data/sat3_table.csv` | `figures/` |
| `task3/` | `01_ensemble_uncertainty.ipynb` | Committed `data/*_table.csv` (out-of-range, DIII-D) and Task 2's `data/scan_db_table.csv` | `figures/` |
| `task4/` | `01_d3d_database_generation.ipynb` | TGLF on `PATH` and the Task 4 DIII-D profile, unpacked into `task4/data/original_data/` | `data/d3d_mydb_27_cases.csv`, `data/d3d_mydb_27_table.csv`, `figures/` |
| `task5/` | `01_d3d_surrogate.ipynb` | Committed `data/` tables, plus the team A database `notebooks/ryan/task4/d3d_team_database.csv` | `figures/` |

To run a notebook, start Jupyter in its task folder (for example
`cd notebooks/sbhatnagar/task2 && jupyter lab`) and run all cells in order.
Paths are relative to the task folder. The notebooks are independent; Task 3
only reads a committed table from Task 2.

Seeds are fixed (`RANDOM_STATE = 42`, ensemble seeds 42 to 46). Torch runs on
one CPU thread with deterministic kernels and XGBoost on one thread, so results
should match on any CPU. Training and prediction times (Task 2, Table 2 and
Figure 2) depend on the machine.

Tasks 2, 3, and 5 run from a fresh clone. Task 1 needs the raw SAT3 dataset,
which is too large for Git (source in [DATA.md](DATA.md)). Put it at
`notebooks/sbhatnagar/task1/data/SAT3_dataset/data/`, or point `SAT3_DIR` at its
`data` folder before starting Jupyter:

```sh
export SAT3_DIR=/path/to/SAT3_dataset/data      # Windows PowerShell: $env:SAT3_DIR = "C:\path\to\SAT3_dataset\data"
```

Task 2 rebuilds `data/sat3_table.csv` from the raw dataset (step 4.1) if you
delete that CSV and set `SAT3_DIR`.

### Joey (`notebooks/jscalderon/`)

Place the four course archives where [DATA.md](DATA.md) lists them. Joey's
tasks read `SAT3_dataset.tar.gz`, `scan_db.tar.gz`, `scan_db_outside.tar.gz`,
and `data_d3d.tar.gz` directly. Use a separate Python 3.12 environment, since
Joey's Task 2 results record pandas 3.1.0 and the shared environment uses
3.0.6. From the repository root:

```sh
joey=notebooks/jscalderon
python3.12 -m venv "$joey/.venv"
source "$joey/.venv/bin/activate"
python -m pip install -r "$joey/task2/results/requirements.txt" scipy==1.18.1 threadpoolctl==3.7.0 jupyterlab==4.6.4 ipykernel==7.4.0

out=data/generated/joey
mkdir -p "$out"
python "$joey/task1/support/analyze_sat3.py"
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 --output=task2 --output-dir="$out" "$joey/task2/01_tglf_surrogate_neural_networks.ipynb"
python "$joey/task2/support/check_results.py" --notebook "$out/task2.ipynb"
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 --output=task3 --output-dir="$out" "$joey/task3/01_ensemble_uncertainty.ipynb"
```

Task 4 needs TGLF on `PATH` (GACODE commit `1df9cdce`). Extract
`task4_d3d.tar.gz`, copy `make_db.py` into its `tools/` folder, and run from
inside `task4_d3d`:

```sh
t4="<repo>/notebooks/jscalderon/task4"
python tools/run_tglf_parallel.py d3d_profile -nproc 16
python "$t4/collect_d3d.py" step1 . "$t4/step1_check.csv"
python tools/make_db.py                                  # seed 3, 1,000 cases
python tools/run_tglf_parallel.py d3d_mydb_3/data -nproc 16
python "$t4/collect_d3d.py" table . 3 "$t4/d3d_database.csv"
python "$t4/plot_d3d.py"
```

| Task | What to check |
| --- | --- |
| 1 | Parses 43 SAT3 cases and checks eight isotope triplets. The parsed JSON is in `data/generated/joey/task1/`. The third flux block is momentum (`Pi`) and the fourth is energy exchange (`S`). |
| 2 | Regenerates tables, predictions, figures, and weights in `task2/results/`. Expected split: 3,496 training, 750 validation, and 750 test cases, plus 43 separate SAT3 cases. Recorded test RMSE: 17.125 (ion heat), 12.700 (electron heat), 3.453 (electron particle flux). The read-only checker recomputes the metrics from the predictions and exits with an error on any mismatch. |
| 3 | Trains its own ensemble and shows results in `data/generated/joey/task3.ipynb`. Expected inputs: 5,000 scan cases, 270 outside-range cases, and 300 perturbed DIII-D cases. Recorded test coverage: 89.7% (electron heat), 89.2% (ion heat), 93.1% (electron particle flux). |
| 4 | All 13 profile radii match `d3d_profile_answers.dat`. `d3d_database.csv` has 1,000 cases (seed 3) in the same columns as Aryan's. All runs succeeded, 994 have positive heat fluxes, and none duplicate a teammate's case. |

The Task 2 run overwrites its saved results. To check the existing results
without rerunning, run `python "$joey/task2/support/check_results.py"`.

### Aryan (`notebooks/aryan/`)

Each task folder holds one notebook. Tasks 1 to 3 show figures and tables inline;
Task 4 saves `cases.csv` next to its notebook. No task writes a `figures/` or
`data/` folder.

| Task | Notebook | Needs (raw data) | Produces |
| --- | --- | --- | --- |
| `task1/` | `01_sat3_parameter_scans.ipynb` | The 43-case SAT3 starter dataset, unpacked locally at `data/Task1_TGLF_SAT3_dataset/SAT3_dataset/data/` (see [DATA.md](DATA.md)) | `plots/*.png` (7 parameter-scan figures) and `findings_summary.md`, both already in `task1/`, also reproducible by running `generate_plots.py` on its own |
| `task2/` | `01_tglf_surrogate_models.ipynb` | `scan_db` (5,000 TGLF cases) and the 43-case `SAT3_dataset`, both outside the repo (see Data paths) | Tuned `HistGradientBoostingRegressor` surrogates for $Q_i$, $Q_e$, $\Gamma_e$, saved to `tglf_surrogate_models.joblib`; otherwise inline tables and plots only |
| `task3/` | `01_surrogate_uncertainty_quantification.ipynb` | `scan_db`, `scan_db_outside` (9 single-parameter out-of-distribution sweeps), and `data_d3d` (DIII-D reference case and 300 perturbed TGLF runs), all outside the repo | A 10-member bootstrap ensemble with validation-calibrated 90% prediction intervals, saved to `tglf_uncertainty_surrogate.joblib`; otherwise inline only |
| `task4/` | `01_d3d_tglf_database.ipynb` | `task4_d3d` (DIII-D discharge #128913 profile cases at 13 radii, plus the case-generation tools), outside the repo | `notebooks/aryan/task4/cases.csv` (the 1,000 generated input cases) and `notebooks/aryan/task4/d3d_database.csv` (the same cases merged with their TGLF fluxes), both committed |

The model files `tglf_surrogate_models.joblib` (Task 2, about 2.5 MB) and
`tglf_uncertainty_surrogate.joblib` (Task 3, about 26 MB) are not committed.
They are written to the notebook's working directory, the `.../DSC 180A/`
folder that contains the repo clone, not inside `dsc180a-project-1/`.

#### Data paths

Task 1 finds its dataset through the `SAT3_DATA_DIR` environment variable or a
fallback, `data/Task1_TGLF_SAT3_dataset/SAT3_dataset/data/`, located relative to
the notebook. Tasks 2 to 4 use plain relative paths with no override
(`scan_db/data`, `SAT3_dataset/data`, `data_d3d/d3d_uq/data`,
`task4_d3d/d3d_profile/...`, `task4_d3d/d3d_mydb_4/cases.csv`). These resolve
only because `scan_db/`, `scan_db_outside/`, `data_d3d/`, `task4_d3d/`, and
`SAT3_dataset/` sit next to the repo clone in `.../DSC 180A/` (DATA.md notes
that shared course data "can also remain outside the repository"). From a fresh
clone, place or symlink those five folders next to the repo and start Jupyter
from that parent folder:

```sh
# from the folder that will contain the repo clone
ln -s /path/to/scan_db scan_db
ln -s /path/to/scan_db_outside scan_db_outside
ln -s /path/to/data_d3d data_d3d
ln -s /path/to/task4_d3d task4_d3d
ln -s /path/to/SAT3_dataset SAT3_dataset
cd dsc180a-project-1 && jupyter lab   # or: jupyter lab from the parent folder itself
```

Tasks 2 and 3 also `import parse_sat3`, a standalone module with the same
`load_all_cases()` as `src/tglf_parser.py`. It is not in this repository, so
those two notebooks need it on the import path (next to the notebook or on
`PYTHONPATH`). Task 1 imports `tglf_parser` from `src/`, finding the repo root
automatically. Task 4 needs neither; it reads `cases.csv` and
`out.tglf.gbflux` directly.

#### Running

Use the shared environment above; Task 2's `xgboost` is already pinned in
`requirements.txt`.

```sh
# Task 1: resolves the dataset from data/ once it is unpacked there (see DATA.md)
cd notebooks/aryan/task1 && jupyter lab   # run 01_sat3_parameter_scans.ipynb top to bottom
# optional override if the dataset lives somewhere else:
export SAT3_DATA_DIR=/path/to/SAT3_dataset/data

# Tasks 2-4: start Jupyter from the parent folder with the symlinks above in place
cd "/path/to/DSC 180A"
jupyter lab   # open and run notebooks/aryan/task{2,3,4}/*.ipynb top to bottom
```

Run each notebook top to bottom, in any order. Tasks 2 and 3 set a local
`random_seed = 42` for the 70/15/15 `scan_db` train/validation/test split, the
`HistGradientBoostingRegressor` fits, and Task 3's bootstrap resampling. Task
4's 1,000 cases came from `SEED = 4` in the external script
`task4_d3d/tools/make_db.py`, which produced the committed `cases.csv`; the
notebook only reads that file. Results should match on any CPU, though timings
depend on the machine.

## Report and proposal

Both documents build with `latexmk` from their own folder. Each folder's
`latexmkrc` picks the engine and puts build files in a gitignored `build/`
subfolder, leaving only the `.tex` and the PDF at the top.

### Quarter 1 report

The final report is [quarter_1_report/final_report.pdf](quarter_1_report/final_report.pdf),
built from [quarter_1_report/final_report.tex](quarter_1_report/final_report.tex). Its data
figures are drawn by `python quarter_1_report/make_figures.py` from result tables committed
under `notebooks/`, without retraining any model.

The checkpoint draft is [quarter_1_report/report.pdf](quarter_1_report/report.pdf),
built from [quarter_1_report/report.tex](quarter_1_report/report.tex). It has
the required title, abstract, and introduction, plus methods and preliminary
results. It needs XeLaTeX and the XCharter and Arial fonts (on Ubuntu,
`sudo apt install texlive-xetex texlive-fonts-extra ttf-mscorefonts-installer`).
Build steps are in [quarter_1_report/README.md](quarter_1_report/README.md).

### Quarter 2 proposal

[quarter_2_proposal/Q2_CGYRO_Proposal.pdf](quarter_2_proposal/Q2_CGYRO_Proposal.pdf)
builds with pdfLaTeX:

```sh
cd quarter_2_proposal && latexmk Q2_CGYRO_Proposal.tex
```

Its two figures are TikZ diagrams in `quarter_2_proposal/figures/`. Each `.tex`
compiles on its own to the `.pdf` the proposal includes, plus a `.png` preview.

## Contributing

1. Pull the latest `main` first, and keep your notebooks in your own folder
   under `notebooks/`.
2. Record the model settings, data splits, preprocessing, metrics, and saved
   results needed to interpret each experiment. Keep data files out of Git.
3. Coordinate report edits with the group: edit `quarter_1_report/report.tex`,
   add citations to `quarter_1_report/reference.bib` and figures to
   `quarter_1_report/figures/`, then rebuild the PDF.
4. Commit and push your own work. The TA requires at least one push from every
   member. Review the combined report before the group submits the PDF and
   repository link to Gradescope.
