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
| `src/` | Reserved for reusable data-processing, modeling, and evaluation code. |
| `notebooks/` | Individual experiment notebooks for each member. |
| `data/README.md` | Dataset descriptions, access instructions, file formats, and quality checks. |
| `data/Task1_TGLF_SAT3_dataset/` | Task 1 SAT3 starter dataset. |
| `data/Task2_TGLF_scan/` | Task 2 TGLF parameter-scan data. |
| `data/Task3_TGLF_Uncertainty/` | Task 3 uncertainty and extrapolation datasets. |
| `data/Task4_DIIID_Database/` | Task 4 DIII-D database materials. |
| `emily_slides/` | Mentor slides and the group's Week 5 check-in presentation. |
| `references/` | Research papers used for background and related work. |
| `tasks/` | Mentor-provided instructions for Tasks 1–4. |
| `report/` | Checkpoint report source and PDF, references, figures, review notes, and the original course template. |
| `requirements.txt` | Python dependency list, to be populated as code is added. |
| `.gitignore` | Excludes datasets, local environments, credentials, and generated build files. |

Dataset files under `data/` are stored locally and excluded from Git; only
`data/README.md` is tracked when committed.

## Data

See [data/README.md](data/README.md) for data access and storage instructions.

## Environment

`requirements.txt` pins the packages used by the notebooks in `notebooks/sbhatnagar/`
(Python 3.12.11, CPU torch). Other members' notebooks may need more packages.

Create a local environment from the repository root (Python 3.12 required;
3.12.11 is the version tested):

```sh
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce results

### Ryan's notebooks (`notebooks/ryan/`)

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

### Joey's Notebooks (`/notebooks/jscalderon`)

### Aryan's Notebooks (`/notebooks/`)

## Report

The current checkpoint draft is [report/report.pdf](report/report.pdf), with
editable source in [report/report.tex](report/report.tex). It includes the
required title, abstract, and introduction, plus documented methods and
preliminary results. See [report/README.md](report/README.md) for build
instructions and [report/draft_notes.md](report/draft_notes.md) for evidence
locations and remaining group work.

## Contributing to the checkpoint

1. Pull the latest `main` before making changes. Put your experiment notebooks
   in your own directory under `notebooks/`.
2. Include the model settings, data splits, preprocessing, metrics, and saved
   results needed to interpret your experiments. Keep data files out of Git.
3. Coordinate report edits using `report/todo.txt` and the evidence notes.
   Edit `report/report.tex`, add citations to `report/reference.bib` and figures
   to `report/figures/`, then rebuild the PDF using the report's build instructions.
4. Commit and push your own contribution. The TA requires at least one push
   from every team member. Review the combined report before the group submits
   the report PDF and repository link to Gradescope.

## AI Usage Report

All details on how AI was used to assist in the process (for both code and text) is available in the [AI_USE.md](AI_USE.md).