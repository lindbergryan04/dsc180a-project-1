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
| `notebooks/` | Individual experiment notebooks; Ryan's current analyses and saved outputs are in `notebooks/ryan/`. |
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

The shared dependency specification is not complete. `requirements.txt` is
currently a placeholder. Add the packages and versions used by the submitted
experiments and document the Python version here before treating this setup
as a reproducible environment.

Create a local environment from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce results

The notebooks in `notebooks/ryan/` contain analysis and saved results used in
the report. Reproduction from a fresh clone has not been verified. Some
notebooks refer to helpers, model artifacts, and paths outside this repository.

Remaining work: include the required helpers or document how to obtain them,
complete the dependency specification, configure data and artifact paths, and
document the commands and execution order needed to reproduce the report's
figures and tables.

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
