# dsc180a-project-checkpoint

DSC 180A Quarter 1 group project.

## Project overview

TODO: Add the project title and research question.

## Mentors:
Igor Sfiligoi (SDSC)
Emily Belli (GA)
Jeff Candy (GA)

## Group members:
Ryan Lindberg
Sweekrit Bahtnagar
Joseph Calderon (Joey)
Aryan Nath

## Repository structure

| Path | Contents |
| --- | --- |
| `src/` | Reserved for reusable data-processing, modeling, and evaluation code. |
| `notebooks/` | Reserved for exploratory analysis and experiment notebooks. |
| `data/README.md` | Dataset descriptions, access instructions, file formats, and quality checks. |
| `data/Task1_TGLF_SAT3_dataset/` | Task 1 SAT3 starter dataset. |
| `data/Task2_TGLF_scan/` | Task 2 TGLF parameter-scan data. |
| `data/Task3_TGLF_Uncertainty/` | Task 3 uncertainty and extrapolation datasets. |
| `data/Task4_DIIID_Database/` | Task 4 DIII-D database materials. |
| `emily_slides/` | Mentor slides covering transport, fluxes, uncertainty, and the project plan. |
| `references/` | Research papers used for background and related work. |
| `tasks/` | Mentor-provided instructions for Tasks 1–4. |
| `report/` | Report materials; currently contains the course LaTeX template PDF. |
| `requirements.txt` | Python dependency list, to be populated as code is added. |
| `.gitignore` | Excludes datasets, local environments, credentials, and generated build files. |

Dataset files under `data/` are stored locally and excluded from Git; only
`data/README.md` is tracked when committed.

## Data

See [data/README.md](data/README.md) for data access and storage instructions.

## Environment

Dependencies have not been selected yet. Add the packages and versions used by
this project to `requirements.txt`, and document the Python version here.

Create a local environment from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce results

TODO: Add the commands to prepare the data, run the analysis, and generate
figures and tables. No analysis pipeline or results are included yet.

## Report

Store the LaTeX source and references in `report/`. The report checkpoint
includes the title, abstract, and introduction (problem context, prior work,
and relevant data).
