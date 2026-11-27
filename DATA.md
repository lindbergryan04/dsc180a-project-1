# Data

The project uses TGLF simulations to explore and compare surrogate methods
for turbulent transport. The prediction targets are ion heat flux (`Q_i`),
electron heat flux (`Q_e`), and electron particle flux (`Gamma_e`). Current
report results use the starter scans and `scan_db`, with additional data for
evaluation. A new DIII-D training database is part of the planned case study;
the `scan_db` surrogate will provide a comparison model.

## Sources and access

Obtain the course-prepared datasets from the mentor's shared course materials.
The task sheets identify the datasets below but do not include direct download
links. Task 1 specifically identifies the course Google Drive as the source
of the prepared `SAT3_dataset`. Ask the mentor for access to any missing files.

| Dataset or archive | Contents | Use |
| --- | --- | --- |
| `task4_d3d.tar.gz` | DIII-D experiment #128913 inputs at 13 radii, from `r/a = 0.20` to `0.80` in steps of `0.05`; reference answers and helper scripts | Starting point for generating the project's DIII-D database |
| `scan_db` | 5,000 TGLF cases with inputs sampled around reference case `a_D`, including H, D, and T ion species | Train and evaluate the generic surrogate for comparison |
| `SAT3_dataset` (course version) | 43 plasma cases with linear/nonlinear CGYRO simulations and matching TGLF SAT2 runs | Initial exploration and a separate evaluation set; CGYRO comparison is an extension |
| `scan_db_outside` | Nine sets of 30 TGLF cases, each with one input outside the `scan_db` training range | Evaluate extrapolation and uncertainty estimates |
| `data_d3d` | One DIII-D reference case at `r/a = 0.5` and 300 TGLF runs with perturbed inputs | Compare surrogate and TGLF predictions under input uncertainty |

The original SAT3 CGYRO simulations are publicly available from
[H. G. Dudding, SAT3 Dataset, Zenodo record 14505760, version v1 (2024)](https://zenodo.org/records/14505760).
The course version adds inputs converted with Pyrokinetics and TGLF runs using
`SAT_RULE = 2`. Obtain that prepared version for Tasks 1–3; the public CGYRO
archive alone does not supply the course's added TGLF runs. Despite the SAT3
dataset name, the supplied TGLF predictions use the **SAT2** saturation rule.

The task sheets do not specify release versions or checksums for the
course-prepared datasets. Record the archive filename, download date, and
checksum when obtaining them so collaborators can identify the same inputs.

## Storage and expected paths

Keep downloaded inputs unchanged. The local archives are organized by task:

| Directory | Archives |
| --- | --- |
| `data/Task1_TGLF_SAT3_dataset/` | `SAT3_dataset.tar.gz` |
| `data/Task2_TGLF_scan/` | `scan_db.tar.gz` |
| `data/Task3_TGLF_Uncertainty/` | `data_d3d.tar.gz`, `scan_db_outside.tar.gz` |
| `data/Task4_DIIID_Database/` | `task4_d3d.tar.gz` |
| `data/Task5_DIIID_Surrogate/` | `team_b_d3d_database.csv`, `d3d_profile_test.tar.gz` (keep both as files) |
| `data/Task6_DIIID_Surrogate/` | `team_b_d3d_database.csv` (the later, larger release) |
| `data/Task7_Second_Discharge/` | `d3d_profile_193870.tar.gz` (keep as a file) |

Extract each Task 1–4 archive within its task directory, preserving the archive's
internal folder structure. The Task 5–7 files are read as they are. These task
directories are local data locations and will need to be populated after cloning
the repository.

### Tasks 5–7 inputs

The Tasks 5–7 handout ("A DIII-D TGLF Surrogate with Error Bars") supplies the
12 intermediate-radius profile cases and the second discharge. The two teams'
Task 4 databases are combined for training. Our own team table is committed as
`notebooks/ryan/task4/d3d_team_database.csv`; the other files are not committed.
`notebooks/ryan/task5/data.py` refuses a Task 5 Team B table whose hash differs
from the one below; check the others with `shasum -a 256` after downloading.

| File | Contents | SHA-256 (first 12) |
| --- | --- | --- |
| `Task5_DIIID_Surrogate/team_b_d3d_database.csv` | Team B's Task 4 database for shot 128913, 3,500 rows | `32be24624677` |
| `Task5_DIIID_Surrogate/d3d_profile_test.tar.gz` | `d3d_profile_test/`: TGLF inputs at the 12 radii r/a = 0.225–0.775 and their answers | `a8d0e48edbdd` |
| `Task6_DIIID_Surrogate/team_b_d3d_database.csv` | Later Team B release, 75,928 rows: Task 4 runs, intermediate-radius variations, NBASIS resolution checks, wider-range scans, a general scan, and shot 193870 | `7fdca421d6a6` |
| `Task7_Second_Discharge/d3d_profile_193870.tar.gz` | `d3d_profile_second`: inputs for DIII-D shot 193870 at the same 13 radii | `a3da767e55bb` |

Generated tables go to `data/processed/` (for example
`data/processed/task6/combined_nbasis12.csv`, written by
`notebooks/ryan/task6/prepare.py clean`).

As analysis code is added, use `data/generated/` for new TGLF cases and
simulation outputs, and `data/processed/` for parsed and cleaned tables.

The repository's `.gitignore` excludes all contents of `data/` except this
README, including generated tables. Shared course data can also remain outside
the repository; configure analysis code to read its actual location rather
than duplicating entire datasets.

Preserve the internal directories when unpacking datasets. Expected paths
relative to each extracted dataset are:

| Dataset root | Case location |
| --- | --- |
| `task4_d3d/` | `d3d_profile/r020/tglf_sat2/` through `d3d_profile/r080/tglf_sat2/` |
| `scan_db/` | `data/r0001/tglf_sat2/` through `data/r5000/tglf_sat2/` |
| `SAT3_dataset/` | `data/<case>/tglf_sat2/`, for example `data/a_D/tglf_sat2/` |
| `scan_db_outside/` | `ood_<input>/data/<case>/tglf_sat2/` |
| `data_d3d/` | `d3d_db/tglf_sat2/` and `d3d_uq/data/u001/tglf_sat2/` through `u300/tglf_sat2/` |

Retain sampling metadata such as `samples.csv` when supplied. The Task 4
archive also contains `d3d_profile/d3d_profile_answers.dat` and a `tools/`
directory with its own `README.txt`.

## Building the DIII-D database

Follow [Task 4](tasks/Task4_DIIID_database.md) for the full generation
procedure. First run the 13 supplied profile cases and compare their fluxes
with `d3d_profile_answers.dat`. The handout requires agreement to at least
three digits before generating the larger database.

Each team member then generates 500–1,000 cases, distributed evenly across
the 13 radii, for at least 2,000 cases across the team. Start from the supplied
`input.tglf` at the selected radius and apply these factors:

| Input | Sampling rule |
| --- | --- |
| `RLTS_1` and `RLTS_2` | Multiply each by a separately drawn factor between `0.6` and `1.4` |
| `RLNS_1` and `RLNS_2` | Multiply both by the same factor between `0.6` and `1.4`, keeping them equal |
| `TAUS_2` | Multiply by a factor between `0.75` and `1.25` |
| `XNUE` | Multiply by a factor between `0.5` and `2`, sampling `log10(factor)` uniformly |

Keep the geometry at its supplied value for that radius and retain the
species and TGLF model settings. Give each team member a distinct random seed,
and save the seed, case identifier, radius, and sampled factors. Check one
generated input against its source before running the full batch. Run
simulations in writable working copies, preserving the downloaded originals.

Combine the parsed results using unique case identifiers, and check for
duplicate input cases. Record the TGLF version, generation code revision, and
any removed cases alongside the resulting table.

## Reading inputs and fluxes

For completed TGLF runs, read these two files from each `tglf_sat2/` folder:

- **`input.tglf.gen`**: one value and parameter name per line, including
  defaults filled in by TGLF. Use this file to build the feature table.
  `input.tglf` is the simulation input and may omit parameters set by default.
- **`out.tglf.gbflux`**: transport fluxes normalized to gyro-Bohm units.
  Read `NS` from `input.tglf.gen` first; the output contains `4 × NS` values.

The flux values are stored in blocks, with one entry per species in each block:

| Block | Quantity | Role in this project |
| --- | --- | --- |
| 1 | Particle flux `Gamma` | Electron value is a prediction target |
| 2 | Heat flux `Q` | Electron and ion values are prediction targets |
| 3 | Toroidal momentum flux `Pi` | Not a current prediction target |
| 4 | Energy exchange `S` | Not a current prediction target |

For `NS = 2`, the order is:

```text
Gamma_1 Gamma_2 Q_1 Q_2 Pi_1 Pi_2 S_1 S_2
```

Determine the electron and ion species from `ZS_<species>` (negative for
electrons, positive for ions), and identify the ion isotope using its mass.
The supplied two-species cases use species 1 for electrons and species 2 for
ions. The reader should still use `NS` rather than assume eight output values.
The block definitions above follow the correction in
[Task 3](tasks/Task3_surrogate_uncertainty.md).

## Quality checks and derived tables

Produce one row per case, retaining its dataset name, case identifier, source
path, plasma inputs, target fluxes, and quality flags. Report missing files,
parse errors, nonfinite values, and unexpected flux-vector lengths explicitly.
Check the case counts against the dataset descriptions above.

Flag negative heat fluxes and exclude them from the cleaned training data as
directed in the project slides, while preserving the raw outputs and a record
of exclusions. Inspect zero heat fluxes before applying logarithms. Particle
flux can be positive or negative, so a negative `Gamma_e` alone is not grounds
for removing a case.

Ryan's Task 1 command writes `sat3_tglf_table.csv` under `notebooks/ryan/runs/task1/`.
Preserve train, validation, and test membership with the
processed data. Task 2 specifies a fixed-seed 70%/15%/15% split for `scan_db`;
Task 3 reuses that split. Keep the SAT3 cases separate from `scan_db` training
and validation.

The [root README](README.md) gives the commands for Ryan’s analysis and the
setup for Sweekrit’s notebooks. Ryan’s commands read these raw task directories
and write derived tables, trained models, and figures under `notebooks/ryan/runs/`.

## Project references

- [Task 1: SAT3 starter database](tasks/Task1_SAT3database_revised.md)
- [Task 2: TGLF surrogate models](tasks/Task2_TGLF_surrogate.md)
- [Task 3: Surrogate uncertainty](tasks/Task3_surrogate_uncertainty.md)
- [Task 4: DIII-D database](tasks/Task4_DIIID_database.md)
- [Project scope and schedule](reference_slides/110226_project.pptx)
