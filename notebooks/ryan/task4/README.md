# Task 4: DIII-D #128913 TGLF database

TGLF cases around DIII-D shot #128913 at 13 radii (r/a = 0.20–0.80), made by scaling five
local inputs while keeping geometry fixed at each radius. Each member contributed a seed.

| File | Purpose |
| --- | --- |
| `tglf_cases.py` | Check the 13 profile runs against the supplied answers; write case plans and input decks; collect fluxes after TGLF runs |
| `team_database.py` | Merge the four members' exports, check every row, and count cases outside the `scan_db` design limits |
| `d3d_database_seed180401.csv` | Ryan's 1,000 cases (seed 180401), run on DSMLP |
| `d3d_team_database.csv` | The 3,884-case team table that Tasks 5–7 train on |
| `task4_results.ipynb` | Read-only summary of the team table |

## Commands

From the repository root, after extracting the Task 2 and Task 4 archives (`DATA.md`).
TGLF itself needs a GACODE install (we used DSMLP); every other step runs anywhere.

```sh
t4=notebooks/ryan/task4
profile=data/Task4_DIIID_Database/task4_d3d/d3d_profile
tools=data/Task4_DIIID_Database/task4_d3d/tools

# Step 1: run TGLF on the 13 supplied radii, then compare with the answers
python $tools/run_tglf_parallel.py $profile -nproc 8
python $t4/tglf_cases.py check-profile --profile $profile

# Step 2: Ryan's 1,000 cases
python $t4/tglf_cases.py plan --profile $profile --seed 180401 --cases 1000 --output notebooks/ryan/runs/task4/seed180401
python $tools/run_tglf_parallel.py notebooks/ryan/runs/task4/seed180401/data -nproc 8
python $t4/tglf_cases.py collect --output notebooks/ryan/runs/task4/seed180401

# Step 3: team table
python $t4/team_database.py --profile $profile --scan-data data/Task2_TGLF_scan/scan_db/data --output notebooks/ryan/runs/task4/team
```

## The frozen team table

`d3d_team_database.csv` has 3,884 cases: Ryan 1,000, Sweekrit 884, Aryan 1,000, Joey 1,000.
It was built from Sweekrit's export at commit `340be1e`. He later regenerated it with all
1,000 cases, so Step 3 on the current repository gives 4,000 cases. To rebuild the frozen
table exactly, add `--sweekrit-table` with the older export:

```sh
mkdir -p data/processed/task4
git show 340be1e:notebooks/sbhatnagar/task4/data/d3d_mydb_27_table.csv > data/processed/task4/d3d_mydb_27_table_340be1e.csv
```

28 rows with a nonpositive heat flux are kept and flagged (`positive_heat_fluxes`). Negative
particle fluxes are kept. 2,614 of the 3,884 cases (67.3%) lie outside at least one Task 2
`scan_db` design interval, most often because `RLNS_1` is below 1.

## Checks run when this code moved into the repository

* `tglf_cases.py plan` reproduces every input value, input hash, and configuration hash in
  `d3d_database_seed180401.csv` (the random factors agree to 2.5e-16 relative; the committed
  run was on DSMLP). With `--varied-test` it reproduces the Task 5 varied-test plan byte for byte.
* `tglf_cases.py collect` reproduces all 240 Task 5 varied-test fluxes and output hashes from
  their raw TGLF outputs.
* `team_database.py` with the `340be1e` export reproduces `d3d_team_database.csv` byte for byte.
* `check-profile` was tested on synthetic outputs equal to, and perturbed from, the answers;
  the original Step 1 TGLF outputs stayed on DSMLP.
