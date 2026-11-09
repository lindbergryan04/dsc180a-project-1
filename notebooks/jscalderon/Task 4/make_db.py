#!/usr/bin/env python
"""
Make JS Calderon's part of the DIII-D TGLF database (Task 4, Step 2).
Filled-in copy of make_db_template.py. Run from the folder that contains d3d_profile/ and tools/:
    python tools/make_db.py
Writes d3d_mydb_<SEED>/data/cNNNN/tglf_sat2/input.tglf and d3d_mydb_<SEED>/cases.csv.
cases.csv uses the same columns as Aryan's notebooks/aryan/task4/cases.csv.
Team seeds already used: Aryan 4, Sweekrit 27, Ryan 180401.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from d3d_profile import Profile, set_values

N_CASES = 1000
SEED = 3                             # JS Calderon; distinct from teammates (4, 27, 180401)
OUT = Path(f"d3d_mydb_{SEED}")

prof = Profile("d3d_profile")
rng = np.random.default_rng(SEED)
rows = []

for k in range(N_CASES):
    # TODO 1: go through the 13 radii in turn so each gets the same number of cases
    i = k % len(prof.names)
    radius, r_over_a = prof.names[i], prof.r[i]

    text = prof.base_input(r_over_a)
    base = prof.base_values(r_over_a)

    # TODO 2: random factors (same recipe for the whole team)
    f_Te = rng.uniform(0.6, 1.4)
    f_Ti = rng.uniform(0.6, 1.4)
    f_n = rng.uniform(0.6, 1.4)
    f_tau = rng.uniform(0.75, 1.25)
    f_nu = 10 ** rng.uniform(np.log10(0.5), np.log10(2.0))
    new = {
        "RLTS_1": f_Te * base["RLTS_1"],
        "RLTS_2": f_Ti * base["RLTS_2"],
        "RLNS_1": f_n * base["RLNS_1"],
        "RLNS_2": f_n * base["RLNS_1"],
        "TAUS_2": f_tau * base["TAUS_2"],
        "XNUE": f_nu * base["XNUE"],
    }
    text = set_values(text, new)

    case = f"c{k + 1:04d}"
    folder = OUT / "data" / case / "tglf_sat2"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.tglf").write_text(text)

    # TODO 3: record radius, factors and new values
    rows.append({"case": case, "radius": radius, "r_over_a": r_over_a,
                 "f_Te": f_Te, "f_Ti": f_Ti, "f_n": f_n, "f_tau": f_tau, "f_nu": f_nu, **new})

pd.DataFrame(rows).to_csv(OUT / "cases.csv", index=False)
print(f"wrote {N_CASES} cases to {OUT / 'data'}")
