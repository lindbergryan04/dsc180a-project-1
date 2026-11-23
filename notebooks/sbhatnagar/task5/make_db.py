#!/usr/bin/env python
"""
Make your part of the DIII-D TGLF database (Task 4, Step 2).
Run it from the folder that contains d3d_profile/ and tools/:
    python tools/make_db.py
It reads the 13 radii from d3d_profile/ and writes d3d_mydb_<SEED>/ next to it.

This version widens the factor ranges of the task sheet so that the cases also cover the second DIII-D profile
(data/original_data/d3d_profile_second), and it also varies the safety factor q (Q_LOC) and the magnetic shear s.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from d3d_profile import Profile, set_values

N_CASES = 8000                       # cases you make (task sheet: 1000)
SEED = 101                           # your own number, different from your teammates (27 was the first database)
ROOT = Path(__file__).resolve().parent.parent   # general-atomics-fusion/
OUT = ROOT / f"d3d_mydb_{SEED}"      # your output folder

# ranges of the random factors (task sheet values in the comments). The widened ones are chosen so that, at every radius,
# base value x factor reaches the values of the second profile (the factors it needs are in the comments).
F_TE = (0.45, 1.4)                   # RLTS_1 factor, uniform.            task sheet 0.6 - 1.4;  second profile needs 0.51 - 0.97
F_TI = (0.6, 1.4)                    # RLTS_2 factor, uniform.            unchanged;             second profile needs 0.65 - 1.28
F_N = (0.04, 1.4)                    # RLNS_1 = RLNS_2 factor, uniform in log10 (like f_nu). task sheet 0.6 - 1.4, uniform; second profile needs 0.05 - 1.39
F_TAU = (0.75, 1.35)                 # TAUS_2 factor, uniform.            task sheet 0.75 - 1.25; second profile needs 0.97 - 1.32
F_NU = (0.5, 2.0)                    # XNUE factor, uniform in log10.     unchanged;             second profile needs 1.46 - 1.94
F_Q = (0.65, 1.35)                   # q (Q_LOC) factor, uniform.         new;                   second profile has q 0.69 - 0.77 x the first profile
F_S = (0.5, 2.5)                     # magnetic shear s factor, uniform.  new;                   second profile has s 0.61 - 2.16 x the first profile

prof = Profile(ROOT / "data" / "original_data" / "d3d_profile")     # prof.names = ['r020', ..., 'r080'], prof.r = their r/a
rng = np.random.default_rng(SEED)
rows = []

for k in range(N_CASES):
    # choose one of the 13 radii, going through them in turn so that every radius gets the same number of cases
    i = k % len(prof.names)           # index 0 ... 12
    radius, r_over_a = prof.names[i], prof.r[i]

    text = prof.base_input(r_over_a)     # that radius's input.tglf
    base = prof.base_values(r_over_a)    # its values, e.g. base["RLTS_2"]

    # draw the random factors and compute the new inputs
    rlts_1_factor = rng.uniform(*F_TE)                                      # electron temperature gradient
    rlts_2_factor = rng.uniform(*F_TI)                                      # ion temperature gradient
    rlns_factor = 10 ** rng.uniform(np.log10(F_N[0]), np.log10(F_N[1]))     # density gradient, one factor for both species
    taus_2_factor = rng.uniform(*F_TAU)                                     # Ti/Te
    xnue_factor = 10 ** rng.uniform(np.log10(F_NU[0]), np.log10(F_NU[1]))   # collisionality
    q_factor = rng.uniform(*F_Q)                                            # safety factor q
    s_factor = rng.uniform(*F_S)                                            # magnetic shear s

    # TGLF takes the shear as Q_PRIME_LOC = s * (q / r)^2 with r = RMIN_LOC, so the base shear is recovered from the base
    # Q_PRIME_LOC, and the new Q_PRIME_LOC is set from the new q and the new s (changing q alone would change s)
    r_loc = base["RMIN_LOC"]
    s_base = base["Q_PRIME_LOC"] * (r_loc / base["Q_LOC"]) ** 2
    q_new = base["Q_LOC"] * q_factor
    s_new = s_base * s_factor

    new = {
        "RLTS_1": base["RLTS_1"] * rlts_1_factor,
        "RLTS_2": base["RLTS_2"] * rlts_2_factor,
        "RLNS_1": base["RLNS_1"] * rlns_factor,
        "RLNS_2": base["RLNS_1"] * rlns_factor,       # RLNS_1 = RLNS_2 (as in the task sheet: f_n * RLNS_1 for both)
        "TAUS_2": base["TAUS_2"] * taus_2_factor,
        "XNUE": base["XNUE"] * xnue_factor,
        "Q_LOC": q_new,
        "Q_PRIME_LOC": s_new * (q_new / r_loc) ** 2,
    }
    text = set_values(text, new)

    # write the case
    case = f"c{k + 1:04d}"
    folder = OUT / "data" / case / "tglf_sat2"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.tglf").write_text(text)

    # record what you drew; SHEAR is s, the quantity that was varied (Q_PRIME_LOC is derived from it)
    rows.append({
        "case": case,
        "radius": radius,
        "r_over_a": r_over_a,
        "rlts_1_factor": rlts_1_factor,
        "rlts_2_factor": rlts_2_factor,
        "rlns_factor": rlns_factor,
        "taus_2_factor": taus_2_factor,
        "xnue_factor": xnue_factor,
        "q_factor": q_factor,
        "s_factor": s_factor,
        **new,
        "SHEAR": s_new,
    })

pd.DataFrame(rows).to_csv(OUT / "cases.csv", index=False)
print(f"wrote {N_CASES} cases to {OUT / 'data'}")
