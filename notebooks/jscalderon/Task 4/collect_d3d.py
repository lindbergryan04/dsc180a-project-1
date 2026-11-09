#!/usr/bin/env python
"""
Task 4 (JS Calderon): check Step 1 and build the DIII-D database table.

    python collect_d3d.py step1 <task4_d3d> <out.csv>   compare 13 radii with d3d_profile_answers.dat
    python collect_d3d.py table <task4_d3d> <seed> <out.csv>

The table has the same columns as notebooks/aryan/task4/d3d_database.csv:
cases.csv columns + Gamma_e, Gamma_i, Q_e, Q_i, Pi_e, Pi_i, valid_heat_flux.
NS is read from input.tglf.gen instead of assuming 8 flux values.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_fluxes(tglf_dir):
    """Return {Gamma_e, Gamma_i, Q_e, Q_i, Pi_e, Pi_i} from out.tglf.gbflux (electron = species 1)."""
    gen = {}
    for line in (tglf_dir / "input.tglf.gen").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            gen[parts[1]] = parts[0]
    ns = int(gen["NS"])
    vals = np.array([float(x) for x in (tglf_dir / "out.tglf.gbflux").read_text().split()])
    if vals.size != 4 * ns or not np.all(np.isfinite(vals)):
        raise ValueError(f"{tglf_dir}: expected {4 * ns} finite values, got {vals.size}")
    if float(gen["ZS_1"]) >= 0 or float(gen["ZS_2"]) <= 0:
        raise ValueError(f"{tglf_dir}: expected species 1 = electron, species 2 = ion")
    gamma, q, pi = vals[:ns], vals[ns:2 * ns], vals[2 * ns:3 * ns]
    return {"Gamma_e": gamma[0], "Gamma_i": gamma[1], "Q_e": q[0], "Q_i": q[1], "Pi_e": pi[0], "Pi_i": pi[1]}


def step1(root, out):
    ans = pd.read_csv(root / "d3d_profile" / "d3d_profile_answers.dat", sep=r"\s+", comment="#",
                      header=None, nrows=13, names=["case", "r_over_a", "Q_e", "Q_i", "Gamma_e"])
    rows = []
    for _, a in ans.iterrows():
        f = read_fluxes(root / "d3d_profile" / a.case / "tglf_sat2")
        rows.append({"case": a.case, "r_over_a": a.r_over_a,
                     **{f"{k}_ref": a[k] for k in ("Q_e", "Q_i", "Gamma_e")},
                     **{f"{k}_ours": f[k] for k in ("Q_e", "Q_i", "Gamma_e")}})
    df = pd.DataFrame(rows)
    for k in ("Q_e", "Q_i", "Gamma_e"):
        df[f"{k}_reldiff"] = (df[f"{k}_ours"] - df[f"{k}_ref"]).abs() / df[f"{k}_ref"].abs()
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    worst = df[[c for c in df if c.endswith("_reldiff")]].max().max()
    print(f"\nworst relative difference: {worst:.2e} -> {'PASS' if worst < 5e-4 else 'FAIL'} (3 digits)")
    return 0 if worst < 5e-4 else 1


def table(root, seed, out):
    db = root / f"d3d_mydb_{seed}"
    cases = pd.read_csv(db / "cases.csv")
    rows, problems = [], []
    for case in cases.case:
        try:
            rows.append({"case": case, **read_fluxes(db / "data" / case / "tglf_sat2")})
        except (OSError, ValueError, KeyError) as e:
            problems.append(f"{case}: {e}")
    df = cases.merge(pd.DataFrame(rows), on="case")
    df["valid_heat_flux"] = (df.Q_i > 0) & (df.Q_e > 0)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} / {len(cases)} cases to {out}; valid heat flux: {df.valid_heat_flux.sum()}")
    for p in problems:
        print("  missing/bad:", p)
    return 0


if __name__ == "__main__":
    mode, root = sys.argv[1], Path(sys.argv[2])
    sys.exit(step1(root, sys.argv[3]) if mode == "step1" else table(root, sys.argv[3], sys.argv[4]))
