"""Task 4, Step 3: combine the four DIII-D #128913 contributions into one team database.

Each member exported a CSV of TGLF cases for shot 128913 (Ryan seed 180401, Sweekrit
seed 27, Aryan seed 4, Joey seed 3). This script puts them in one schema, checks every row
against the supplied radius profile, rejects repeated inputs, lists Sweekrit's planned cases
that have no exported result, and counts how many cases fall outside the Task 2 ``scan_db``
design limits. It writes ``d3d_team_database.csv``, ``scan_db_coverage.csv``,
``seed27_cases_without_results.csv`` and ``manifest.json`` (input hashes).

Run from the repository root after extracting the Task 2 and Task 4 course archives
(see ``DATA.md``)::

    python notebooks/ryan/task4/team_database.py \\
        --profile data/Task4_DIIID_Database/task4_d3d/d3d_profile \\
        --scan-data data/Task2_TGLF_scan/scan_db/data \\
        --output notebooks/ryan/runs/task4/team

The frozen team table used by Tasks 5-7 is ``d3d_team_database.csv`` beside this script. It
was built from Sweekrit's export at commit 340be1e, before he regenerated it. To rebuild that
exact table, pass the older export with ``--sweekrit-table`` (see the Task 4 README).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent))
from d3d_profile import read_values  # noqa: E402  (same helper as the course task4_d3d/tools)
from tglf_data import load_cases  # noqa: E402

MEMBERS = REPO / "notebooks"
FROZEN_TABLE = HERE / "d3d_team_database.csv"
SOURCES = [
    # contributor, seed, exported CSV, where the varied inputs come from, source_file label.
    # The labels are the file names used in the original merge (Joey's copy was renamed then).
    ("Ryan", 180401, MEMBERS / "ryan/task4/d3d_database_seed180401.csv", "case inputs", "d3d_database_seed180401.csv"),
    ("Sweekrit", 27, MEMBERS / "sbhatnagar/task4/data/d3d_mydb_27_table.csv", "case inputs", "d3d_mydb_27_table.csv"),
    ("Aryan", 4, MEMBERS / "aryan/task4/d3d_database.csv", "case plan", "d3d_database.csv"),
    ("Joey", 3, MEMBERS / "jscalderon/task4/d3d_database.csv", "case plan", "joey_d3d_database.csv"),
]
SWEEKRIT_PLAN = MEMBERS / "sbhatnagar/task4/data/d3d_mydb_27_cases.csv"
VARIED = ["RLTS_1", "RLTS_2", "RLNS_1", "TAUS_2", "XNUE"]
FEATURES = ["r_over_a", *VARIED]
FACTORS = ["f_Te", "f_Ti", "f_n", "f_tau", "f_nu"]
TARGETS = ["Q_e", "Q_i", "Gamma_e"]
TEAM_GOAL = 2000
RADII = [f"r{r:03d}" for r in range(20, 81, 5)]
FACTOR_ALIASES = dict(zip(["rlts_1_factor", "rlts_2_factor", "rlns_factor", "taus_2_factor", "xnue_factor"], FACTORS))
ALIASES = {**FACTOR_ALIASES, "energy_flux_1": "Q_e", "energy_flux_2": "Q_i", "particle_flux_1": "Gamma_e"}
# Inclusive design limits from the Task 2 handout, in FEATURES order.
SCAN_BOUNDS = dict(zip(FEATURES, [(0.25, 0.75), (1.5, 3.5), (1.5, 3.5), (1, 3), (0.5, 1.5), (0.01, 1)]))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_baseline(profile):
    baseline = pd.DataFrame.from_dict(
        {r: read_values((profile / r / "tglf_sat2" / "input.tglf").read_text()) for r in RADII}, orient="index")
    return baseline


def read_contribution(name, seed, path, input_source, filename, baseline):
    """Return one member's cases in the shared team schema."""
    input_columns = baseline.columns.tolist()
    fixed = [key for key in input_columns if key not in VARIED + ["RLNS_2"]]
    team_columns = (["case_id", "contributor", "seed", "case", "source_file", "radius", "r_over_a",
                     "varied_inputs_source", "fixed_inputs_source"]
                    + FACTORS + input_columns + TARGETS + ["positive_heat_fluxes"])
    df = pd.read_csv(path, float_precision="round_trip").rename(columns=ALIASES)
    if not df.columns.is_unique:
        raise ValueError(f"{filename} has ambiguous columns after alias mapping.")
    required = {"case", "radius", *FACTORS, *VARIED, "RLNS_2", *TARGETS}
    required.update({"r_over_a"} if input_source == "case plan" else fixed)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
    if df["case"].isna().any() or df["case"].duplicated().any():
        raise ValueError(f"{filename} has a missing or repeated source case ID.")
    if not df["radius"].isin(RADII).all():
        raise ValueError(f"{filename} has an unknown radius label.")
    if "seed" in df and not df["seed"].eq(seed).all():
        raise ValueError(f"{filename} disagrees with its configured seed.")
    case_ids = "seed" + str(seed) + "_" + df["case"].astype(str)
    if "case_id" in df and not df["case_id"].eq(case_ids).all():
        raise ValueError(f"{filename} has inconsistent case IDs.")
    for key, expected in {"UNITS": "GYRO", "species_type_1": "e", "species_type_2": "D"}.items():
        if key in df and not df[key].eq(expected).all():
            raise ValueError(f"{filename} has unexpected {key} values.")

    radius_value = df["r_over_a"] if "r_over_a" in df else df["RMIN_LOC"]
    expected_r = df["radius"].map(baseline["RMIN_LOC"])
    if not np.allclose(radius_value, expected_r, rtol=0, atol=1e-12):
        raise ValueError(f"{filename} has inconsistent radius labels and coordinates.")

    # A case-plan export omits the fixed settings; fill them from the supplied radius profile.
    absent = [key for key in fixed if key not in df]
    if absent:
        inferred = baseline.loc[df["radius"], absent].set_axis(df.index)
        df = pd.concat([df, inferred], axis=1)
    reported = df[FACTORS + input_columns + TARGETS].apply(pd.to_numeric, errors="raise").astype(float)
    positive = reported["Q_e"].gt(0) & reported["Q_i"].gt(0)
    if "valid_heat_flux" in df and not df["valid_heat_flux"].eq(positive).all():
        raise ValueError(f"{filename} has a heat-flux flag inconsistent with its values.")
    ids = pd.DataFrame({
        "case_id": case_ids, "contributor": name, "seed": seed, "case": df["case"],
        "source_file": filename, "radius": df["radius"], "r_over_a": radius_value.astype(float),
        "varied_inputs_source": input_source,
        "fixed_inputs_source": ("radius profile" if len(absent) == len(fixed) else
                                "case inputs + radius profile" if absent else "case inputs"),
        "positive_heat_fluxes": positive,
    })
    return pd.concat([ids, reported], axis=1)[team_columns]


def check_team(team, baseline):
    """Check every row against its radius profile and the Task 4 sampling limits."""
    input_columns = baseline.columns.tolist()
    fixed = [key for key in input_columns if key not in VARIED + ["RLNS_2"]]
    numbers = FACTORS + input_columns + TARGETS
    if not np.isfinite(team[numbers].to_numpy(dtype=float)).all():
        raise ValueError("A required factor, input or flux is missing or nonfinite.")
    if team["case_id"].duplicated().any() or not team["radius"].isin(RADII).all():
        raise ValueError("Case IDs must be unique and radius labels must match the supplied profiles.")
    ref = baseline.loc[team["radius"]].reset_index(drop=True)
    for key in fixed:
        if not np.allclose(team[key], ref[key], rtol=5e-6, atol=1e-12):
            raise ValueError(f"{key} differs from the baseline for its labeled radius.")
    if not team["RLNS_1"].eq(team["RLNS_2"]).all():
        raise ValueError("Electron and ion density gradients differ.")
    limits = [(0.6, 1.4)] * 3 + [(0.75, 1.25), (0.5, 2.0)]
    for key, factor, (low, high) in zip(VARIED, FACTORS, limits):
        if not team[factor].between(low, high).all():
            raise ValueError(f"{factor} is outside the assignment's limits.")
        if not np.allclose(team[key], ref[key] * team[factor], rtol=5e-6, atol=1e-12):
            raise ValueError(f"{key} disagrees with its baseline and recorded factor.")
    # Repeats at the course helper's six-significant-figure precision would write the same input.
    written = team[FEATURES].apply(lambda col: col.map(lambda x: format(x, ".6g")))
    duplicates = team.loc[written.duplicated(keep=False), ["case_id", *FEATURES, *TARGETS]]
    if len(duplicates):
        print(duplicates.sort_values(FEATURES).to_string())
        raise ValueError("Repeated input configuration. Inspect the source rows before exporting.")


def seed27_missing(team, plan_path):
    """List Sweekrit's planned seed-27 cases that have no row in his exported results."""
    plan = pd.read_csv(plan_path, float_precision="round_trip").rename(columns=FACTOR_ALIASES)
    s27 = team.loc[team["seed"].eq(27)].set_index("case")
    plan = plan.set_index("case")
    if not plan.index.is_unique or plan.index.hasnans or len(s27.index.difference(plan.index)):
        raise ValueError("The seed 27 plan and results have incompatible case IDs.")
    matched = plan.loc[s27.index]
    if not s27["radius"].eq(matched["radius"]).all() or not np.allclose(
            s27[FACTORS], matched[FACTORS], rtol=1e-12, atol=1e-12):
        raise ValueError("The seed 27 plan and exported draws disagree.")
    missing = plan.loc[~plan.index.isin(s27.index)].reset_index()
    missing.insert(0, "seed", 27)
    print(f"Seed 27: {len(plan):,} planned, {len(s27):,} exported, {len(missing):,} without an exported result.")
    return missing


def scan_coverage(team, scan):
    outside = pd.DataFrame(index=team.index)
    rows = []
    for key, (low, high) in SCAN_BOUNDS.items():
        below, above = team[key].lt(low), team[key].gt(high)
        outside[key] = below | above
        rows.append({"feature": key, "scan_design_min": low, "scan_design_max": high,
                     "scan_observed_min": scan[key].min(), "scan_observed_max": scan[key].max(),
                     "team_min": team[key].min(), "team_max": team[key].max(),
                     "below": below.sum(), "above": above.sum(), "outside_pct": 100 * outside[key].mean()})
    coverage = pd.DataFrame(rows).set_index("feature")
    any_outside = outside.any(axis=1)
    print(coverage.round(4).to_string())
    print(f"{any_outside.sum():,}/{len(team):,} ({any_outside.mean():.1%}) are outside at least one "
          "of the six scan design intervals.")
    print(f"{(~any_outside).sum():,} are within all six intervals.")
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, required=True,
                        help="task4_d3d/d3d_profile directory from the Task 4 course archive")
    parser.add_argument("--scan-data", type=Path, required=True, help="scan_db/data case directory (Task 2)")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the team files")
    parser.add_argument("--sweekrit-table", type=Path,
                        help="use this seed-27 export instead of the current one in notebooks/sbhatnagar")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"{args.output} is not empty. Choose a new --output directory.")

    sources = list(SOURCES)
    if args.sweekrit_table:
        name, seed, _, input_source, label = sources[1]
        sources[1] = (name, seed, args.sweekrit_table.resolve(), input_source, label)
    baseline = load_baseline(args.profile)
    team = pd.concat([read_contribution(*source, baseline) for source in sources], ignore_index=True)
    print(team.groupby(["contributor", "seed"], sort=False).size().rename("exported_cases").to_string())
    check_team(team, baseline)
    print(f"{len(team):,} distinct cases across all contributors. Assignment minimum: {TEAM_GOAL:,}.")
    print(f"{(~team['positive_heat_fluxes']).sum()} rows retained with at least one nonpositive heat flux.")
    missing = seed27_missing(team, SWEEKRIT_PLAN)

    scan = load_cases(args.scan_data).reset_index().rename(columns={"RMIN_LOC": "r_over_a"})
    if len(scan) != 5000 or scan["case"].nunique() != len(scan) or not np.isfinite(scan[FEATURES]).all().all():
        raise ValueError("Expected 5,000 distinct scan_db cases with finite inputs.")
    coverage = scan_coverage(team, scan)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    team.to_csv(out / "d3d_team_database.csv", index=False)
    missing.to_csv(out / "seed27_cases_without_results.csv", index=False)
    coverage.to_csv(out / "scan_db_coverage.csv")
    manifest = {
        "inputs_sha256": {name: sha256(path) for name, _, path, _, _ in sources},
        "seed27_plan_sha256": sha256(SWEEKRIT_PLAN),
        "rows": len(team),
        "team_database_sha256": sha256(out / "d3d_team_database.csv"),
        "matches_frozen_table": sha256(out / "d3d_team_database.csv") == sha256(FROZEN_TABLE),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(team):,} rows to {out / 'd3d_team_database.csv'}")
    print("Identical to the frozen table used by Tasks 5-7:", manifest["matches_frozen_table"])


if __name__ == "__main__":
    main()
