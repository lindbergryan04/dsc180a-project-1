"""Task 6 data preparation: clean both teams' runs, record the specialists, freeze the split.

``clean``        Keep shot-128913 deuterium runs at NBASIS 12/12 that pass the quality checks
                 and lie inside the Task 4 factor ranges at their radius. Writes
                 ``data/processed/task6/combined_nbasis12.csv``.
``specialists``  Record the three Task 5 specialists (architecture, transforms, and full
                 training settings) from a completed Task 5 search. Writes ``specialists.json``.
``split``        Freeze a 70/15/15 split of the standard Task 4 runs. Every case that was in
                 the Task 5 split keeps its Task 5 membership; the rest are split with seeds
                 180610/180611, stratified by radius. Writes ``split.csv``.

The committed ``specialists.json`` and ``split.csv`` in this folder are the frozen copies;
by default the commands write to ``notebooks/ryan/runs/task6/`` so they can be compared.
From the repository root::

    python notebooks/ryan/task6/prepare.py clean
    python notebooks/ryan/task6/prepare.py split
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[2]
sys.path.insert(0, str(REPO / "notebooks/ryan/task5"))
import data as task5  # noqa: E402

TEAM_A = REPO / "notebooks/ryan/task4/d3d_team_database.csv"
TEAM_B = REPO / "data/Task6_DIIID_Surrogate/team_b_d3d_database.csv"
COMBINED = REPO / "data/processed/task6/combined_nbasis12.csv"
RUNS = REPO / "notebooks/ryan/runs/task6"
FEATURES = ["RMIN_LOC", "RLTS_1", "RLTS_2", "RLNS_1", "TAUS_2", "XNUE"]
TARGETS = ["Q_e", "Q_i", "Gamma_e"]
TEAM_A_SHA256 = "1439a46a93c35b0dfe59de504b7a04a50dcf01b32562e99d984856f8e4c13653"
# The 42 numeric template fields of the 25 nominal 128913 profiles, sorted by radius.
TEMPLATE_SHA256 = "0a3b4f16fe697d593e2eda7142c072546f6b2b1979b860e01ce32b676e0ad9f1"
BOUNDS = pd.DataFrame({"RLTS_1": (0.60, 1.40), "RLTS_2": (0.60, 1.40), "RLNS_1": (0.60, 1.40),
                       "TAUS_2": (0.75, 1.25), "XNUE": (0.50, 2.00)}, index=["lower", "upper"])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(output):
    if sha256(TEAM_A) != TEAM_A_SHA256:
        raise ValueError("The Task 4 team table changed. Review it before assigning a discharge.")
    tables = {}
    for team, path in [("A", TEAM_A), ("B", TEAM_B)]:
        frame = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        frame["source_team"] = team
        frame["source_row"] = np.arange(2, len(frame) + 2)  # The CSV header is line 1.
        tables[team] = frame
    team_a, team_b = tables["A"], tables["B"]

    # Team A has no shot label; confirm its geometry matches Team B's labeled 128913 Task 4 runs.
    profile_fields = ["RMIN_LOC", "RMAJ_LOC", "DRMAJDX_LOC", "Q_LOC", "Q_PRIME_LOC", "KAPPA_LOC",
                      "S_KAPPA_LOC", "DELTA_LOC", "S_DELTA_LOC", "NS", "ZS_1", "ZS_2", "MASS_1", "MASS_2"]
    labeled = team_b.loc[team_b["set"].eq("task4") & team_b["discharge"].eq(128913) & team_b["isotope"].eq("D")]
    reference = pd.MultiIndex.from_frame(labeled[profile_fields].drop_duplicates())
    if not (pd.MultiIndex.from_frame(team_a[profile_fields]).isin(reference).all() and team_a["RMIN_LOC"].nunique() == 13):
        raise ValueError("Team A geometry does not match shot 128913.")
    team_a = team_a.assign(set="task4", discharge=128913, isotope="D", discharge_source="Task 4 archive")
    team_b = team_b.assign(discharge_source=np.where(team_b["discharge"].isna(), "unreported", "source CSV"))
    raw = pd.concat([team_a, team_b], ignore_index=True)
    raw["discharge"] = raw["discharge"].astype("Int64")
    raw["seed"] = raw["seed"].astype("Int64")

    run_inputs = [column for column in team_a.columns if column.isupper()]  # 42 TGLF input fields
    required = run_inputs + TARGETS
    raw[required] = raw[required].apply(pd.to_numeric, errors="coerce")
    shear = raw["Q_PRIME_LOC"] * (raw["RMIN_LOC"] / raw["Q_LOC"]) ** 2
    reported = raw["SHEAR"].notna() & shear.notna() & np.isfinite(shear)
    if not np.allclose(raw.loc[reported, "SHEAR"], shear[reported], rtol=1e-10, atol=1e-12):
        raise ValueError("The reported magnetic shear differs from the calculated value.")
    raw["SHEAR"] = raw["SHEAR"].fillna(shear.where(np.isfinite(shear)))

    # Quality checks, in order; each row keeps its first failing reason.
    status = pd.Series("kept", index=raw.index)
    basis12 = raw["NBASIS_MIN"].eq(12) & raw["NBASIS_MAX"].eq(12)
    status.loc[~basis12] = "other_basis"
    status.loc[status.eq("kept") & ~np.isfinite(raw[required]).all(axis=1)] = "nonfinite"
    status.loc[status.eq("kept") & ~(raw["TAUS_2"].gt(0) & raw["XNUE"].gt(0))] = "invalid_input"
    status.loc[status.eq("kept") & raw[["Q_e", "Q_i"]].lt(0).any(axis=1)] = "negative_heat"
    eligible = raw.loc[status.eq("kept")]
    repeated = eligible.loc[eligible.duplicated(run_inputs, keep=False)]
    if not repeated.empty:
        if repeated.groupby(run_inputs, sort=False)[TARGETS + ["set", "discharge"]].nunique(dropna=False).gt(1).any().any():
            raise ValueError("Repeated inputs have conflicting outputs or cohort labels.")
        status.loc[eligible.index[eligible.duplicated(run_inputs)]] = "duplicate"
    quality = raw.loc[status.eq("kept")].copy()
    if quality["case_id"].isna().any() or quality["case_id"].duplicated().any():
        raise ValueError("Missing or conflicting case IDs.")
    quality[["NBASIS_MIN", "NBASIS_MAX"]] = quality[["NBASIS_MIN", "NBASIS_MAX"]].astype(int)

    # Keep runs of shot 128913 at a known radius whose fixed inputs match that radius's profile
    # and whose five varied inputs lie inside the Task 4 factor ranges.
    nominal = raw.loc[basis12 & raw["set"].eq("answers_check") & raw["discharge"].eq(128913)
                      & raw["isotope"].eq("D"), run_inputs].sort_values("RMIN_LOC")
    if hashlib.sha256(nominal[sorted(run_inputs)].to_numpy(dtype="<f8").tobytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("The nominal 128913 profiles changed.")
    profile = nominal.set_index("RMIN_LOC")
    fixed = [c for c in run_inputs if c not in {*BOUNDS.columns, "RLNS_2", "RMIN_LOC"}]
    local = profile.reindex(quality["RMIN_LOC"]).set_axis(quality.index)
    measured = quality[BOUNDS.columns]
    checks = {
        "other_or_unknown_discharge": quality["discharge"].eq(128913).fillna(False).astype(bool),
        "other_species": quality["isotope"].eq("D").fillna(False).astype(bool),
        "unknown_radius": quality["RMIN_LOC"].isin(profile.index),
        "changed_fixed_input": quality[fixed].eq(local[fixed]).all(axis=1),
        "unequal_density_gradients": quality["RLNS_1"].eq(quality["RLNS_2"]),
        "outside_local_bounds": (measured.ge(local[BOUNDS.columns].mul(BOUNDS.loc["lower"], axis="columns"))
                                 & measured.le(local[BOUNDS.columns].mul(BOUNDS.loc["upper"], axis="columns"))).all(axis=1),
    }
    reason = np.select([~ok for ok in checks.values()], list(checks), default="kept")
    status.loc[quality.index] = reason
    kept = quality.loc[reason == "kept"].copy().reset_index(drop=True)

    front = ["case_id", "source_team", "source_row", "contributor", "set", "discharge",
             "discharge_source", "isotope", "NBASIS_MIN", "NBASIS_MAX"] + FEATURES + TARGETS
    kept = kept[front + [c for c in kept.columns if c not in front]]
    assert kept["discharge"].drop_duplicates().tolist() == [128913] and kept["isotope"].eq("D").all()
    assert np.isfinite(kept[required]).all().all() and kept[["Q_e", "Q_i"]].ge(0).all().all()
    assert not kept.duplicated(run_inputs).any()
    output.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output, index=False)
    flow = pd.crosstab([raw["source_team"], raw["set"]], status)
    print(flow.to_string())
    print(f"Kept {len(kept):,} runs; {kept['set'].eq('task4').sum():,} are standard Task 4 runs at 13 radii.")


def specialists(search_root, output):
    """Record each selected Task 5 specialist and its full training settings."""
    selection = json.loads((search_root / "results/selection.json").read_text())
    winners = selection["final"]["systems"]["separate"]
    fields = ["output_mode", "width", "depth", "hidden_activation", "bottleneck_width", "bottleneck_activation",
              "parameters_per_member", *[f"input_{f}" for f in task5.FEATURES]]
    design = {"features": task5.FEATURES, "output_activation": "identity", "standardize_inputs": True,
              "standardize_transformed_target": True, "specialists": {}}
    for target in TARGETS:
        configs = [json.loads((search_root / "runs" / winners[target] / f"repeat_{r}" / "config.json").read_text())["config"]
                   for r in (1, 2, 3)]
        keys = fields + [f"target_{target}"]
        spec = {k: configs[0][k] for k in keys}
        if spec["output_mode"] != target or any({k: c[k] for k in keys} != spec for c in configs[1:]):
            raise ValueError(f"Task 5 confirmation repeats disagree for {target}.")
        training = {k: v for k, v in configs[0].items() if k not in {"repeat", "member_seeds"}}
        design["specialists"][target] = {"a5_run": winners[target], **spec, "training_config": training}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(design, indent=2) + "\n")
    print(f"Wrote {output}")


def split(output):
    combined = pd.read_csv(COMBINED, usecols=["case_id", "source_team", "contributor", "case", "set", *FEATURES],
                           float_precision="round_trip")
    pool = combined.loc[combined["set"].eq("task4")].sort_values("case_id").reset_index(drop=True)
    assert np.array_equal(np.sort(pool["RMIN_LOC"].unique()), np.arange(20, 81, 5) / 100)

    # Match every Task 5 case to its run here and keep its Task 5 membership.
    old_split = pd.read_csv(REPO / "notebooks/ryan/task5/split.csv", usecols=["sample_id", "split"])
    aliases = {"member": "contributor", "RLTS_e": "RLTS_1", "RLTS_i": "RLTS_2", "RLNS_e": "RLNS_1", "TAUS_i": "TAUS_2"}
    old_inputs = []
    for name, path in task5.SOURCE_PATHS.items():
        if sha256(path) != task5.SOURCES[name]:
            raise ValueError(f"Task 5 source changed: {name}")
        frame = pd.read_csv(path, float_precision="round_trip", low_memory=False).rename(columns=aliases)
        frame["sample_id"] = Path(name).stem + ":" + frame["contributor"].astype(str) + ":" + frame["case"].astype(str)
        old_inputs.append(frame[["sample_id", "contributor", "case", *FEATURES]])
    old = old_split.merge(pd.concat(old_inputs, ignore_index=True), on="sample_id", how="left", validate="one_to_one")
    matches = old.merge(pool[["case_id", "contributor", "case", *FEATURES]], on=FEATURES, how="left",
                        validate="one_to_one", suffixes=("_a5", ""))
    assert matches["case_id"].notna().all() and matches["case_id"].is_unique
    assert matches["contributor_a5"].str.casefold().equals(matches["contributor"].str.casefold())
    assert matches["case_a5"].equals(matches["case"])
    old_cases = matches.set_index("case_id")[["sample_id", "split"]]

    from sklearn.model_selection import train_test_split
    assignments = pool[["case_id", "source_team", "contributor", "RMIN_LOC"]].copy()
    assignments["a5_sample_id"] = assignments["case_id"].map(old_cases["sample_id"])
    assignments["split"] = assignments["case_id"].map(old_cases["split"])
    n_eval = round(0.15 * len(pool))
    desired = pd.Series({"train": len(pool) - 2 * n_eval, "validation": n_eval, "test": n_eval})
    needed = desired - assignments["split"].value_counts().reindex(desired.index, fill_value=0)
    pending = assignments.loc[assignments["split"].isna()]
    assert (needed > 0).all() and needed.sum() == len(pending)
    train, holdout = train_test_split(pending.index, train_size=int(needed["train"]), random_state=180610,
                                      stratify=pending["RMIN_LOC"])
    validation, test = train_test_split(holdout, test_size=int(needed["test"]), random_state=180611,
                                        stratify=assignments.loc[holdout, "RMIN_LOC"])
    for partition, indices in [("train", train), ("validation", validation), ("test", test)]:
        assignments.loc[indices, "split"] = partition
    assert assignments["split"].value_counts().reindex(desired.index).equals(desired)
    assert not assignments["case_id"].isin(combined.loc[combined["set"].ne("task4"), "case_id"]).any()
    assignments = assignments[["case_id", "split", "RMIN_LOC", "source_team", "contributor", "a5_sample_id"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(assignments.to_csv(index=False, lineterminator="\n"))
    committed = CODE / "split.csv"
    print(f"Wrote {len(assignments):,} assignments; identical to committed split.csv:",
          committed.is_file() and sha256(output) == sha256(committed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["clean", "specialists", "split"])
    parser.add_argument("--search-root", type=Path, default=REPO / "notebooks/ryan/runs/task5/search",
                        help="completed Task 5 search (specialists only)")
    parser.add_argument("--output", type=Path, help="file to write (defaults described above)")
    args = parser.parse_args(argv)
    defaults = {"clean": COMBINED, "specialists": RUNS / "specialists.json", "split": RUNS / "split.csv"}
    output = (args.output or defaults[args.command]).resolve()
    if args.command != "clean" and output.is_relative_to(CODE):
        parser.error("The committed copies here are frozen; write to notebooks/ryan/runs/ and compare.")
    {"clean": lambda: clean(output), "specialists": lambda: specialists(args.search_root.resolve(), output),
     "split": lambda: split(output)}[args.command]()


if __name__ == "__main__":
    main()
