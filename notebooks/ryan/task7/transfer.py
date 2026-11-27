"""Task 7 (Part C, step 1): do surrogates built on DIII-D shot 128913 predict shot 193870?

Scores, without refitting anything, all 81 Task 6 learning-curve ensembles and the three
pooled Task 5 specialists (15 networks each) on two cohorts from shot 193870: the 13
supplied profile inputs and 517 perturbed cases (Team B's runs, NBASIS 12/12, nonnegative
heat flux). For reference the same models are scored on shot-128913 cohorts. Fluxes are
TGLF results in gyro-Bohm units; nothing here is compared with experiment or CGYRO.

From the repository root, after the Task 5 and Task 6 pipelines::

    python notebooks/ryan/task7/transfer.py

Inputs: ``data/Task7_Second_Discharge/d3d_profile_193870.tar.gz`` (course), Team B's Task 6
table, ``data/processed/task6/combined_nbasis12.csv``, the Task 6 split and checkpoints, and
the Task 5 search checkpoints with their interval freeze record. Output goes to ``--output``
(default ``notebooks/ryan/runs/task7``): ``transfer_scores.csv``,
``shot_193870_predictions.csv.gz``, and the scored shot-193870 cases.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[2]
sys.path.insert(0, str(REPO / "notebooks/ryan/task6"))
import learning_curves as lc  # noqa: E402  (imports the Task 5 engine)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

engine = lc.engine
FEATURES, TARGETS = engine.FEATURES, engine.TARGETS
RADII = np.arange(20, 81, 5) / 100
ARCHIVE = REPO / "data/Task7_Second_Discharge/d3d_profile_193870.tar.gz"
TEAM_B = REPO / "data/Task6_DIIID_Surrogate/team_b_d3d_database.csv"
TASK5_RUNS = REPO / "notebooks/ryan/runs/task5"
BOUNDS = pd.DataFrame({"RLTS_1": (.6, 1.4), "RLTS_2": (.6, 1.4), "RLNS_1": (.6, 1.4), "TAUS_2": (.75, 1.25),
                       "XNUE": (.5, 2.)}, index=["low", "high"])
GEOMETRY = ["RMAJ_LOC", "DRMAJDX_LOC", "Q_LOC", "Q_PRIME_LOC", "KAPPA_LOC", "S_KAPPA_LOC", "DELTA_LOC", "S_DELTA_LOC"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shot_193870(physical_inputs):
    """Team B's shot-193870 runs, checked against the supplied profile inputs."""
    templates = []
    with tarfile.open(ARCHIVE) as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith("/input.tglf"):
                continue
            params = {}
            for line in bundle.extractfile(member).read().decode().splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip() in physical_inputs:
                    params[key.strip()] = float(value.split("!")[0].split("#")[0].strip().replace("D", "e"))
            assert set(params) == set(physical_inputs)
            templates.append(params)
    template = pd.DataFrame(templates).set_index("RMIN_LOC").sort_index()
    np.testing.assert_array_equal(template.index, RADII)

    team_b = pd.read_csv(TEAM_B, low_memory=False, float_precision="round_trip")
    raw = team_b.loc[team_b["discharge"].eq(193870)].copy()
    assert set(raw["set"]) == {"second", "second_varied"} and raw["isotope"].eq("D").all()
    assert raw["case_id"].is_unique and raw["case_id"].notna().all()
    base = raw.loc[raw["set"].eq("second")].set_index("RMIN_LOC").sort_index()
    others = [k for k in physical_inputs if k != "RMIN_LOC"]
    np.testing.assert_array_equal(base[others].to_numpy(), template[others].to_numpy())

    # Every varied run must equal its template times its recorded factors, at six figures.
    varied = raw.loc[raw["set"].eq("second_varied")]
    perturbed = ["RLTS_1", "RLTS_2", "RLNS_1", "RLNS_2", "TAUS_2", "XNUE"]
    factors = ["f_Te", "f_Ti", "f_n", "f_n", "f_tau", "f_nu"]
    fixed = [c for c in others if c not in perturbed]
    expected = template.loc[varied["RMIN_LOC"]]
    np.testing.assert_array_equal(varied[fixed].to_numpy(), expected[fixed].to_numpy())
    product = expected[perturbed].to_numpy() * varied[factors].to_numpy()
    rounded = np.array([float(format(x, ".6g")) for x in product.ravel()]).reshape(product.shape)
    np.testing.assert_array_equal(varied[perturbed].to_numpy(), rounded)
    assert varied.groupby("RMIN_LOC").size().eq(40).all()

    keep = (raw[["NBASIS_MIN", "NBASIS_MAX"]].eq(12).all(axis=1) & np.isfinite(raw[physical_inputs + TARGETS]).all(axis=1)
            & raw[["TAUS_2", "XNUE"]].gt(0).all(axis=1) & raw[["Q_e", "Q_i"]].ge(0).all(axis=1))
    external = raw.loc[keep].copy()
    assert not external.duplicated(physical_inputs).any() and not external.duplicated(FEATURES).any()
    assert external["set"].value_counts().to_dict() == {"second_varied": 517, "second": 13}
    external["discharge"] = external["discharge"].astype(int)
    print(f"Shot 193870: {len(raw)} Team B runs, {len(external)} scored "
          f"({(~keep).sum()} excluded for basis, nonfinite values, or negative heat flux).")
    return external, template


def cohorts(external, template):
    original = pd.read_csv(lc.DATA_PATH, low_memory=False, float_precision="round_trip")
    split = pd.read_csv(lc.SPLIT_PATH)
    assert original["discharge"].eq(128913).all()
    assert set(split["case_id"]) == set(original.loc[original["set"].eq("task4"), "case_id"])
    assert not external["case_id"].isin(original["case_id"]).any()
    assert pd.MultiIndex.from_frame(external[FEATURES]).intersection(pd.MultiIndex.from_frame(original[FEATURES])).empty

    reference = original.loc[original["set"].eq("answers_check") & original["RMIN_LOC"].isin(RADII)]
    reference = reference.set_index("RMIN_LOC").sort_index()
    assert reference.index.equals(template.index)
    local = reference.reindex(external["RMIN_LOC"]).set_axis(external.index)
    x = external[BOUNDS.columns]
    outside = x.lt(local[BOUNDS.columns] * BOUNDS.loc["low"]) | x.gt(local[BOUNDS.columns] * BOUNDS.loc["high"])
    external["outside_128913_bounds"] = outside.any(axis=1)
    external["outside_features"] = outside.apply(lambda row: ", ".join(row.index[row]), axis=1)
    print("Shot-193870 cases outside the shot-128913 factor ranges, by input:")
    print(outside.assign(any_input=outside.any(axis=1), cohort=external["set"]).groupby("cohort").sum().T.to_string())
    changed = reference[GEOMETRY].ne(template[GEOMETRY]).sum()
    print("Radii (of 13) where each geometry input differs between the shots:", changed.to_dict())

    intermediate = original["set"].eq("answers_check") & ~original["RMIN_LOC"].isin(RADII)
    groups = {
        "193870_profile": external.loc[external["set"].eq("second")],
        "193870_varied": external.loc[external["set"].eq("second_varied")],
        "128913_profile": reference.reset_index(),
        "128913_intermediate_profile": original.loc[intermediate],
        "128913_intermediate_varied": original.loc[original["set"].eq("test_varied")],
        "128913_test": original.loc[original["case_id"].isin(split.loc[split["split"].eq("test"), "case_id"])],
    }
    assert {k: len(v) for k, v in groups.items()} == {
        "193870_profile": 13, "193870_varied": 517, "128913_profile": 13, "128913_intermediate_profile": 12,
        "128913_intermediate_varied": 1186, "128913_test": 3554}
    return groups


def model_jobs(task5_runs):
    """(metadata, [(checkpoint folder, expected checkpoint SHA-256)]) for every frozen model."""
    protocol = json.loads(lc.PROTOCOL.read_text())
    assert sha(lc.DATA_PATH) == protocol["inputs_sha256"]["combined"]
    assert sha(lc.SPLIT_PATH) == protocol["inputs_sha256"]["split"]
    freeze = json.loads((lc.RESULTS / "validation_freeze.json").read_text())
    plan = pd.read_csv(lc.PLAN, dtype={"draw": str})
    assert set(plan.fit_id) == set(freeze["checkpoint_sha256"]) and plan.fit_id.is_unique
    winners = {r["target"]: r["width"] for r in freeze["lowest_mean_validation_rmse_by_target_and_size"] if r["size"] == 16586}
    jobs = []
    for job in plan.itertuples(index=False):
        jobs.append(({"family": "Task 6", "fit_id": job.fit_id, "target": job.target, "train_size": job.size,
                      "draw": job.draw, "width": job.width,
                      "selected_at_full_size_on_128913": job.size == 16586 and winners[job.target] == job.width},
                     [(lc.RUNS / job.fit_id, freeze["checkpoint_sha256"][job.fit_id])]))
    selection = json.loads((task5_runs / "search/results/selection.json").read_text())["final"]["systems"]["separate"]
    hashes = json.loads((task5_runs / "pooled_intervals/freeze.json").read_text())["checkpoint_sha256"]
    for target in TARGETS:
        dirs = [f"{selection[target]}/repeat_{i}" for i in (1, 2, 3)]
        jobs.append(({"family": "Task 5 pooled", "fit_id": f"a5_pooled_{target}", "target": target,
                      "train_size": 5131, "draw": "pooled", "width": "original",
                      "selected_at_full_size_on_128913": False},
                     [(task5_runs / "search/runs" / d, hashes[d]) for d in dirs]))
    return jobs


def scores(y, yhat):
    error = yhat - y
    sse = float(error @ error)
    centered = y - y.mean()
    return {"n": len(y), "rmse": np.sqrt(sse / len(y)), "mae": np.abs(error).mean(),
            "bias_pred_minus_tglf": error.mean(), "r2": 1 - sse / (centered @ centered)}


def score_models(jobs, groups):
    test_rmse = pd.read_csv(lc.RESULTS / "per_fit_rmse.csv").set_index("fit_id")["test"]
    rows, frames = [], []
    for meta, files in jobs:
        target = meta["target"]
        models = []
        for folder, digest in files:
            assert sha(folder / "checkpoints.pt") == digest, folder
            model = engine.load_ensemble(folder)
            assert model.targets == [target] and len(model.models) == 5
            models.append(model)
        # Only the original widths were scored on the shot-128913 test set in Task 6.
        data = {k: v for k, v in groups.items() if k != "128913_test" or meta["width"] == "original"}
        batch = pd.concat([frame.assign(cohort=k) for k, frame in data.items()], ignore_index=True)
        members = np.concatenate([model.predict(batch[FEATURES])["members"][:, :, 0] for model in models], axis=0)
        assert np.isfinite(members).all()
        yhat, spread, y = members.mean(axis=0), members.std(axis=0, ddof=1), batch[target].to_numpy()
        for cohort, positions in batch.groupby("cohort", sort=False).indices.items():
            m = scores(y[positions], yhat[positions])
            rows.append({**meta, "members": len(members), "cohort": cohort, **m})
            if meta["family"] == "Task 6" and cohort == "128913_test":  # must reproduce Task 6 exactly
                np.testing.assert_allclose(m["rmse"], test_rmse.loc[meta["fit_id"]], rtol=1e-9)
            if cohort.startswith("193870_"):
                part = batch.iloc[positions][["case_id", "RMIN_LOC", "outside_128913_bounds"]].copy()
                frames.append(part.assign(fit_id=meta["fit_id"], target=target, cohort=cohort, tglf=y[positions],
                                          prediction=yhat[positions], error=yhat[positions] - y[positions],
                                          member_sd=spread[positions]))
    metrics, predictions = pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
    assert metrics["fit_id"].nunique() == 84 and not metrics[["fit_id", "cohort"]].duplicated().any()
    assert predictions.groupby("fit_id").size().eq(530).all()
    return metrics, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task5-runs", type=Path, default=TASK5_RUNS)
    parser.add_argument("--output", type=Path, default=REPO / "notebooks/ryan/runs/task7")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(CODE):
        parser.error("Choose an --output under notebooks/ryan/runs/, not the source folder.")
    engine.torch.set_num_threads(1)
    physical_inputs = [s for s in pd.read_csv(REPO / "notebooks/ryan/task4/d3d_team_database.csv", nrows=0) if s.isupper()]
    assert len(physical_inputs) == 42
    external, template = shot_193870(physical_inputs)
    external = external.sort_values(["set", "RMIN_LOC", "case_id"]).reset_index(drop=True)
    groups = cohorts(external, template)
    metrics, predictions = score_models(model_jobs(args.task5_runs.resolve()), groups)
    output.mkdir(parents=True, exist_ok=True)
    external.to_csv(output / "shot_193870_nbasis12.csv", index=False)
    metrics.to_csv(output / "transfer_scores.csv", index=False)
    (output / "shot_193870_predictions.csv.gz").write_bytes(
        gzip.compress(predictions.to_csv(index=False).encode(), compresslevel=6, mtime=0))
    full = metrics.loc[metrics.family.eq("Task 6") & metrics.train_size.eq(16586) & metrics.width.eq("original")]
    print(full.pivot(index="target", columns="cohort", values="r2").reindex(TARGETS).round(3).to_string())
    print(f"{metrics['fit_id'].nunique()} frozen ensembles scored; {len(predictions):,} shot-193870 predictions.")


if __name__ == "__main__":
    main()
