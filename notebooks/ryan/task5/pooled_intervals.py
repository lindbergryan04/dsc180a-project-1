"""Task 5 pooled specialists and 90% prediction intervals.

The search selects one five-network specialist per flux and trains it in three seed groups.
Pooling the groups gives 15 networks per flux; each network's prediction is averaged in
gyro-Bohm units after its own inverse target transform. Intervals are
``mean ± q · max(member SD, 1% of training SD)``, with ``q`` the split-conformal 90% quantile
of ``|error| / scale`` on the validation set (five-fold out of fold for validation itself).
The shared reference ensemble (``joint_5``) gets the same treatment for comparison.

Subcommands, run in order from the repository root after ``search.py all`` and ``report.py``:

``pool``       score the 15-network pool against each five-network group
``calibrate``  fit the interval multipliers and freeze forecasts for the 240 varied cases
``score``      score forecasts and intervals on validation, test, profile, and varied cases

Outputs go to ``--output`` (default ``notebooks/ryan/runs/task5/pooled_intervals``). The
original varied-case forecasts were frozen (``freeze.json``) before TGLF was run on those
inputs; only those forecasts are a prospective test of the committed TGLF answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import data  # noqa: E402
import engine  # noqa: E402

RUNS = CODE.parents[2] / "notebooks/ryan/runs/task5"
VARIED = CODE / "varied_test.csv"
REPEATS = (1, 2, 3)
GROUPS = [f"repeat_{repeat}" for repeat in REPEATS]
COUNTS = {"validation": 1100, "test": 1100, "profile_test": 12}
COVERAGE, FOLDS, FOLD_SEED = 0.90, 5, 180521
MODEL_KEYS = ["model", "target"]
CASE_TARGET = ["sample_id", "target"]
NUMERIC = ["r_over_a", "truth", "prediction", "spread", "lower", "upper"]


def read_csv(path, **kw):
    return pd.read_csv(path, float_precision="round_trip", **kw)


def selected_specialists(root):
    return json.loads((Path(root) / "results/selection.json").read_text())["final"]["systems"]


def load_predictor(root=RUNS / "search", *, repeats=REPEATS):
    """Return ``predict(frame) -> DataFrame[Q_e, Q_i, Gamma_e]`` using 5 networks per repeat."""
    winners = selected_specialists(root)["separate"]
    models = {target: [engine.load_ensemble(Path(root) / "runs" / winners[target] / f"repeat_{r}") for r in repeats]
              for target in engine.TARGETS}

    def predict(frame):
        return pd.DataFrame({target: np.mean([e.predict(frame)["prediction"][:, 0] for e in ensembles], axis=0)
                             for target, ensembles in models.items()}, index=frame.index)
    return predict


# ---------------------------------------------------------------- pool
def archived_predictions(root, case_id, target):
    """Each repeat's saved ensemble predictions for one specialist, plus their 15-network pool."""
    frames, seeds_seen = [], set()
    for repeat in REPEATS:
        folder = root / "runs" / case_id / f"repeat_{repeat}"
        metadata = json.loads((folder / "config.json").read_text())
        seeds = [str(n) for n in metadata["config"]["member_seeds"]]
        if len(set(seeds)) != 5 or seeds_seen.intersection(seeds) or metadata["status"] != "complete":
            raise ValueError(f"Incomplete or overlapping seed groups in {folder}")
        seeds_seen.update(seeds)
        rows = read_csv(folder / "predictions.csv", dtype={"member_seed": str})
        if not (rows.target.eq(target) & rows.case_id.eq(case_id) & rows.repeat.eq(repeat)).all():
            raise ValueError(f"Unexpected target, repeat, or architecture in {folder}")
        keys = ["split", "sample_id"]
        outcomes = rows[keys + ["truth", "r_over_a"]].drop_duplicates().set_index(keys).sort_index()
        wide = rows.pivot(index=keys, columns="member_seed", values="prediction").sort_index()
        if (not outcomes.index.is_unique or not outcomes.index.equals(wide.index)
                or set(wide.columns) != set(seeds + ["ensemble"])
                or wide.groupby(level="split").size().to_dict() != COUNTS):
            raise ValueError(f"Incomplete or inconsistent case records in {folder}")
        np.testing.assert_allclose(wide[seeds].mean(axis=1), wide.ensemble, rtol=1e-13, atol=1e-13)
        if frames:
            pd.testing.assert_frame_equal(outcomes, frames[0][["truth", "r_over_a"]])
        frames.append(outcomes.join(wide.ensemble.rename(f"repeat_{repeat}")))
    result = frames[0].join([frame.iloc[:, -1] for frame in frames[1:]])
    result["pooled_15"] = result[GROUPS].mean(axis=1)
    return result.reset_index()


def rmse_r2(truth, predictions):
    mse = np.square(predictions - truth[:, None]).mean(axis=0)
    return np.sqrt(mse), 1 - mse / truth.var()


def paired_interval(test, model):
    """95% interval for pooled minus five-network RMSE, resampling cases within each radius."""
    groups = test.groupby("r_over_a").indices.values()
    squared = (test[["pooled_15", model]].to_numpy() - test.truth.to_numpy()[:, None]) ** 2
    rng = np.random.default_rng(180515)
    deltas = []
    for _ in range(40):
        picks = np.concatenate([rng.choice(idx, size=(100, len(idx))) for idx in groups], axis=1)
        errors = np.sqrt(squared[picks].mean(axis=1))
        deltas.extend(errors[:, 0] - errors[:, 1])
    return np.quantile(deltas, [.025, .975])


def pool(root, output):
    rows, comparisons, predictions = [], [], []
    for target, case_id in selected_specialists(root)["separate"].items():
        forecasts = archived_predictions(root, case_id, target)
        predictions.append(forecasts.assign(target=target))
        for split, group in forecasts.groupby("split"):
            rmse, r2 = rmse_r2(group.truth.to_numpy(), group[GROUPS + ["pooled_15"]].to_numpy())
            for model, error, fit in zip(GROUPS + ["pooled_15", "original_mean"],
                                         [*rmse, rmse[:3].mean()], [*r2, r2[:3].mean()], strict=True):
                rows.append(dict(split=split, target=target, model=model, n=len(group), rmse=error, r2=fit))
        # The fair baseline is one five-network group chosen on validation, not the mean of three.
        validation = forecasts.loc[forecasts.split.eq("validation")]
        winner = GROUPS[np.argmin(rmse_r2(validation.truth.to_numpy(), validation[GROUPS].to_numpy())[0])]
        test = forecasts.loc[forecasts.split.eq("test")]
        rmse, r2 = rmse_r2(test.truth.to_numpy(), test[[winner, "pooled_15"]].to_numpy())
        low, high = paired_interval(test, winner)
        comparisons.append(dict(target=target, selected_by_validation=winner, five_rmse=rmse[0],
                                pooled_rmse=rmse[1], delta_rmse=rmse[1] - rmse[0],
                                delta_rmse_pct=100 * (rmse[1] / rmse[0] - 1),
                                ci95_low=low, ci95_high=high, five_r2=r2[0], pooled_r2=r2[1]))
    summary = pd.DataFrame(rows)
    archived = read_csv(root / "results/system_summary.csv").query("system == 'separate'")
    check = archived.merge(summary.query("model == 'original_mean'"), on=["split", "target"], validate="one_to_one")
    if len(check) != 6:
        raise ValueError("Expected the search's specialist summaries for test and profile test")
    np.testing.assert_allclose(check[["rmse", "r2"]], check[["rmse_mean", "r2_mean"]], atol=1e-12, rtol=0)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "pooled_scores.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "pooled_vs_five.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "pooled_predictions.csv.gz", index=False,
                                                    compression={"method": "gzip", "mtime": 0})
    print(pd.DataFrame(comparisons).to_string(index=False))


# ---------------------------------------------------------------- calibrate
def load_models(root):
    systems = selected_specialists(root)
    joint_case, = set(systems["shared_reference"].values())
    hashes = {}

    def load(case, repeat):
        folder = root / "runs" / case / f"repeat_{repeat}"
        hashes[f"{case}/repeat_{repeat}"] = data.sha256(folder / "checkpoints.pt")
        return engine.load_ensemble(folder)

    joint = load(joint_case, 1)
    if joint.targets != data.TARGETS:
        raise ValueError("Unexpected target order in joint checkpoint")
    specialists = {}
    for index, target in enumerate(data.TARGETS):
        groups = [load(systems["separate"][target], repeat) for repeat in REPEATS]
        seeds = [seed for group in groups for seed in group.config["member_seeds"]]
        if len(set(seeds)) != 15 or any(group.targets != [target] for group in groups):
            raise ValueError(f"Expected 15 distinct trained specialist members for {target}")
        for group in groups:
            np.testing.assert_allclose(group.raw_training_std, joint.raw_training_std[index], rtol=1e-12)
        specialists[target] = groups
    return joint, specialists, hashes


def forecast(frame, split, joint, specialists):
    if not frame.sample_id.is_unique:
        raise ValueError(f"Duplicate case IDs in {split}")
    x = frame[["sample_id", *data.FEATURES]]
    all_joint = joint.predict(x)["members"]
    parts = []
    for index, target in enumerate(data.TARGETS):
        members = np.concatenate([group.predict(x)["members"][:, :, 0] for group in specialists[target]])
        for model, values in [("pooled_15", members), ("joint_5", all_joint[:, :, index])]:
            part = frame[["sample_id", "RMIN_LOC"]].rename(columns={"RMIN_LOC": "r_over_a"}).copy()
            part["split"], part["model"], part["target"] = split, model, target
            part["prediction"], part["spread"] = values.mean(axis=0), values.std(axis=0, ddof=1)
            # The floor is 1% of the target's original-unit training SD (checkpoint ddof=0).
            part["scale"] = np.maximum(part.spread, max(1e-8, 0.01 * joint.raw_training_std[index]))
            if target in frame:
                part["truth"] = frame[target]
            if "input_sha256" in frame:
                part["input_sha256"] = frame.input_sha256
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit_multipliers(calibration):
    rows = []
    for (model, target), cases in calibration.groupby(MODEL_KEYS):
        residuals = ((cases.truth - cases.prediction).abs() / cases.scale).to_numpy()
        n = residuals.size
        rank = math.ceil((n + 1) * COVERAGE)
        if rank > n or not np.isfinite(residuals).all():
            raise ValueError("Insufficient or nonfinite calibration scores")
        rows.append(dict(model=model, target=target, calibration_n=n, rank=rank,
                         multiplier=np.partition(residuals, rank - 1)[rank - 1]))
    return pd.DataFrame(rows)


def add_intervals(frame, multipliers):
    joined = frame.merge(multipliers[MODEL_KEYS + ["multiplier"]], on=MODEL_KEYS, how="left", validate="many_to_one")
    half_width = joined.scale * joined.multiplier
    if not np.isfinite(half_width).all():
        raise ValueError("Nonfinite interval widths")
    return joined.assign(lower=joined.prediction - half_width, upper=joined.prediction + half_width)


def fold_assignments(validation):
    from sklearn.model_selection import StratifiedKFold
    cases = validation[["sample_id", "r_over_a"]].drop_duplicates().sort_values("sample_id")
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=FOLD_SEED)
    assignments = np.zeros(len(cases), dtype=int)
    for fold, (_, held) in enumerate(splitter.split(cases, cases.r_over_a.astype(str)), start=1):
        assignments[held] = fold
    return pd.Series(assignments, index=cases.sample_id)


def verify_archived(scored, root, joint_case, output):
    """Forecast centers must equal the saved joint and pooled predictions on every old case."""
    old_joint = read_csv(root / "runs" / joint_case / "repeat_1/predictions.csv", dtype={"member_seed": str})
    old_joint = old_joint.loc[old_joint.member_seed.eq("ensemble")].assign(model="joint_5")
    old_pool = read_csv(output / "pooled_predictions.csv.gz").rename(columns={"pooled_15": "prediction"})
    keys = ["split", "model", "target", "sample_id"]
    old = pd.concat([part[keys + ["truth", "prediction"]] for part in (old_joint, old_pool.assign(model="pooled_15"))])
    old["split"] = old.split.replace({"validation": "validation_oof", "test": "historical_test"})
    joined = old.merge(scored, on=keys, suffixes=("_archived", ""), how="outer", indicator=True, validate="one_to_one")
    if not joined._merge.eq("both").all():
        raise ValueError("Archived evaluation case membership mismatch")
    np.testing.assert_allclose(joined.truth, joined.truth_archived, rtol=0, atol=1e-12)
    np.testing.assert_allclose(joined.prediction, joined.prediction_archived, rtol=1e-11, atol=1e-11)
    return float((joined.prediction - joined.prediction_archived).abs().max())


def varied_plan():
    """The 240 varied-case inputs, exactly as written before TGLF was run on them."""
    plan = read_csv(VARIED).loc[:, :"input_sha256"]
    text = plan.to_csv(index=False)
    return plan, hashlib.sha256(text.encode()).hexdigest()


def calibrate(root, output):
    import torch
    torch.set_num_threads(2)
    joint, specialists, hashes = load_models(root)
    validation = forecast(data.load_split("validation", root), "validation_oof", joint, specialists)
    all_q = fit_multipliers(validation)
    validation["fold"] = validation.sample_id.map(fold_assignments(validation))
    labeled, qs = [], [all_q.assign(fold=0)]
    for fold in range(1, FOLDS + 1):
        held = validation.fold.eq(fold)
        q = fit_multipliers(validation.loc[~held])
        qs.append(q.assign(fold=fold))
        labeled.append(add_intervals(validation.loc[held], q))
    for split, frame in [("historical_test", data.load_split("test", root)),
                         ("profile_test", data.load_profile_test(root))]:
        labeled.append(add_intervals(forecast(frame, split, joint, specialists), all_q).assign(fold=0))
    scored = pd.concat(labeled, ignore_index=True).sort_values(["split", "sample_id", *MODEL_KEYS])
    parity = verify_archived(scored, root, joint.config["case_id"], output)
    plan, plan_sha = varied_plan()
    plan = plan[["sample_id", *data.FEATURES, "input_sha256"]]
    if (len(plan) != 240 or not plan.input_sha256.is_unique
            or not np.array_equal(plan.groupby("RMIN_LOC").size().index, np.arange(225, 776, 50) / 1000)
            or not plan.groupby("RMIN_LOC").size().eq(20).all()):
        raise ValueError("Unexpected 12-radius varied input plan")
    varied = add_intervals(forecast(plan, "varied_test", joint, specialists), all_q).assign(fold=0)
    varied = varied.sort_values(["sample_id", *MODEL_KEYS])
    calibration = pd.concat(qs, ignore_index=True)
    floors = dict(zip(data.TARGETS, np.maximum(1e-8, 0.01 * joint.raw_training_std), strict=True))
    calibration["floor"] = calibration.target.map(floors)
    output.mkdir(parents=True, exist_ok=True)
    varied.to_csv(output / "varied_forecasts.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    scored.to_csv(output / "forecasts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    freeze = {"nominal_marginal_coverage": COVERAGE, "folds": FOLDS, "fold_seed": FOLD_SEED,
              "final_calibration_cases": int(validation.sample_id.nunique()), "varied_cases": len(plan),
              "varied_forecasts": len(varied), "mean_parity_max_abs": parity, "checkpoint_sha256": hashes,
              "input_plan_sha256": plan_sha,
              "varied_predictions_sha256": data.sha256(output / "varied_forecasts.csv"),
              "calibration_sha256": data.sha256(output / "calibration.csv")}
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print("Frozen varied forecasts:", freeze["varied_predictions_sha256"])


# ---------------------------------------------------------------- score
def truth_table(wide):
    return (wide[["sample_id", "RMIN_LOC", *data.TARGETS]].rename(columns={"RMIN_LOC": "r_over_a"})
            .melt(id_vars=["sample_id", "r_over_a"], value_vars=data.TARGETS, var_name="target", value_name="truth"))


def indexed(frame, keys, columns):
    values = frame.set_index(keys)[columns].sort_index()
    if not values.index.is_unique:
        raise ValueError(f"Duplicate case keys: {keys}")
    return values


def verify_forecasts(forecasts, truth):
    """Every model must cover exactly the expected cases, outputs, radii, and truths."""
    expected = indexed(truth, CASE_TARGET, ["r_over_a", "truth"])
    if set(forecasts.model) != {"pooled_15", "joint_5"} or not np.isfinite(forecasts[NUMERIC]).all().all():
        raise ValueError("Missing model or nonfinite forecast, interval or truth")
    if ((forecasts.spread < 0) | (forecasts.lower > forecasts.prediction) | (forecasts.upper < forecasts.prediction)).any():
        raise ValueError("Invalid spread or interval limits")
    for _, group in forecasts.groupby("model"):
        pd.testing.assert_frame_equal(indexed(group, CASE_TARGET, ["r_over_a", "truth"]), expected,
                                      check_dtype=False, check_exact=True)


def interval_score(group, train_sd):
    error = (group.prediction - group.truth).to_numpy()
    widths = (group.upper - group.lower).to_numpy()
    covered = group.truth.between(group.lower, group.upper, inclusive="both")
    rmse = float(np.sqrt(np.mean(error ** 2)))
    variance = group.truth.var(ddof=0)
    return {"n": len(group), "n_covered": int(covered.sum()), "coverage": covered.mean(),
            "mean_width": widths.mean(), "median_width": np.median(widths), "rmse": rmse,
            "mae": np.mean(np.abs(error)), "r2": 1 - rmse ** 2 / variance if variance > 0 else np.nan,
            "nrmse": rmse / train_sd, "train_sd": train_sd, "n_negative_truth": int(group.truth.lt(0).sum()),
            "n_negative_prediction": int(group.prediction.lt(0).sum())}


def summarize(forecasts, train_sd, levels):
    rows = []
    for keys, group in forecasts.groupby(levels, sort=True):
        fields = dict(zip(levels, keys, strict=True))
        rows.append({**fields, **interval_score(group, train_sd[fields["target"]])})
    return pd.DataFrame(rows)


def score_varied(output):
    freeze = json.loads((output / "freeze.json").read_text())
    forecasts = read_csv(output / "varied_forecasts.csv")
    if data.sha256(output / "varied_forecasts.csv") != freeze["varied_predictions_sha256"]:
        raise ValueError("Varied forecasts changed since the prediction freeze")
    plan, plan_sha = varied_plan()
    if plan_sha != freeze["input_plan_sha256"]:
        raise ValueError("The varied inputs differ from the ones the forecasts were frozen on")
    expected = indexed(plan.rename(columns={"RMIN_LOC": "r_over_a"}), "sample_id", ["r_over_a", "input_sha256"])
    if set(forecasts.split) != {"varied_test"} or "truth" in forecasts:
        raise ValueError("Expected label-free forecasts for the varied cohort")
    for _, group in forecasts.groupby(MODEL_KEYS):
        pd.testing.assert_frame_equal(indexed(group, "sample_id", ["r_over_a", "input_sha256"]), expected,
                                      check_dtype=False, check_exact=True)
    truth = truth_table(read_csv(VARIED))
    scored = forecasts.merge(truth, on=CASE_TARGET + ["r_over_a"], how="left", validate="many_to_one")
    verify_forecasts(scored, truth)
    return scored


def score(root, output):
    original = read_csv(output / "forecasts.csv.gz")
    sources = {"validation_oof": data.load_split("validation", root),
               "historical_test": data.load_split("test", root), "profile_test": data.load_profile_test()}
    if set(original.split) != set(sources):
        raise ValueError("Expected out-of-fold validation, historical test and profile forecasts")
    for split, frame in sources.items():
        verify_forecasts(original.loc[original.split.eq(split)], truth_table(frame))
    varied = score_varied(output)
    forecasts = pd.concat([original, varied], ignore_index=True)
    train_sd = data.load_split("train", root)[data.TARGETS].std(ddof=0)
    groups = ["split", "model", "target"]
    summarize(forecasts, train_sd, groups).to_csv(output / "interval_scores.csv", index=False)
    summarize(forecasts, train_sd, groups + ["r_over_a"]).to_csv(output / "interval_scores_by_radius.csv", index=False)
    varied.to_csv(output / "varied_scored.csv", index=False)
    table = read_csv(output / "interval_scores.csv")
    print(table.loc[table.split.eq("varied_test"), ["model", "target", "n", "coverage", "mean_width", "rmse"]]
          .to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pool", "calibrate", "score"])
    parser.add_argument("--root", type=Path, default=RUNS / "search", help="completed search directory")
    parser.add_argument("--output", type=Path, default=RUNS / "pooled_intervals")
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    if output.is_relative_to(CODE):
        parser.error("Choose an --output under notebooks/ryan/runs/, not the source folder.")
    {"pool": lambda: pool(root, output), "calibrate": lambda: calibrate(root, output),
     "score": lambda: score(root, output)}[args.command]()


if __name__ == "__main__":
    main()
