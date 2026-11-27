"""Summarize the frozen Task 5 search results; no model is fitted or selected here.

Reads ``results/`` of a completed search (``search.py all``) and writes four tables there:

* ``system_summary.csv``: mean and SD over three repeats of RMSE, R², bias and negative-heat
  rate for the six final systems on the test set and the 12 profile cases;
* ``paired_bootstrap.csv``: 95% radius-stratified case-bootstrap intervals for paired RMSE
  differences between systems on the test set;
* ``flux_regime_metrics.csv``: errors in the bottom 50%, middle 49%, and top 1% of true flux;
* ``system_cost_summary.csv``: networks, parameters, and inference latency per system.

From the repository root: ``python notebooks/ryan/task5/report.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2] / "notebooks/ryan/runs/task5/search"
TARGETS = ["Q_e", "Q_i", "Gamma_e"]
SYSTEMS = ["shared", "output_specific_shared", "separate", "shared_reference",
           "separate_reference", "capacity_control"]


def _load(root):
    results = root / "results"
    selection = json.loads((results / "selection.json").read_text())
    final = selection.get("final", {})
    if not final.get("evaluation_completed_at"):
        raise RuntimeError("Final evaluation has not completed; reporting cannot open final outputs yet.")
    if final.get("selection_split") != "validation":
        raise ValueError("Expected final model choices frozen using validation scores")
    if set(final["systems"]) != set(SYSTEMS):
        raise ValueError("Final output must contain the three planned systems and three controls")
    splits = final["evaluated_datasets"]
    if not {"test", "profile_test"} <= set(splits):
        raise ValueError("Both fixed-test and supplied profile evaluations are required")
    metrics = pd.read_csv(results / "system_metrics.csv")
    predictions = pd.read_csv(results / "system_predictions.csv", dtype={"sample_id": str})
    costs = pd.read_csv(results / "system_costs.csv")
    required_metrics = {"system", "repeat", "split", "target", "case_id", "rmse", "nrmse",
                        "mae", "r2", "bias", "negative_heat_rate", "n_cases"}
    required_predictions = {"system", "repeat", "split", "target", "case_id", "sample_id",
                            "r_over_a", "truth", "prediction"}
    required_costs = {"system", "repeat", "split", "parameters_total", "networks", "unique_ensembles",
                      "inference_single_seconds", "inference_batch_seconds", "inference_batch_cases"}
    for name, table, required in [("metrics", metrics, required_metrics),
                                  ("predictions", predictions, required_predictions),
                                  ("costs", costs, required_costs)]:
        if not required <= set(table):
            raise ValueError(f"Missing {name} columns: {sorted(required - set(table))}")
        if set(table.split) != set(splits) or set(table.system) != set(SYSTEMS):
            raise ValueError(f"Unexpected systems or datasets in {name}")
    keys = ["system", "repeat", "split", "target"]
    if metrics.duplicated(keys).any() or predictions.duplicated(keys + ["sample_id"]).any():
        raise ValueError("Duplicate final metrics or predictions")
    if costs.duplicated(["system", "repeat", "split"]).any():
        raise ValueError("Duplicate cost measurements")
    if not np.isfinite(metrics[["rmse", "nrmse", "mae", "bias"]].to_numpy(float)).all():
        raise ValueError("Nonfinite final error metrics; investigate before reporting")
    if not np.isfinite(predictions[["truth", "prediction", "r_over_a"]].to_numpy(float)).all():
        raise ValueError("Nonfinite final predictions; investigate affected configurations")
    if not np.isfinite(costs[["parameters_total", "inference_single_seconds", "inference_batch_seconds"]].to_numpy(float)).all():
        raise ValueError("Nonfinite final timing or parameter measurements")
    for split, count in splits.items():
        reference = None
        for target in TARGETS:
            truth_reference = None
            for system in SYSTEMS:
                case_id = final["systems"][system][target]
                observed = metrics[(metrics.split == split) & (metrics.target == target) & (metrics.system == system)]
                if len(observed) != 3 or set(observed["repeat"]) != {1, 2, 3} or not observed.n_cases.eq(count).all():
                    raise ValueError(f"Incomplete metrics for {split}/{target}/{system}")
                if not observed.case_id.eq(case_id).all():
                    raise ValueError("Final scores disagree with the frozen model mapping")
                for repeat in (1, 2, 3):
                    block = predictions[(predictions.split == split) & (predictions.target == target)
                                        & (predictions.system == system) & (predictions["repeat"] == repeat)]
                    block = block.sort_values("sample_id")
                    if len(block) != count or not block.case_id.eq(case_id).all():
                        raise ValueError(f"Incomplete predictions for {split}/{target}/{system}/{repeat}")
                    comparable = block[["sample_id", "r_over_a", "truth"]].reset_index(drop=True)
                    if truth_reference is None:
                        truth_reference = comparable
                    elif not comparable.equals(truth_reference):
                        raise ValueError("Systems/repeats do not use the same paired cases and truths")
                    identities = comparable[["sample_id", "r_over_a"]]
                    if reference is None:
                        reference = identities
                    elif not identities.equals(reference):
                        raise ValueError("Outputs do not share the same evaluation cases")
                    measured = float(np.sqrt(np.mean((block.prediction - block.truth) ** 2)))
                    saved = observed.loc[observed["repeat"].eq(repeat), "rmse"].iloc[0]
                    if not np.isclose(measured, saved, rtol=1e-9, atol=1e-12):
                        raise ValueError("Saved RMSE does not agree with saved predictions")
        for system in SYSTEMS:
            block = costs[(costs.split == split) & (costs.system == system)]
            if len(block) != 3 or set(block["repeat"]) != {1, 2, 3}:
                raise ValueError(f"Missing cost repeats for {split}/{system}")
            for name in ["parameters_total", "networks", "unique_ensembles", "inference_batch_cases"]:
                if block[name].nunique() != 1:
                    raise ValueError(f"Inconsistent deployment {name} across repeats")
    return selection, metrics, predictions, costs


def aggregate_metrics(metrics):
    grouped = metrics.groupby(["split", "system", "target"], sort=False)
    return grouped.agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                       nrmse_mean=("nrmse", "mean"), mae_mean=("mae", "mean"),
                       r2_mean=("r2", "mean"), r2_sd=("r2", "std"), bias_mean=("bias", "mean"),
                       negative_heat_rate_mean=("negative_heat_rate", "mean"),
                       n_cases=("n_cases", "first"), repeats=("repeat", "nunique")).reset_index()


def bootstrap_differences(predictions, draws=2000, seed=180513):
    """Case uncertainty for the mean paired RMSE gap, holding the 3 fits fixed.

    Resample the same cases for both systems and all repeats, preserving each
    original radius's case count. Do not bootstrap the one-case-per-radius profile.
    """
    rows = []
    contrasts = [("separate", "shared"), ("output_specific_shared", "shared"),
                 ("separate", "capacity_control")]
    test = predictions[predictions.split.eq("test")]
    for target_index, target in enumerate(TARGETS):
        block = test[test.target.eq(target)]
        for candidate, reference in contrasts:
            a = block[block.system.eq(candidate)].sort_values(["repeat", "sample_id"])
            b = block[block.system.eq(reference)].sort_values(["repeat", "sample_id"])
            n_cases = a.sample_id.nunique()
            loss_a = ((a.prediction - a.truth).to_numpy() ** 2).reshape(3, n_cases)
            loss_b = ((b.prediction - b.truth).to_numpy() ** 2).reshape(3, n_cases)
            radii = a.loc[a["repeat"].eq(1), "r_over_a"].to_numpy()
            strata = [np.flatnonzero(radii == radius) for radius in np.unique(radii)]
            rng = np.random.default_rng(seed + target_index)
            differences = []
            for offset in range(0, draws, 100):
                size = min(100, draws - offset)
                indices = np.concatenate([indices[rng.integers(0, len(indices), size=(size, len(indices)))]
                                          for indices in strata], axis=1)
                differences.extend((np.sqrt(loss_a[:, indices].mean(axis=2))
                                    - np.sqrt(loss_b[:, indices].mean(axis=2))).mean(axis=0))
            point = (np.sqrt(loss_a.mean(axis=1)) - np.sqrt(loss_b.mean(axis=1))).mean()
            low, high = np.quantile(differences, [.025, .975])
            rows.append({"split": "test", "target": target, "candidate": candidate,
                         "reference": reference, "mean_rmse_difference": point,
                         "ci_95_low": low, "ci_95_high": high, "bootstrap_draws": draws,
                         "bootstrap_seed": seed + target_index, "n_cases": n_cases,
                         "strata": len(strata), "estimand": "mean paired RMSE difference across three fixed repeats"})
    return pd.DataFrame(rows)


def flux_regimes(predictions):
    """Post-selection descriptive bins based on each dataset's signed true flux."""
    records = []
    for (split, target), block in predictions.groupby(["split", "target"], sort=False):
        reference = block[(block.system == "shared") & (block["repeat"] == 1)].sort_values("sample_id")
        q50, q99 = np.quantile(reference.truth, [.5, .99])
        for (system, repeat), subset in block.groupby(["system", "repeat"], sort=False):
            y = subset.truth.to_numpy()
            error = subset.prediction.to_numpy() - y
            total_sse = float(np.sum(error ** 2))
            masks = {"bottom_50": y <= q50, "middle_49": (y > q50) & (y <= q99), "top_1": y > q99}
            for label, mask in masks.items():
                count = int(mask.sum())
                records.append({"split": split, "target": target, "system": system, "repeat": repeat,
                                "regime": label, "q50_true_flux": q50, "q99_true_flux": q99,
                                "n_cases": count, "rmse": float(np.sqrt(np.mean(error[mask] ** 2))) if count else np.nan,
                                "mae": float(np.mean(np.abs(error[mask]))) if count else np.nan,
                                "sse_share": float(np.sum(error[mask] ** 2) / total_sse) if total_sse else np.nan,
                                "negative_heat_rate": float(np.mean(subset.prediction.to_numpy()[mask] < 0))
                                if count and target in TARGETS[:2] else np.nan})
    return pd.DataFrame(records)


def cost_summary(costs):
    test = costs[costs.split.eq("test")]
    return test.groupby("system", sort=False).agg(
        networks=("networks", "first"), parameters_total=("parameters_total", "first"),
        inference_single_ms=("inference_single_seconds", lambda s: 1000 * s.mean()),
        inference_batch_ms=("inference_batch_seconds", lambda s: 1000 * s.mean()),
        inference_batch_cases=("inference_batch_cases", "first")).reset_index()


def build_report(root=None, bootstrap_draws=2000, bootstrap_seed=180513):
    root = Path(root or ROOT).resolve()
    if root.is_relative_to(CODE):
        raise PermissionError("Pass a search --root under notebooks/ryan/runs/, not the source folder.")
    if bootstrap_draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    selection, metrics, predictions, costs = _load(root)
    tables = {"system_summary.csv": aggregate_metrics(metrics),
              "paired_bootstrap.csv": bootstrap_differences(predictions, bootstrap_draws, bootstrap_seed),
              "flux_regime_metrics.csv": flux_regimes(predictions),
              "system_cost_summary.csv": cost_summary(costs)}
    for name, table in tables.items():
        table.to_csv(root / "results" / name, index=False)
    return {"root": str(root), "tables": sorted(tables), "bootstrap_draws": bootstrap_draws}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="completed search directory")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=180513)
    args = parser.parse_args(argv)
    print(json.dumps(build_report(args.root, args.bootstrap_draws, args.bootstrap_seed), indent=2))


if __name__ == "__main__":
    main()
