"""Task 3: the fixed 15-network ensemble, calibration, and TGLF data utilities."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

# Allow direct invocation from the repository root or another working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2 import nn_ensemble
from task2.scan import make_features
from tglf_data import load_cases

TARGETS = ["Q_i", "Q_e", "Gamma_e"]
ENSEMBLE_SIZE = 15
ENSEMBLE_SEEDS = tuple(range(42, 57))
OOD_SETS = ["ood_RLTS_1", "ood_RLTS_2", "ood_RLN", "ood_TAUS_2", "ood_XNUE",
            "ood_Q_LOC", "ood_KAPPA_LOC", "ood_DRMAJDX_LOC", "ood_RMIN_LOC"]
INPUT_RANGES = {
    "RLTS_1": (1.5, 3.5), "RLTS_2": (1.5, 3.5), "RLNS_1": (1, 3),
    "TAUS_2": (0.5, 1.5), "XNUE": (0.01, 1), "Q_LOC": (1.5, 2.5),
    "KAPPA_LOC": (1, 2), "DRMAJDX_LOC": (-0.5, 0), "RMIN_LOC": (0.25, 0.75),
}
INPUT_FRACTIONS = {"RLTS_1": .10, "RLTS_2": .10, "RLNS_1": .10,
                   "TAUS_2": .10, "XNUE": .20, "Q_LOC": .05, "Q_PRIME_LOC": .10}


def save_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


class Workflow:
    def __init__(self, task2_dir, task3_dir, threads=4):
        self.task2_dir, self.task3_dir = Path(task2_dir).resolve(), Path(task3_dir).resolve()
        if threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(threads)
        self.task2 = nn_ensemble
        self.training = nn_ensemble.load_data(self.task2_dir / "prepared")
        self.manifest = nn_ensemble.check_manifest(self.task2_dir / "models", self.training)
        self.constant_inputs = json.loads(
            (self.task2_dir / "prepared/preparation_metadata.json").read_text()
        )["constant_inputs"]
        self.features = pd.read_csv(self.task2_dir / "prepared/features.csv", index_col="case",
                                    float_precision="round_trip")
        assigned = pd.read_csv(self.task2_dir / "prepared/split_assignments.csv", index_col="case")["split"]
        self.split = assigned.loc[self.features.index]
        self.ids = {name: self.features.index[self.split.eq(name)]
                    for name in ("train", "validation", "test")}
        self.columns = self.training["feature_columns"]
        self.training_scale = self.training["raw_training_std"]
        self.base_members = nn_ensemble.load_ensemble(self.task2_dir / "models")
        for (_, saved), entry in zip(self.base_members, self.manifest["members"], strict=True):
            saved["task3_checkpoint_sha256"] = entry["checksums_sha256"]["checkpoint.pt"]
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            np.testing.assert_allclose(self.training[key], self.base_members[0][1][key],
                                       rtol=1e-12, atol=1e-12)
            # Use the exact saved Task 2 scaler values for new members, too.
            self.training[key] = self.base_members[0][1][key].copy()
        self.extra_dir = self.task3_dir / "models"

    def truth(self, split):
        if split not in self.ids:
            raise ValueError(f"Unknown split: {split}")
        return pd.read_csv(self.task2_dir / "prepared/targets_raw.csv", index_col="case",
                           float_precision="round_trip").loc[self.ids[split], TARGETS]

    def _config(self, seed):
        return self.task2.config_from_saved(dict(self.manifest["fixed_nn_config"], seed=seed))

    def members(self, train_missing=True):
        """Extend the five Task 2 models with seeds 47–56. Reuse completed fits on reruns."""
        members = list(self.base_members)
        self.extra_dir.mkdir(parents=True, exist_ok=True)
        for seed in ENSEMBLE_SEEDS[len(self.base_members):]:
            config = self._config(seed)
            checkpoint = self.extra_dir / f"nn_seed{seed}.pt"
            if not checkpoint.exists():
                if not train_missing:
                    raise FileNotFoundError(f"Missing Task 3 network for seed {seed}")
                print(f"Training seed {seed} on CPU...", flush=True)
                # Publish only a completed checkpoint. An interrupted fit can be retried.
                with tempfile.TemporaryDirectory(dir=self.extra_dir) as tmp:
                    self.task2.fit_trial(config, self.training, tmp, device="cpu")
                    (Path(tmp) / "checkpoint.pt").replace(checkpoint)
            model, saved = self.task2.load_checkpoint(checkpoint)
            if (saved["config"] != asdict(config) or saved["source_sha256"] != self.training["hashes"]
                    or saved["source_code_sha256"] != self.task2.digest(self.task2.__file__)):
                raise ValueError(f"Seed {seed} was trained on different inputs, settings or code; use a fresh output folder.")
            if saved["feature_columns"] != self.columns:
                raise ValueError("Feature order changed.")
            for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
                np.testing.assert_array_equal(saved[key], self.base_members[0][1][key])
            saved["task3_checkpoint_sha256"] = self.task2.digest(checkpoint)
            members.append((model, saved))
        if tuple(saved["config"]["seed"] for _, saved in members) != ENSEMBLE_SEEDS:
            raise ValueError("Task 3 requires exactly seeds 42 through 56.")
        return members

    def predict(self, frame, members):
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.index.is_unique:
            raise ValueError("Supply a nonempty DataFrame with unique case IDs.")
        if tuple(saved["config"]["seed"] for _, saved in members) != ENSEMBLE_SEEDS:
            raise ValueError("Task 3 predictions require all 15 members, seeds 42 through 56.")
        raw = frame.loc[:, self.columns].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite features.")
        reference = members[0][1]
        x = ((raw - reference["input_mean"]) / reference["input_scale"]).astype(np.float32)
        predictions = np.stack([self.task2.predict(model, x, saved)["prediction"] for model, saved in members])
        return summarize_members(predictions, frame.index, members)

    def load_external(self, root, expected_count=None):
        """Read a course dataset with the same feature and species conventions as Task 2."""
        table = load_cases(root)
        if expected_count is not None and len(table) != expected_count:
            raise ValueError(f"Expected {expected_count} cases in {root}, found {len(table)}")
        np.testing.assert_allclose(table.RLNS_1, table.RLNS_2, rtol=0, atol=1e-10)
        for column, value in self.constant_inputs.items():
            np.testing.assert_allclose(table[column], value, rtol=1e-8, atol=1e-10)
        return table, engineer_features(table, self.columns), table[TARGETS]


def summarize_members(values, index, members):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(members), len(index), 3) or not np.isfinite(values).all():
        raise ValueError("Invalid member prediction array.")
    identity = [{"seed": saved["config"]["seed"], "sha256": saved["task3_checkpoint_sha256"]}
                for _, saved in members]
    model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"mean": values.mean(axis=0), "std": values.std(axis=0, ddof=1),
            "members": values, "index": pd.Index(index).copy(), "model_id": model_id,
            "member_identity": identity}


def aligned_truth(prediction, truth):
    if not truth.index.equals(prediction["index"]) or truth.columns.tolist() != TARGETS:
        raise ValueError("Truth rows and target order must exactly match predictions.")
    values = truth.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite truth values.")
    return values


def calibrate(prediction, truth, training_scale, alpha=.1, floor_fraction=.01):
    """Freeze per-target empirical normalized-residual and fixed-width intervals."""
    y = aligned_truth(prediction, truth)
    if not 0 < alpha < 1 or floor_fraction <= 0:
        raise ValueError("Use 0 < alpha < 1 and a positive floor fraction.")
    floor = floor_fraction * np.asarray(training_scale, dtype=float)
    if floor.shape != (3,) or not np.isfinite(floor).all() or not (floor > 0).all():
        raise ValueError("Training scales must be three finite positive values.")
    n = len(y)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError("Too few calibration cases for this coverage level.")
    residual = np.abs(y - prediction["mean"])
    scale = np.maximum(prediction["std"], floor)
    return {"targets": TARGETS, "alpha": alpha, "n_calibration": n,
            "quantile_rank_1_based": rank, "floor_fraction": floor_fraction,
            "floor": floor.tolist(),
            "scaled_multiplier": np.sort(residual / scale, axis=0)[rank - 1].tolist(),
            "fixed_half_width": np.sort(residual, axis=0)[rank - 1].tolist(),
            "model_id": prediction["model_id"], "members": prediction["member_identity"],
            "calibration_cases": prediction["index"].tolist(),
            "interpretation": "Empirical calibration on reused Task 2 validation data; no formal coverage guarantee."}


def bounds(prediction, calibration, method="scaled"):
    if calibration["model_id"] != prediction["model_id"]:
        raise ValueError("Calibration belongs to different model weights or ensemble size. Recalibrate.")
    if method == "scaled":
        half_width = np.maximum(prediction["std"], calibration["floor"]) * calibration["scaled_multiplier"]
    elif method == "fixed":
        half_width = np.broadcast_to(calibration["fixed_half_width"], prediction["mean"].shape)
    else:
        raise ValueError("Choose scaled or fixed intervals.")
    return prediction["mean"] - half_width, prediction["mean"] + half_width


def metrics(prediction, truth, calibration, dataset, method="scaled"):
    y = aligned_truth(prediction, truth)
    lo, hi = bounds(prediction, calibration, method)
    error = prediction["mean"] - y
    rows = []
    for j, target in enumerate(TARGETS):
        absolute = np.abs(error[:, j])
        spread = prediction["std"][:, j]
        rank_corr = (float(spearmanr(spread, absolute).statistic)
                     if np.ptp(spread) > 0 and np.ptp(absolute) > 0 else np.nan)
        interval_score = hi[:, j] - lo[:, j] + 2 / calibration["alpha"] * (
            np.maximum(lo[:, j] - y[:, j], 0) + np.maximum(y[:, j] - hi[:, j], 0))
        rows.append({"dataset": dataset, "method": method, "target": target, "n": len(y),
                     "members": len(prediction["members"]),
                     "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(error[:, j] ** 2))),
                     "bias": float(error[:, j].mean()),
                     "coverage": float(np.mean((y[:, j] >= lo[:, j]) & (y[:, j] <= hi[:, j]))),
                     "mean_width": float(np.mean(hi[:, j] - lo[:, j])),
                     "mean_std": float(spread.mean()), "error_uncertainty_spearman": rank_corr,
                     "interval_score": float(interval_score.mean())})
    return pd.DataFrame(rows)


def prediction_frame(prediction, calibration=None, truth=None):
    frame = pd.DataFrame(index=prediction["index"])
    if truth is not None:
        y = aligned_truth(prediction, truth)
    if calibration is not None:
        lo, hi = bounds(prediction, calibration)
        level = f"{100 * (1 - calibration['alpha']):g}"
    for j, target in enumerate(TARGETS):
        frame[f"{target}_prediction"] = prediction["mean"][:, j]
        frame[f"{target}_std"] = prediction["std"][:, j]
        if calibration is not None:
            frame[f"{target}_lower{level}"] = lo[:, j]
            frame[f"{target}_upper{level}"] = hi[:, j]
        if truth is not None:
            frame[f"{target}_true"] = y[:, j]
            frame[f"{target}_absolute_error"] = np.abs(y[:, j] - prediction["mean"][:, j])
    return frame


def region_summary(features, prediction, truth, calibration):
    y = aligned_truth(prediction, truth)
    if not features.index.equals(prediction["index"]):
        raise ValueError("Feature rows must align with predictions.")
    lo, hi = bounds(prediction, calibration)
    rows = []
    for feature in features:
        bins = pd.qcut(features[feature], q=4, duplicates="drop")
        for interval in bins.cat.categories:
            mask = bins.eq(interval).to_numpy()
            if not mask.any():
                continue
            for j, target in enumerate(TARGETS):
                rows.append({"feature": feature, "range": f"({interval.left:.3f}, {interval.right:.3f}]", "target": target,
                             "n": int(mask.sum()), "mean_std": float(prediction["std"][mask, j].mean()),
                             "mae": float(np.abs(prediction["mean"][mask, j] - y[mask, j]).mean()),
                             "coverage": float(((y[mask, j] >= lo[mask, j]) & (y[mask, j] <= hi[mask, j])).mean())})
    return pd.DataFrame(rows)


def engineer_features(table, columns):
    return make_features(table).loc[:, columns]


def range_flags(raw):
    flags = pd.DataFrame({column: ~raw[column].between(lo, hi)
                          for column, (lo, hi) in INPUT_RANGES.items()}, index=raw.index)
    flags["MASS_2"] = ~raw.MASS_2.isin([.5, 1., 1.5])
    # Q_PRIME_LOC has no independent numeric range in the handout.
    flags["any_outside"] = flags.any(axis=1)
    return flags


def sample_inputs(reference, n=10000, seed=2026, varying=None):
    """Uniform physical-input draws; paired density gradients stay equal.

    Generate the same underlying random draws for joint and one-input experiments.
    """
    if n < 2:
        raise ValueError("Use at least two samples.")
    chosen = set(INPUT_FRACTIONS if varying is None else varying)
    if not chosen.issubset(INPUT_FRACTIONS):
        raise ValueError("Unknown uncertain input.")
    values = pd.DataFrame([reference.to_dict()] * n,
                          index=pd.Index([f"mc_{i:05d}" for i in range(n)], name="case"))
    draws = np.random.default_rng(seed).uniform(-1, 1, size=(n, len(INPUT_FRACTIONS)))
    for j, (column, fraction) in enumerate(INPUT_FRACTIONS.items()):
        if column in chosen:
            values[column] = float(reference[column]) * (1 + fraction * draws[:, j])
    values["RLNS_2"] = values.RLNS_1
    return values


def distribution_summary(values, label):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame([{"source": label, "target": target, "n": len(values),
                          "mean": values[:, j].mean(), "std": values[:, j].std(ddof=1),
                          "q05": np.quantile(values[:, j], .05),
                          "median": np.quantile(values[:, j], .5),
                          "q95": np.quantile(values[:, j], .95)}
                         for j, target in enumerate(TARGETS)])


def reproduce(task2, data, output, threads):
    """Regenerate Task 3 tables and figures from a Task 2 run and raw TGLF cases."""
    from task3_plots import (calibration_figure, test_figure, ood_figure,
                          forward_figure, sensitivity_figure)
    import matplotlib.pyplot as plt

    workflow = Workflow(task2, output, threads)
    outside = {group: workflow.load_external(data / 'scan_db_outside' / group / 'data', 30)
               for group in OOD_SETS}
    reference, _, _ = workflow.load_external(data / 'data_d3d/d3d_db', 1)
    _, paired_x, paired_y = workflow.load_external(data / 'data_d3d/d3d_uq/data', 300)
    members = workflow.members()

    # Choose interval widths using validation cases, then evaluate on the test split.
    validation_y = workflow.truth('validation')
    validation = workflow.predict(workflow.features.loc[validation_y.index], members)
    calibration = calibrate(validation, validation_y, workflow.training_scale)
    test_y = workflow.truth('test')
    test_x = workflow.features.loc[test_y.index]
    test = workflow.predict(test_x, members)
    test_scores = metrics(test, test_y, calibration, 'test').set_index('target')
    outside_scores = []
    for group, (_, x, y) in outside.items():
        outside_scores.append(metrics(workflow.predict(x, members), y, calibration, group))
    ood = pd.concat(outside_scores, ignore_index=True)
    ood['RMSE ratio'] = ood.rmse / ood.target.map(test_scores.rmse.replace(0, np.nan))
    ood['SD ratio'] = ood.mean_std / ood.target.map(test_scores.mean_std.replace(0, np.nan))
    ood['Coverage (%)'] = 100 * ood.coverage

    # The 300 matched TGLF cases and 10,000 simulated input draws answer different questions.
    paired = workflow.predict(paired_x, members)
    columns = list(INPUT_FRACTIONS) + ['RLNS_2', 'KAPPA_LOC', 'DRMAJDX_LOC', 'RMIN_LOC', 'MASS_2']
    profile = reference.iloc[0][columns].astype(float)
    joint = workflow.predict(engineer_features(sample_inputs(profile), workflow.columns), members)
    distributions = pd.concat([
        distribution_summary(joint['mean'], 'surrogate_10000_input_draws'),
        distribution_summary(paired['mean'], 'surrogate_300_paired_cases'),
        distribution_summary(paired_y.to_numpy(), 'tglf_300_paired_cases'),
    ], ignore_index=True)
    sensitivity = []
    for name in INPUT_FRACTIONS:
        draws = sample_inputs(profile, varying=[name])
        response = workflow.predict(engineer_features(draws, workflow.columns), members)
        sensitivity.append(distribution_summary(response['mean'], name))
    sensitivity = pd.concat(sensitivity, ignore_index=True).rename(columns={'source': 'input'})

    results = output / 'results'
    results.mkdir(parents=True, exist_ok=True)
    save_json(results / 'calibration.json', calibration)
    test_scores.to_csv(results / 'test_metrics.csv')
    prediction_frame(test, calibration, test_y).to_csv(results / 'test_predictions.csv')
    region_summary(test_x, test, test_y, calibration).to_csv(results / 'test_regions.csv', index=False)
    ood.to_csv(results / 'outside_metrics.csv', index=False)
    prediction_frame(paired, calibration, paired_y).to_csv(results / 'd3d_paired_predictions.csv')
    metrics(paired, paired_y, calibration, 'data_d3d').to_csv(results / 'd3d_paired_metrics.csv', index=False)
    distributions.to_csv(results / 'd3d_distributions.csv', index=False)
    sensitivity.to_csv(results / 'd3d_input_sensitivity.csv', index=False)

    figures = output / 'figures'
    for figure, _ in [calibration_figure(validation, validation_y, calibration, figures),
                      test_figure(test, test_y, calibration, test_scores, figures),
                      ood_figure(ood, figures), forward_figure(joint, paired, paired_y, figures),
                      sensitivity_figure(sensitivity, figures)]:
        plt.close(figure)
    print(test_scores[['rmse', 'coverage', 'mean_width']].to_string())
    print(f'Task 3 tables and figures written under {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task2', type=Path, required=True, help='Generated Task 2 directory with prepared/ and models/')
    parser.add_argument('--data', type=Path, required=True, help='Directory containing scan_db_outside/ and data_d3d/')
    parser.add_argument('--output', type=Path, required=True, help='Generated Task 3 model, result, and figure directory')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    reproduce(args.task2, args.data, args.output, args.threads)
