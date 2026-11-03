"""A3: the fixed 15-network ensemble, calibration, and TGLF data utilities."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

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
    def __init__(self, a2_dir, a3_dir, threads=4):
        self.a2_dir, self.a3_dir = Path(a2_dir).resolve(), Path(a3_dir).resolve()
        sys.path.insert(0, str(self.a2_dir))
        import nn_ensemble as a2
        if Path(a2.__file__).resolve() != self.a2_dir / "nn_ensemble.py":
            raise RuntimeError("Restart the kernel: a different A2 module is already loaded.")
        self.a2 = a2
        torch.set_num_threads(threads)
        self.manifest = a2.check_manifest(self.a2_dir / "models")
        self.training = a2.load_data(self.a2_dir / "prepared")
        a2.check_manifest(self.a2_dir / "models", self.training)
        self.features = pd.read_csv(self.a2_dir / "prepared/features.csv", index_col="case",
                                    float_precision="round_trip")
        assigned = pd.read_csv(self.a2_dir / "prepared/split_assignments.csv", index_col="case")["split"]
        self.split = assigned.loc[self.features.index]
        self.ids = {name: self.features.index[self.split.eq(name)]
                    for name in ("train", "validation", "test")}
        self.columns = self.training["feature_columns"]
        self.training_scale = self.training["raw_training_std"]
        self.base_members = a2.load_ensemble(self.a2_dir / "models")
        for (_, saved), entry in zip(self.base_members, self.manifest["members"], strict=True):
            saved["a3_checkpoint_sha256"] = entry["checksums_sha256"]["checkpoint.pt"]
        for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
            np.testing.assert_allclose(self.training[key], self.base_members[0][1][key],
                                       rtol=1e-12, atol=1e-12)
            # Use the exact saved A2 scaler values for new members, too.
            self.training[key] = self.base_members[0][1][key].copy()
        self.extra_dir = self.a3_dir / "models"

    def truth(self, split):
        if split not in self.ids:
            raise ValueError(f"Unknown split: {split}")
        return pd.read_csv(self.a2_dir / "prepared/targets_raw.csv", index_col="case",
                           float_precision="round_trip").loc[self.ids[split], TARGETS]

    def _config(self, seed):
        return self.a2.config_from_saved(dict(self.manifest["fixed_nn_config"], seed=seed))

    def members(self, train_missing=True):
        """Load exactly 15 networks, training missing members when requested.

        Finished members are reused. A failed training attempt remains in a separate
        temporary directory and is never mistaken for a completed checkpoint.
        """
        members = list(self.base_members)
        for seed in ENSEMBLE_SEEDS[5:]:
            directory = self.extra_dir / f"nn_seed{seed}"
            if not directory.exists():
                if not train_missing:
                    raise FileNotFoundError(f"Missing member {seed}. Train the full 15-network ensemble first.")
                self.extra_dir.mkdir(parents=True, exist_ok=True)
                pending = Path(tempfile.mkdtemp(prefix=f".seed{seed}-", dir=self.extra_dir))
                print(f"Training member {seed - 41}/{ENSEMBLE_SIZE} (seed {seed}) on CPU...", flush=True)
                self.a2.fit_trial(self._config(seed), self.training, pending, device="cpu")
                save_json(pending / "completed.json", {
                    "seed": seed, "config": asdict(self._config(seed)),
                    "source_sha256": self.training["hashes"],
                    "checkpoint_sha256": self.a2.digest(pending / "checkpoint.pt"),
                    "training_code_sha256": self.a2.digest(self.a2_dir / "nn_ensemble.py"),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "platform": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "numpy": np.__version__,
                    "threads": torch.get_num_threads(), "device": "cpu",
                })
                pending.rename(directory)
            marker = directory / "completed.json"
            if not marker.is_file():
                raise ValueError(f"Incomplete member: {directory}. Move it aside before rerunning.")
            record = json.loads(marker.read_text())
            if record["source_sha256"] != self.training["hashes"] or record["config"] != asdict(self._config(seed)):
                raise ValueError(f"Member {seed} uses different data or settings.")
            checksum = self.a2.digest(directory / "checkpoint.pt")
            if checksum != record["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed for seed {seed}.")
            model, saved = self.a2.load_checkpoint(directory / "checkpoint.pt")
            if saved["source_sha256"] != record["source_sha256"] or saved["config"] != record["config"]:
                raise ValueError(f"Checkpoint metadata mismatch for seed {seed}.")
            if saved["source_code_sha256"] != record["training_code_sha256"]:
                raise ValueError(f"Training-code provenance mismatch for seed {seed}.")
            if saved["feature_columns"] != self.columns:
                raise ValueError("Feature order changed.")
            for key in ("input_mean", "input_scale", "target_mean", "target_scale"):
                np.testing.assert_array_equal(saved[key], self.base_members[0][1][key])
            saved["a3_checkpoint_sha256"] = checksum
            members.append((model, saved))
        if tuple(saved["config"]["seed"] for _, saved in members) != ENSEMBLE_SEEDS:
            raise ValueError("A3 requires exactly seeds 42 through 56.")
        return members

    def predict(self, frame, members):
        if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.index.is_unique:
            raise ValueError("Supply a nonempty DataFrame with unique case IDs.")
        if tuple(saved["config"]["seed"] for _, saved in members) != ENSEMBLE_SEEDS:
            raise ValueError("A3 predictions require all 15 members, seeds 42 through 56.")
        raw = frame.loc[:, self.columns].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite features.")
        reference = members[0][1]
        x = ((raw - reference["input_mean"]) / reference["input_scale"]).astype(np.float32)
        predictions = np.stack([self.a2.predict(model, x, saved)["prediction"] for model, saved in members])
        return summarize_members(predictions, frame.index, members)

    def load_external(self, root, expected_count=None):
        """Read TGLF files using A1; preserve all cases, including signed fluxes."""
        root = Path(root)
        loader_path = self.a2_dir.parent / "A1/analysis/sat3_loader.py"
        spec = importlib.util.spec_from_file_location("a3_tglf_loader", loader_path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        if (root / "tglf_sat2").is_dir():
            records, errors = [loader.load_case(root)], []
        else:
            records, errors = loader.load_cases(root)
        if errors:
            raise ValueError(f"TGLF parsing failed: {errors}")
        table = pd.DataFrame(records).set_index("case").sort_index()
        if not table.index.is_unique or (expected_count is not None and len(table) != expected_count):
            raise ValueError(f"Expected {expected_count} unique cases in {root}; found {len(table)}.")
        if not (table.NS.eq(2) & table.ZS_1.lt(0) & table.ZS_2.gt(0)).all():
            raise ValueError("Expected species 1 = electron and species 2 = ion.")
        np.testing.assert_allclose(table.RLNS_1, table.RLNS_2, rtol=0, atol=1e-10)
        prep = json.loads((self.a2_dir / "prepared/preparation_metadata.json").read_text())
        source = pd.read_csv(self.a2_dir / "prepared/scan_db_table.csv", nrows=1)
        for column in prep["constant_inputs_dropped"]:
            np.testing.assert_allclose(table[column], source[column].iloc[0], rtol=1e-8, atol=1e-10)
        table["Q_i"], table["Q_e"], table["Gamma_e"] = table.Q_2, table.Q_1, table.Gamma_1
        if not np.isfinite(table[TARGETS].to_numpy()).all():
            raise ValueError("Non-finite reference fluxes.")
        return table, engineer_features(table, self.columns), table[TARGETS]


def summarize_members(values, index, members):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(members), len(index), 3) or not np.isfinite(values).all():
        raise ValueError("Invalid member prediction array.")
    identity = [{"seed": saved["config"]["seed"], "sha256": saved["a3_checkpoint_sha256"]}
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
            "interpretation": "Empirical calibration on reused A2 validation data; no formal coverage guarantee."}


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
    if not (table.XNUE.gt(0) & table.Q_LOC.gt(0)).all():
        raise ValueError("XNUE and Q_LOC must be positive.")
    result = table.assign(log10_XNUE=np.log10(table.XNUE),
                          magnetic_shear=table.Q_PRIME_LOC * table.RMIN_LOC**2 / table.Q_LOC**2)
    result = result.loc[:, columns].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Non-finite external features.")
    return result


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
