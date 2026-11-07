"""Rebuild Ryan's scan_db ensemble experiment from raw TGLF case directories."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tglf_data import INPUTS, load_cases
from task2.nn_ensemble import (Config, SEEDS, TABLE_FILES, TARGETS, digest, fit_trial, inverse_mean,
                         load_data, load_ensemble, predict)

DIRECT = ["RLTS_1", "RLTS_2", "RLNS_1", "TAUS_2", "Q_LOC", "KAPPA_LOC",
          "DRMAJDX_LOC", "RMIN_LOC", "MASS_2"]
VARIANTS = ("gelu_tanh", "relu_tanh", "gelu_relu", "relu_relu")


def make_features(frame):
    if frame["XNUE"].le(0).any() or frame["Q_LOC"].eq(0).any():
        raise ValueError("Computing features requires positive XNUE and nonzero Q_LOC.")
    if not np.allclose(frame["RLNS_1"], frame["RLNS_2"], rtol=0, atol=1e-12):
        raise ValueError("This model assumes equal electron and ion density gradients.")
    result = frame[DIRECT].copy()
    result["log10_XNUE"] = np.log10(frame["XNUE"])
    result["magnetic_shear"] = frame["Q_PRIME_LOC"] * frame["RMIN_LOC"]**2 / frame["Q_LOC"]**2
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite engineered input.")
    return result


def prepare(frame, output):
    """Split all cases first, then keep positive-heat rows for modeling."""
    holdout = train_test_split(frame.index.to_numpy(), test_size=0.30, random_state=42)[1]
    validation, test = train_test_split(holdout, test_size=0.50, random_state=42)
    split = pd.Series("train", index=frame.index, name="split")
    split.loc[validation], split.loc[test] = "validation", "test"
    eligible = frame[["Q_i", "Q_e"]].gt(0).all(axis=1)
    raw = frame.loc[eligible, TARGETS].copy()
    model_targets = raw.copy()
    model_targets[["Q_i", "Q_e"]] = np.log10(raw[["Q_i", "Q_e"]])
    model_targets.columns = ["log10_Q_i", "log10_Q_e", "Gamma_e"]
    features = make_features(frame.loc[eligible])
    if features.duplicated().any():
        raise ValueError("Duplicate scan inputs should be reviewed before splitting.")
    output.mkdir()
    for name, table in zip(TABLE_FILES, (features, model_targets, raw, split), strict=True):
        table.to_csv(output / name, index_label="case")
    train = frame.loc[eligible & split.eq("train"), list(INPUTS)]
    meta = {
        "features": list(features), "excluded_cases": frame.index[~eligible].tolist(),
        "constant_inputs": {k: float(train[k].iloc[0]) for k in INPUTS if train[k].nunique() == 1},
        "checksums_sha256": {name: digest(output / name) for name in TABLE_FILES},
    }
    (output / "preparation_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return features, raw, model_targets, split


def train_members(data, folder, config):
    entries = []
    for seed in SEEDS:
        member_dir = folder / f"nn_seed{seed}"
        fit_trial(replace(config, seed=seed), data, member_dir)
        entries.append({"seed": seed, "directory": member_dir.name,
                        "checksums_sha256": {"checkpoint.pt": digest(member_dir / "checkpoint.pt")}})
    manifest = dict(seeds=list(SEEDS), fixed_nn_config=asdict(config), source_sha256=data["hashes"],
                    members=entries, training_code_sha256=digest(Path(__file__).with_name("nn_ensemble.py")))
    (folder / "member_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return load_ensemble(folder)


def predict_members(members, x):
    saved = members[0][1]
    scaled = ((np.asarray(x) - saved["input_mean"]) / saved["input_scale"]).astype(np.float32)
    return np.mean([predict(model, scaled, data)["prediction"] for model, data in members], axis=0)


def prediction_functions(members, x, raw, transformed, train):
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(x.loc[train].to_numpy(),
                                                                     transformed.loc[train].to_numpy())
    mean = raw.loc[train].mean().to_numpy()
    return {"mean": lambda cases: np.tile(mean, (len(cases), 1)),
            "linear": lambda cases: inverse_mean(linear.predict(np.asarray(cases))),
            "ensemble": lambda cases: predict_members(members, cases)}


def metrics(true, predicted, scale, **labels):
    rmse = np.sqrt(np.mean((true - predicted) ** 2, axis=0))
    r2 = r2_score(true, predicted, multioutput="raw_values")
    return [dict(**labels, target=target, n_cases=len(true), rmse=value, nrmse=value/scale[i], r2=r2[i])
            for i, (target, value) in enumerate(zip(TARGETS, rmse, strict=True))]


def select_model(x, raw, data, config, output, compare, sat3):
    validation = data["ids"]["validation"]
    variants = VARIANTS if compare else ("gelu_relu",)
    rows, scores, durations = [], {}, {}
    with TemporaryDirectory(dir=output) as temp:
        for variant in variants:
            hidden, bottleneck = variant.split("_")
            cfg = replace(config, hidden_activation=hidden, bottleneck_activation=bottleneck)
            start = perf_counter()
            members = train_members(data, Path(temp) / variant, cfg)
            durations[variant] = perf_counter() - start
            predictions = predict_members(members, x.loc[validation])
            result = metrics(raw.loc[validation].to_numpy(), predictions, data["raw_training_std"],
                             variant=variant)
            scores[variant] = np.mean([row["nrmse"] for row in result])
            rows.extend(result)
        best = min(scores, key=scores.get)
        # The transfer cases never enter activation selection.
        if compare and sat3 is not None:
            transfer = []
            features = make_features(sat3)
            for variant in variants:
                members = load_ensemble(Path(temp) / variant)
                predictions = predict_members(members, features)
                transfer.extend(metrics(sat3[TARGETS].to_numpy(), predictions,
                                        data["raw_training_std"], variant=variant))
            pd.DataFrame(transfer).to_csv(output / "activation_sat3.csv", index=False)
        (Path(temp) / best).rename(output / "models")
    pd.DataFrame(rows).to_csv(output / "activation_validation.csv", index=False)
    print(f"Selected {best} (validation mean nRMSE {scores[best]:.6f}).", flush=True)
    return load_ensemble(output / "models"), replace(config, hidden_activation=best.split("_")[0],
                                                    bottleneck_activation=best.split("_")[1]), durations[best]


def evaluate(label, x, raw, functions, scale, output):
    rows = []
    predictions = raw.copy()
    for name, fn in functions.items():
        values = fn(x)
        rows.extend(metrics(raw.to_numpy(), values, scale, dataset=label, model=name))
        for i, target in enumerate(TARGETS):
            predictions[f"{name}_{target}"] = values[:, i]
    predictions.to_csv(output / f"{label}_predictions.csv", index_label="case")
    pd.DataFrame(rows).to_csv(output / f"{label}_metrics.csv", index=False)
    return predictions


def sat3_trends(predictions, output):
    groups = [("temperature_gradient", [f"{letter}_{iso}" for letter in "bcad"]) for iso in "HDT"]
    letters = sorted({case.rsplit("_", 1)[0] for case in predictions.index})
    groups += [("isotope", [f"{letter}_{iso}" for iso in "HDT"]) for letter in letters]
    rows = []
    for kind, cases in groups:
        if not set(cases).issubset(predictions.index):
            continue
        for a, b in zip(cases, cases[1:]):
            for target in TARGETS:
                true = predictions.loc[b, target] - predictions.loc[a, target]
                pred = predictions.loc[b, "ensemble_" + target] - predictions.loc[a, "ensemble_" + target]
                same = (np.sign(true) if abs(true) > 1e-8 else 0) == (np.sign(pred) if abs(pred) > 1e-8 else 0)
                rows.append((kind, a, b, target, true, pred, same))
    pd.DataFrame(rows, columns=["scan", "from", "to", "target", "true_change", "predicted_change",
                                "direction_matches"]).to_csv(output / "sat3_trends.csv", index=False)


def learning_curves(output, prepared, x, raw, transformed, test, full_data, full_members, cfg, smoke):
    train = full_data["ids"]["train"]
    ordering = np.random.default_rng(2026).permutation(train)
    sizes = [64, 128] if smoke else [250, 500, 1000, 2000]
    rows = []
    with TemporaryDirectory(dir=output) as temp:
        for n in [n for n in sizes if n < len(train)] + [len(train)]:
            ids = train[train.isin(ordering[:n])]
            members = full_members if n == len(train) else train_members(
                load_data(prepared, ids), Path(temp) / str(n), cfg)
            functions = prediction_functions(members, x, raw, transformed, ids)
            for name, fn in functions.items():
                rows.extend(metrics(raw.loc[test].to_numpy(), fn(x.loc[test]),
                                    full_data["raw_training_std"], n_train=n, model=name))
    table = pd.DataFrame(rows)
    table.to_csv(output / "learning_curves.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        for name, part in table[table.target.eq(target)].groupby("model"):
            ax.plot(part.n_train, part.nrmse, marker="o", label=name)
        ax.set(xlabel="Training cases", ylabel="Test normalized RMSE", title=target)
    axes[0].legend()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def timing(functions, x, smoke):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    results = []
    try:
        with threadpool_limits(limits=1):
            for name, fn in functions.items():
                for mode, cases, calls in (("single", x[:1], 256), ("batch", x, 50)):
                    measurements = []
                    fn(cases)
                    for _ in range(2 if smoke else 7):
                        start = perf_counter()
                        repeats = 2 if smoke else calls
                        for j in range(repeats):
                            fn(x[j % min(128, len(x)):j % min(128, len(x)) + 1] if mode == "single" else cases)
                        measurements.append(1e6 * (perf_counter() - start) / (repeats * len(cases)))
                    results.append((name, mode, len(cases), np.median(measurements)))
    finally:
        torch.set_num_threads(threads)
    return pd.DataFrame(results, columns=["model", "mode", "batch_size", "median_us_per_case"])


def run(args, output):
    torch.set_num_threads(args.threads)
    table = load_cases(args.scan_data)
    if not args.smoke and set(table.index) != {f"r{i:04d}" for i in range(1, 5001)}:
        raise ValueError("Full runs require the 5,000 scan cases r0001 through r5000.")
    prepared = output / "prepared"
    x, raw, transformed, split = prepare(table, prepared)
    data = load_data(prepared)
    cfg = Config(pretrain_epochs=2, epochs=3, eval_every=1, batch_size=64) if args.smoke else Config()
    sat3 = load_cases(args.sat3_data) if args.sat3_data else None
    if sat3 is not None and not args.smoke and len(sat3) != 43:
        raise ValueError("Full transfer evaluation requires all 43 SAT3 cases.")
    members, cfg, seconds = select_model(x, raw, data, cfg, output, args.activations, sat3)
    functions = prediction_functions(members, x, raw, transformed, data["ids"]["train"])
    test = x.index[split.loc[x.index].eq("test")]
    evaluate("test", x.loc[test], raw.loc[test], functions, data["raw_training_std"], output)
    if sat3 is not None:
        predictions = evaluate("sat3", make_features(sat3), sat3[TARGETS], functions,
                               data["raw_training_std"], output)
        sat3_trends(predictions, output)
    if args.learning_curves:
        learning_curves(output, prepared, x, raw, transformed, test, data, members, cfg, args.smoke)
    timing(functions, x.loc[test].to_numpy(), args.smoke).to_csv(output / "inference_timing.csv", index=False)
    info = dict(smoke=args.smoke, threads=args.threads, train=len(data["ids"]["train"]),
                validation=len(data["ids"]["validation"]), test=len(test), fit_seconds=seconds,
                config=asdict(cfg))
    (output / "run.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-data", type=Path, required=True, help="scan_db/data case directory")
    parser.add_argument("--sat3-data", type=Path, help="SAT3_dataset/data case directory")
    parser.add_argument("--output", type=Path, required=True, help="new directory for generated results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--activations", action="store_true", help="compare all four combinations on validation")
    parser.add_argument("--learning-curves", action="store_true", help="train on nested subsets")
    parser.add_argument("--smoke", action="store_true", help="use short schedules to check execution")
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("Use positive --threads and a new --output directory.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        pending = Path(temp) / "results"
        pending.mkdir()
        with threadpool_limits(limits=args.threads):
            run(args, pending)
        pending.rename(args.output)
