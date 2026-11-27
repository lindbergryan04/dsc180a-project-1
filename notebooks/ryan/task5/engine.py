"""Training, reversible preprocessing, and checkpointed five-member ensembles.

``run_case`` accepts training and validation frames only. Test evaluation is a
separate explicit call to ``evaluate`` after configuration selection is frozen.
Checkpoints contain plain Python values and tensors, so weights_only=True works.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import PowerTransformer, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["RLTS_1", "RLTS_2", "RLNS_1", "TAUS_2", "XNUE", "RMIN_LOC"]
TARGETS = ["Q_e", "Q_i", "Gamma_e"]
TRANSFORMS = {"identity", "log10", "log1p", "signed_log1p", "asinh", "yeo_johnson"}
ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU,
               "tanh": nn.Tanh, "identity": nn.Identity}
INTEGER_FIELDS = ["width", "depth", "bottleneck_width", "ensemble_size", "warmup_epochs",
                  "main_epochs", "batch_size", "patience_epochs", "eval_every_epochs"]
FLOAT_FIELDS = ["warmup_lr", "main_lr", "weight_decay", "gradient_clip_norm"]
SCHEMA_VERSION = 1


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_jsonable(value), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _atomic_csv(frame, path):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _save_bundle(bundle, path):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(bundle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def normalize_config(config):
    """Resolve CSV strings and require the protocol's five distinct member seeds."""
    c = {k: ("" if v is None or isinstance(v, float) and math.isnan(v) else _jsonable(v))
         for k, v in dict(config).items()}
    for name in INTEGER_FIELDS:
        c[name] = int(c[name])
    for name in FLOAT_FIELDS:
        c[name] = float(c[name])
    c["repeat"] = int(c.get("repeat", 0))
    seeds = c["member_seeds"]
    if isinstance(seeds, str):
        seeds = seeds.split("|")
    c["member_seeds"] = [int(seed) for seed in seeds]
    if c["ensemble_size"] != 5 or len(c["member_seeds"]) != 5 or len(set(c["member_seeds"])) != 5:
        raise ValueError("An ensemble must have exactly five distinct member seeds.")
    if c["output_mode"] not in ["joint", *TARGETS]:
        raise ValueError(f"Unknown output mode {c['output_mode']!r}")
    for name in ["width", "depth", "bottleneck_width", "batch_size", "main_epochs",
                 "patience_epochs", "eval_every_epochs"]:
        if c[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if c["warmup_epochs"] < 0:
        raise ValueError("warmup_epochs cannot be negative")
    if c["hidden_activation"] not in ACTIVATIONS or c["bottleneck_activation"] not in ACTIVATIONS:
        raise ValueError("Unknown activation")
    for name in FEATURES:
        if c[f"input_{name}"] not in TRANSFORMS:
            raise ValueError(f"Unknown input transform for {name}")
    for name in targets_for(c):
        if c[f"target_{name}"] not in TRANSFORMS:
            raise ValueError(f"Unknown target transform for {name}")
    return c


def targets_for(config):
    return list(TARGETS) if config["output_mode"] == "joint" else [config["output_mode"]]


def parameter_count(config):
    dims = [len(FEATURES)] + [int(config["width"])] * int(config["depth"])
    dims += [int(config["bottleneck_width"]), len(targets_for(config))]
    return sum((a + 1) * b for a, b in zip(dims, dims[1:]))


def build_model(config):
    layers = []
    current = len(FEATURES)
    for _ in range(int(config["depth"])):
        layers += [nn.Linear(current, int(config["width"])), ACTIVATIONS[config["hidden_activation"]]()]
        current = int(config["width"])
    layers += [nn.Linear(current, int(config["bottleneck_width"])),
               ACTIVATIONS[config["bottleneck_activation"]](),
               nn.Linear(int(config["bottleneck_width"]), len(targets_for(config)))]
    return nn.Sequential(*layers)


def _finite(values, label, ids=None):
    bad = np.flatnonzero(~np.isfinite(values))
    if len(bad):
        samples = [str(ids[i]) for i in bad[:12]] if ids is not None else bad[:12].tolist()
        raise ValueError(f"{label}: {len(bad)} nonfinite values, affected cases {samples}")
    return values


def _yeo_johnson(x, power, inverse=False):
    """Numerically stable Yeo–Johnson with explicit inverse-domain validation."""
    z = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    if not inverse:
        if abs(power) < np.spacing(1.0):
            z[positive] = np.log1p(x[positive])
        else:
            z[positive] = np.expm1(power * np.log1p(x[positive])) / power
        if abs(power - 2) < np.spacing(1.0):
            z[~positive] = -np.log1p(-x[~positive])
        else:
            z[~positive] = -np.expm1((2 - power) * np.log1p(-x[~positive])) / (2 - power)
    else:
        if abs(power) < np.spacing(1.0):
            z[positive] = np.expm1(x[positive])
        else:
            argument = power * x[positive]
            if np.any(argument <= -1):
                raise ValueError("Yeo–Johnson positive-branch inverse is outside its fitted domain")
            z[positive] = np.expm1(np.log1p(argument) / power)
        if abs(power - 2) < np.spacing(1.0):
            z[~positive] = -np.expm1(-x[~positive])
        else:
            argument = -(2 - power) * x[~positive]
            if np.any(argument <= -1):
                raise ValueError("Yeo–Johnson negative-branch inverse is outside its fitted domain")
            z[~positive] = -np.expm1(np.log1p(argument) / (2 - power))
    return z


def _nonlinear(values, spec, inverse=False, ids=None):
    x = np.asarray(values, dtype=np.float64)
    _finite(x, spec["name"], ids)
    kind, s = spec["kind"], spec["scale"]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if kind == "identity":
                out = x.copy()
            elif kind == "log10":
                if not inverse and np.any(x <= 0):
                    bad = np.flatnonzero(x <= 0)
                    cases = [str(ids[i]) for i in bad[:12]] if ids is not None else bad[:12].tolist()
                    raise ValueError(f"log10 requires positive values, affected cases {cases}")
                out = 10.0 ** x if inverse else np.log10(x)
            elif kind == "log1p":
                out = s * np.expm1(x) if inverse else np.log1p(x / s)
            elif kind == "signed_log1p":
                out = s * np.sign(x) * np.expm1(np.abs(x)) if inverse else np.sign(x) * np.log1p(np.abs(x) / s)
            elif kind == "asinh":
                out = s * np.sinh(x) if inverse else np.arcsinh(x / s)
            elif kind == "yeo_johnson":
                if inverse:
                    power = spec["power"]
                    bad = ((x >= 0) & (power * x <= -1)) if abs(power) >= np.spacing(1.0) else np.zeros(len(x), dtype=bool)
                    if abs(power - 2) >= np.spacing(1.0):
                        bad |= (x < 0) & (-(2 - power) * x <= -1)
                    if bad.any():
                        positions = np.flatnonzero(bad)
                        cases = [str(ids[i]) for i in positions[:12]] if ids is not None else positions[:12].tolist()
                        raise ValueError(f"Yeo–Johnson inverse outside fitted domain, affected cases {cases}")
                out = _yeo_johnson(x, spec["power"], inverse)
            else:
                raise ValueError(f"Unsupported transform {kind}")
    except (FloatingPointError, ValueError) as exc:
        raise ValueError(f"{spec['name']} {kind} {'inverse' if inverse else 'forward'} failed: {exc}") from exc
    return _finite(out, spec["name"] + (" inverse" if inverse else " transform"), ids)


class Preprocessor:
    """Train-fitted column transforms and StandardScaler state, stored as primitives."""

    def __init__(self, state):
        self.state = state
        self.columns = [spec["name"] for spec in state["columns"]]

    @classmethod
    def fit(cls, frame, names, kinds):
        ids = sample_ids(frame)
        specs, transformed = [], []
        for name, kind in zip(names, kinds):
            x = frame[name].to_numpy(dtype=np.float64)
            _finite(x, name, ids)
            spec = {"name": name, "kind": kind, "scale": max(float(np.median(np.abs(x))), 1e-8)}
            if kind == "yeo_johnson":
                power = PowerTransformer(method="yeo-johnson", standardize=False).fit(x[:, None])
                spec["power"] = float(power.lambdas_[0])
            transformed.append(_nonlinear(x, spec, ids=ids))
            specs.append(spec)
        raw = np.column_stack(transformed)
        scaler = StandardScaler().fit(raw)
        return cls({"columns": specs, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                    "n_training_cases": len(frame)})

    def transform(self, frame):
        ids = sample_ids(frame)
        raw = np.column_stack([_nonlinear(frame[s["name"]].to_numpy(dtype=np.float64), s, ids=ids)
                               for s in self.state["columns"]])
        out = (raw - np.asarray(self.state["mean"])) / np.asarray(self.state["scale"])
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite standardized features or targets")
        return out

    def inverse_transform(self, values, ids=None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError("Prediction shape disagrees with target preprocessing")
        raw = values * np.asarray(self.state["scale"]) + np.asarray(self.state["mean"])
        return np.column_stack([_nonlinear(raw[:, i], spec, inverse=True, ids=ids)
                                for i, spec in enumerate(self.state["columns"])])


def sample_ids(frame):
    return frame["sample_id"].astype(str).to_numpy() if "sample_id" in frame else frame.index.astype(str).to_numpy()


def _frame_hash(frame, targets):
    cols = FEATURES + targets
    hashed = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy().tobytes()
    h = hashlib.sha256(json.dumps(cols).encode() + hashed)
    h.update(json.dumps(sample_ids(frame).tolist()).encode())
    return h.hexdigest()


def _state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _tree_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tree_cpu(v) for v in value)
    return value


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _runtime(device, threads):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "device": str(device), "threads": int(threads)}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


@torch.no_grad()
def _predict_model(model, x, yprocessor, device, batch_size=256, ids=None):
    model.eval()
    if not len(x):
        return np.empty((0, len(yprocessor.columns)), dtype=np.float64)
    chunks = []
    for start in range(0, len(x), batch_size):
        xt = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model(xt).detach().cpu().numpy())
    return yprocessor.inverse_transform(np.concatenate(chunks), ids)


def _score(truth, prediction, scale):
    return float(np.mean(np.sqrt(np.mean((truth - prediction) ** 2, axis=0)) / scale))


def _peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss / (1024 ** 2 if sys.platform == "darwin" else 1024))


class Ensemble:
    def __init__(self, bundle, device="cpu"):
        self.config = bundle["config"]
        self.targets = targets_for(self.config)
        self.raw_training_std = np.asarray(bundle["raw_training_std"])
        self.xprocessor = Preprocessor(bundle["input_preprocessing"])
        self.yprocessor = Preprocessor(bundle["target_preprocessing"])
        self.device = device
        if len(bundle["members"]) != 5 or any(m["phase"] != "complete" for m in bundle["members"]):
            raise ValueError("All five members must finish before using an ensemble")
        self.models = []
        for member in bundle["members"]:
            model = build_model(self.config)
            model.load_state_dict(member["best_weights"])
            self.models.append(model.to(device).eval())

    def predict(self, frame):
        x = self.xprocessor.transform(frame)
        ids = sample_ids(frame)
        members = np.stack([_predict_model(m, x, self.yprocessor, self.device,
                                          self.config["batch_size"], ids) for m in self.models])
        return {"prediction": members.mean(axis=0), "members": members}


def load_ensemble(run_dir, device="cpu"):
    """Load a locally generated checkpoint with safe weights-only deserialization."""
    bundle = torch.load(Path(run_dir) / "checkpoints.pt", map_location="cpu", weights_only=True)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    return Ensemble(bundle, device)


def metric_rows(frame, prediction, targets, raw_training_std, config, *, split="validation", member_seed="ensemble"):
    """Return original-unit error rows for one member or a full ensemble."""
    rows = []
    truth = frame[targets].to_numpy(dtype=np.float64)
    for i, target in enumerate(targets):
        delta = prediction[:, i] - truth[:, i]
        mse = float(np.mean(delta ** 2))
        denominator = float(np.sum((truth[:, i] - truth[:, i].mean()) ** 2))
        r2 = 1 - float(np.sum(delta ** 2)) / denominator if len(frame) > 1 and denominator > 0 else np.nan
        rows.append({"case_id": config["case_id"], "repeat": int(config.get("repeat", 0)),
                     "member_seed": str(member_seed), "split": split, "target": target,
                     "n_cases": len(frame), "rmse": math.sqrt(mse),
                     "nrmse": math.sqrt(mse) / float(raw_training_std[i]),
                     "mae": float(np.mean(np.abs(delta))), "r2": r2,
                     "bias": float(np.mean(delta)),
                     "negative_heat_rate": float(np.mean(prediction[:, i] < 0)) if target in ["Q_e", "Q_i"] else np.nan,
                     "parameters_total": parameter_count(config) * (5 if member_seed == "ensemble" else 1)})
    return rows


def prediction_rows(frame, prediction, targets, config, *, split="validation", member_seed="ensemble"):
    frames = []
    for i, target in enumerate(targets):
        frames.append(pd.DataFrame({"case_id": config["case_id"], "repeat": int(config.get("repeat", 0)),
                                    "member_seed": str(member_seed), "split": split,
                                    "sample_id": sample_ids(frame), "target": target,
                                    "r_over_a": frame["RMIN_LOC"].to_numpy(),
                                    "truth": frame[target].to_numpy(), "prediction": prediction[:, i]}))
    return pd.concat(frames, ignore_index=True)


def _evaluate_model(ensemble, frame, split):
    result = ensemble.predict(frame)
    predictions, metrics, radial = [], [], []
    seeds = ensemble.config["member_seeds"] + ["ensemble"]
    for seed, prediction in zip(seeds, [*result["members"], result["prediction"]]):
        predictions.append(prediction_rows(frame, prediction, ensemble.targets, ensemble.config,
                                           split=split, member_seed=seed))
        metrics.extend(metric_rows(frame, prediction, ensemble.targets, ensemble.raw_training_std,
                                   ensemble.config, split=split, member_seed=seed))
        for radius in sorted(frame["RMIN_LOC"].unique()):
            mask = frame["RMIN_LOC"].to_numpy() == radius
            rows = metric_rows(frame.loc[mask], prediction[mask], ensemble.targets, ensemble.raw_training_std,
                               ensemble.config, split=split, member_seed=seed)
            radial.extend([dict(row, r_over_a=float(radius)) for row in rows])
    return {"predictions": pd.concat(predictions, ignore_index=True),
            "metrics": pd.DataFrame(metrics), "radius_metrics": pd.DataFrame(radial)}


def evaluate(run_dir, frame, split, device="cpu"):
    """Evaluate a completed ensemble; replace only rows for this named split.

    The orchestration layer controls when test labels can enter this function.
    This call never fits or refits any preprocessing.
    """
    if split not in {"validation", "test", "new_radius", "varied_radius", "profile_test", "external"}:
        raise ValueError(f"Unknown evaluation split {split!r}")
    run_dir = Path(run_dir)
    result = _evaluate_model(load_ensemble(run_dir, device), frame, split)
    for name, new in result.items():
        path = run_dir / f"{name}.csv"
        if path.exists():
            old = pd.read_csv(path, dtype={"member_seed": str, "sample_id": str})
            new = pd.concat([old.loc[old["split"] != split], new], ignore_index=True)
        sort = [c for c in ["split", "member_seed", "target", "sample_id", "r_over_a"] if c in new]
        _atomic_csv(new.sort_values(sort).reset_index(drop=True), path)
    return result


def _histories(bundle):
    rows = []
    for member in bundle["members"]:
        rows.extend([dict(row, case_id=bundle["config"]["case_id"], repeat=bundle["config"]["repeat"],
                          member_seed=member["seed"]) for row in member["history"]])
    return pd.DataFrame(rows)


def run_case(config, train, validation, run_dir, *, device="cpu", threads=2,
             resume=True, stop_after_epochs=None):
    """Fit five members, resuming exactly from saved epoch boundaries.

    ``stop_after_epochs`` limits new epochs in this invocation across all members
    and is intended for restart tests. A paused result is never ensemble-scored.
    Checkpoints are saved every evaluation interval and whenever pausing. Every
    complete member is saved before the next starts. Existing runs are immutable
    with respect to configuration, data, source code, and execution environment.
    """
    c = normalize_config(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoints.pt"
    targets = targets_for(c)
    torch.set_num_threads(int(threads))
    if not len(train) or not len(validation):
        raise ValueError("Training and validation frames must be nonempty")
    train_ids, val_ids = sample_ids(train), sample_ids(validation)
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate sample IDs inside a partition")
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation sample IDs overlap")
    for frame in [train, validation]:
        if not np.isfinite(frame[FEATURES + targets].to_numpy(dtype=float)).all():
            raise ValueError("Inputs or target columns contain nonfinite values")
    runtime = _runtime(device, threads)
    identity = {"config": c, "train_sha256": _frame_hash(train, targets),
                "validation_sha256": _frame_hash(validation, targets),
                "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runtime": runtime}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config_path = run_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["identity_sha256"] != identity_sha:
            raise ValueError(f"Run identity differs from saved config/data/code/runtime at {run_dir}")
        if not resume:
            raise FileExistsError(f"Run already exists: {run_dir}")
    atomic_json(config_path, dict(identity, identity_sha256=identity_sha, status="running"))
    epochs_this_call = 0
    bundle = None
    try:
        if checkpoint.exists():
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if bundle["identity_sha256"] != identity_sha:
                raise ValueError("Checkpoint identity does not match current inputs")
        else:
            xprocessor = Preprocessor.fit(train, FEATURES, [c[f"input_{name}"] for name in FEATURES])
            yprocessor = Preprocessor.fit(train, targets, [c[f"target_{name}"] for name in targets])
            raw_std = train[targets].std(ddof=0).to_numpy(dtype=float)
            if not np.all(raw_std > 0):
                raise ValueError("Every target must have positive training standard deviation")
            bundle = {"schema_version": SCHEMA_VERSION, "identity_sha256": identity_sha, "config": c,
                      "input_preprocessing": xprocessor.state, "target_preprocessing": yprocessor.state,
                      "raw_training_std": raw_std.tolist(), "members": []}
            _save_bundle(bundle, checkpoint)
        if all(m["phase"] == "complete" for m in bundle["members"]) and len(bundle["members"]) == 5:
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["status"] == "complete":
                    atomic_json(config_path, dict(identity, identity_sha256=identity_sha, status="complete"))
                    return summary
        xprocessor = Preprocessor(bundle["input_preprocessing"])
        yprocessor = Preprocessor(bundle["target_preprocessing"])
        x = xprocessor.transform(train).astype(np.float32)
        y = yprocessor.transform(train).astype(np.float32)
        xv = xprocessor.transform(validation).astype(np.float32)
        # Validate the validation transform's domain, without fitting on it.
        yprocessor.transform(validation)
        truth = validation[targets].to_numpy(dtype=np.float64)
        raw_std = np.asarray(bundle["raw_training_std"])
        xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
        for member_index, seed in enumerate(c["member_seeds"]):
            if member_index < len(bundle["members"]) and bundle["members"][member_index]["phase"] == "complete":
                continue
            _seed(seed)
            model = build_model(c).to(device)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(TensorDataset(xt, yt), batch_size=c["batch_size"], shuffle=True,
                                generator=generator, num_workers=0)
            if member_index == len(bundle["members"]):
                member = {"seed": seed, "phase": "warmup", "epoch": 0, "best_score": None,
                          "best_epoch": None, "best_weights": None, "weights": _state_cpu(model),
                          "optimizer": None, "generator_state": generator.get_state(),
                          "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                          "history": [], "fit_seconds": 0.0}
                bundle["members"].append(member)
            else:
                member = bundle["members"][member_index]
                model.load_state_dict(member["weights"])
                generator.set_state(member["generator_state"])
                torch.set_rng_state(member["rng_cpu"])
                if member["rng_cuda"]:
                    torch.cuda.set_rng_state_all(member["rng_cuda"])
            lr = c["warmup_lr"] if member["phase"] == "warmup" else c["main_lr"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c["weight_decay"])
            if member["optimizer"] is not None:
                optimizer.load_state_dict(member["optimizer"])
            last_clock = time.perf_counter()

            def persist():
                nonlocal last_clock
                _sync(device)
                member["fit_seconds"] += time.perf_counter() - last_clock
                member["weights"] = _state_cpu(model)
                member["optimizer"] = _tree_cpu(optimizer.state_dict())
                member["generator_state"] = generator.get_state()
                member["rng_cpu"] = torch.get_rng_state()
                member["rng_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                _save_bundle(bundle, checkpoint)
                _atomic_csv(_histories(bundle), run_dir / "history.csv")
                last_clock = time.perf_counter()

            while member["phase"] != "complete":
                if member["phase"] == "warmup" and member["epoch"] >= c["warmup_epochs"]:
                    generator.manual_seed(seed + 1)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=c["main_lr"], weight_decay=c["weight_decay"])
                    prediction = _predict_model(model, xv, yprocessor, device, c["batch_size"], val_ids)
                    score = _score(truth, prediction, raw_std)
                    if not math.isfinite(score):
                        raise ValueError(f"Nonfinite validation score after warmup for seed {seed}")
                    member.update(phase="main", epoch=0, best_score=score, best_epoch=0,
                                  best_weights=_state_cpu(model))
                    member["history"].append({"phase": "main", "epoch": 0, "training_loss": None,
                                              "learning_rate": c["main_lr"], "validation_nrmse": score})
                    persist()
                    continue
                if stop_after_epochs is not None and epochs_this_call >= stop_after_epochs:
                    persist()
                    summary = {"status": "paused", "case_id": c["case_id"], "repeat": c["repeat"],
                               "members_complete": sum(m["phase"] == "complete" for m in bundle["members"]),
                               "fit_seconds": sum(m["fit_seconds"] for m in bundle["members"])}
                    atomic_json(run_dir / "summary.json", summary)
                    atomic_json(config_path, dict(identity, identity_sha256=identity_sha, status="paused"))
                    return summary
                phase = member["phase"]
                epoch = member["epoch"] + 1
                model.train()
                total = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(xb), yb)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite loss for seed {seed}, {phase} epoch {epoch}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach()) * len(xb)
                member["epoch"] = epoch
                epochs_this_call += 1
                row = {"phase": phase, "epoch": epoch, "training_loss": total / len(xt),
                       "learning_rate": c["warmup_lr"] if phase == "warmup" else c["main_lr"],
                       "validation_nrmse": None}
                if phase == "main" and (epoch == 1 or epoch % c["eval_every_epochs"] == 0 or epoch == c["main_epochs"]):
                    prediction = _predict_model(model, xv, yprocessor, device, c["batch_size"], val_ids)
                    score = _score(truth, prediction, raw_std)
                    if not math.isfinite(score):
                        raise ValueError(f"Nonfinite validation score for seed {seed}")
                    row["validation_nrmse"] = score
                    if score < member["best_score"] - 1e-5:
                        member.update(best_score=score, best_epoch=epoch, best_weights=_state_cpu(model))
                member["history"].append(row)
                if phase == "main" and (epoch >= c["main_epochs"] or epoch - member["best_epoch"] >= c["patience_epochs"]):
                    member["phase"] = "complete"
                    member["epochs_run"] = epoch
                    member["hit_epoch_limit"] = epoch >= c["main_epochs"]
                if epoch % c["eval_every_epochs"] == 0 or member["phase"] == "complete":
                    persist()
            print(f"{c['case_id']} repeat {c['repeat']} seed {seed}: best validation NRMSE={member['best_score']:.5f}, epoch {member['best_epoch']}", flush=True)
        ensemble = Ensemble(bundle, device)
        tables = _evaluate_model(ensemble, validation, "validation")
        for name, frame in tables.items():
            _atomic_csv(frame, run_dir / f"{name}.csv")
        # Warm model kernels before timing; include preprocessing and inverse transforms.
        ensemble.predict(validation.iloc[:1])
        _sync(device)
        timings = {}
        for label, frame, repetitions in [("single_case", validation.iloc[:1], 20),
                                           ("validation_batch", validation, 3)]:
            start = time.perf_counter()
            for _ in range(repetitions):
                ensemble.predict(frame)
            _sync(device)
            timings[label + "_seconds"] = (time.perf_counter() - start) / repetitions
        emetrics = tables["metrics"].loc[tables["metrics"]["member_seed"] == "ensemble"]
        summary = {"status": "complete", "case_id": c["case_id"], "repeat": c["repeat"],
                   "targets": targets, "validation_nrmse": float(emetrics["nrmse"].mean()),
                   "fit_seconds": sum(m["fit_seconds"] for m in bundle["members"]),
                   "parameters_per_member": parameter_count(c), "parameters_total": parameter_count(c) * 5,
                   "peak_process_rss_mb": _peak_rss_mb(),
                   "memory_measurement": "Process lifetime maximum resident set; includes earlier cases in this process",
                   "inference_timing": dict(timings, batch_cases=len(validation)),
                   "member_summaries": [{k: m[k] for k in ["seed", "best_score", "best_epoch", "epochs_run", "hit_epoch_limit", "fit_seconds"]}
                                        for m in bundle["members"]], "runtime": runtime}
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(config_path, dict(identity, identity_sha256=identity_sha, status="complete"))
        return summary
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        atomic_json(run_dir / "summary.json", {"status": status, "case_id": c["case_id"],
                    "repeat": c["repeat"], "error": f"{type(exc).__name__}: {exc}",
                    "fit_seconds": sum(m["fit_seconds"] for m in bundle["members"]) if bundle else 0.0})
        atomic_json(config_path, dict(identity, identity_sha256=identity_sha, status=status))
        raise
