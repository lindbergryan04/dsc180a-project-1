"""CPU training, checkpoint loading, and prediction for the Task 2 and Task 3 ensembles."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ["Q_i", "Q_e", "Gamma_e"]
SEEDS = tuple(range(42, 47))
SCALER_KEYS = ("input_mean", "input_scale", "target_mean", "target_scale", "raw_training_std")
TABLE_FILES = ("features.csv", "targets_model.csv", "targets_raw.csv", "split_assignments.csv")


@dataclass
class Config:
    seed: int = 42
    hidden_activation: str = "gelu"
    bottleneck_activation: str = "relu"
    batch_size: int = 256
    pretrain_epochs: int = 30
    epochs: int = 120
    patience: int = 25
    eval_every: int = 5
    warmup_lr: float = 0.001
    encoder_lr: float = 0.003
    weight_decay: float = 0.0001


def config_from_saved(values):
    return Config(**values)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_data(prepared_dir, train_ids=None):
    """Fit both scalers on the requested training rows and reserve test labels."""
    root = Path(prepared_dir)
    meta = json.loads((root / "preparation_metadata.json").read_text())
    hashes = {name: digest(root / name) for name in TABLE_FILES}
    if hashes != meta["checksums_sha256"]:
        raise ValueError("Prepared tables do not match the recorded hashes.")
    x, y, raw, split = [
        pd.read_csv(root / name, index_col="case", float_precision="round_trip")
        for name in TABLE_FILES
    ]
    if not (x.index.equals(y.index) and x.index.equals(raw.index)):
        raise ValueError("Feature and target rows are misaligned.")
    assigned = split.loc[x.index, "split"]
    train = x.index[assigned.eq("train")]
    if train_ids is not None:
        if not pd.Index(train_ids).isin(train).all():
            raise ValueError("Training subsets must contain only training cases.")
        train = train[train.isin(train_ids)]
    ids = {"train": train, "validation": x.index[assigned.eq("validation")]}
    xs = StandardScaler().fit(x.loc[train])
    ys = StandardScaler().fit(y.loc[train])
    data = dict(feature_columns=list(x), hashes=hashes, ids=ids,
                input_mean=xs.mean_, input_scale=xs.scale_, target_mean=ys.mean_,
                target_scale=ys.scale_, raw_training_std=raw.loc[train].std(ddof=0).to_numpy())
    if (data["raw_training_std"] <= 0).any():
        raise ValueError("Each flux needs positive training variance for normalized RMSE.")
    for group, indices in ids.items():
        data[f"X_{group}"] = xs.transform(x.loc[indices]).astype(np.float32)
        data[f"y_{group}"] = ys.transform(y.loc[indices]).astype(np.float32)
        data[f"raw_{group}"] = raw.loc[indices].to_numpy()
    return data


def network(n_features, cfg):
    activations = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    hidden = activations[cfg.hidden_activation]
    return nn.Sequential(nn.Linear(n_features, 64), hidden(), nn.Linear(64, 64), hidden(),
                         nn.Linear(64, 8), activations[cfg.bottleneck_activation](),
                         nn.Linear(8, len(TARGETS)))


def inverse_mean(transformed):
    result = np.asarray(transformed, dtype=np.float64).copy()
    with np.errstate(over="raise", invalid="raise"):
        result[:, :2] = 10.0 ** result[:, :2]
    return result


@torch.no_grad()
def predict(model, x, data, device="cpu", batch_size=256):
    model.eval()
    transformed = np.concatenate([
        model(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ]).astype(np.float64) * data["target_scale"] + data["target_mean"]
    return {"prediction": inverse_mean(transformed), "transformed_mean": transformed}


def fit_trial(config, data, out_dir, device="cpu"):
    """Initialize and shuffle by member seed. Restart AdamW after warmup."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    rng = torch.Generator().manual_seed(config.seed)
    features = torch.as_tensor(data["X_train"], device=device)
    targets = torch.as_tensor(data["y_train"], device=device)
    loader = DataLoader(TensorDataset(features, targets), batch_size=config.batch_size,
                        shuffle=True, generator=rng)
    model = network(features.shape[1], config).to(device)

    def train_epoch(optimizer):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Training produced nonfinite loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

    def validation_score():
        pred = predict(model, data["X_validation"], data, device)["prediction"]
        rmse = np.sqrt(np.mean((pred - data["raw_validation"]) ** 2, axis=0))
        return float(np.mean(rmse / data["raw_training_std"]))

    def save(epoch):
        torch.save(dict(config=asdict(config), state_dict=model.state_dict(), best_epoch=epoch,
                        feature_columns=data["feature_columns"], source_sha256=data["hashes"],
                        source_code_sha256=digest(__file__),
                        **{key: data[key].tolist() for key in SCALER_KEYS}), out_dir / "checkpoint.pt")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.warmup_lr,
                                  weight_decay=config.weight_decay)
    for _ in range(config.pretrain_epochs):
        train_epoch(optimizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.encoder_lr,
                                  weight_decay=config.weight_decay)
    rng.manual_seed(config.seed + 1)
    best_epoch, best = 0, validation_score()
    save(best_epoch)
    for epoch in range(1, config.epochs + 1):
        train_epoch(optimizer)
        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            score = validation_score()
            if score < best - 1e-5:
                best_epoch, best = epoch, score
                save(best_epoch)
        if epoch - best_epoch >= config.patience:
            break


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = network(len(saved["feature_columns"]), config_from_saved(saved["config"])).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    for key in SCALER_KEYS:
        saved[key] = np.asarray(saved[key])
    return model, saved


def check_manifest(directory, data=None):
    root = Path(directory)
    manifest = json.loads((root / "member_manifest.json").read_text())
    if manifest["seeds"] != list(SEEDS) or [e["seed"] for e in manifest["members"]] != list(SEEDS):
        raise ValueError("Expected exactly the five Task 2 member seeds.")
    if data is not None and manifest["source_sha256"] != data["hashes"]:
        raise ValueError("These models were trained on different prepared inputs.")
    for entry in manifest["members"]:
        path = root / entry["directory"] / "checkpoint.pt"
        if digest(path) != entry["checksums_sha256"]["checkpoint.pt"]:
            raise ValueError(f"Checkpoint differs from manifest: {path}")
    return manifest


def load_ensemble(directory, device="cpu"):
    manifest = check_manifest(directory)
    return [load_checkpoint(Path(directory) / e["directory"] / "checkpoint.pt", device)
            for e in manifest["members"]]
