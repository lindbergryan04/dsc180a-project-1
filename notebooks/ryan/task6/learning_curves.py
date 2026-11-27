"""Task 6 learning curves and width test for DIII-D shot 128913 (Part B).

For each flux, train fresh five-network ensembles of the Task 5 specialist on nested,
radius-stratified subsets of 1,000 / 2,000 / 4,000 / 8,000 / 16,586 training runs (three draws
below full size), and at 1,000 / 4,000 / 16,586 runs also at half and double hidden width:
81 ensembles in all. Width comparisons and the validation results are frozen before the
original-width curves are scored on the test set.

From the repository root (several hours of CPU)::

    python notebooks/ryan/task6/learning_curves.py train --workers 5
    python notebooks/ryan/task6/learning_curves.py summarize

Inputs: ``data/processed/task6/combined_nbasis12.csv`` (``prepare.py clean``), ``split.csv``,
and ``specialists.json`` in this folder. Everything is written to ``--output`` (default
``notebooks/ryan/runs/task6/learning_curves``): the subset plan, checkpoints, and ``results/``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager, redirect_stdout
from itertools import combinations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[2]
TASK5 = REPO / "notebooks/ryan/task5"
sys.path.insert(0, str(TASK5))
import engine  # noqa: E402  The Task 5 builder, train-only preprocessing, and fitter.

SIZES = (1000, 2000, 4000, 8000, 16586)
WIDTH_SIZES = (1000, 4000, 16586)
DRAW_SEEDS = (180621, 180622, 180623)
MEMBER_SEEDS = (142, 143, 144, 145, 146)
TARGETS = ("Q_e", "Q_i", "Gamma_e")
SCALES = {"narrow": 0.5, "original": 1.0, "wide": 2.0}
SPLIT_PATH = CODE / "split.csv"
DATA_PATH = REPO / "data/processed/task6/combined_nbasis12.csv"
ARCH_PATH = CODE / "specialists.json"
OUT = Path(os.environ.get("TASK6_CURVES_DIR", REPO / "notebooks/ryan/runs/task6/learning_curves")).resolve()
RUNS, RESULTS = OUT / "runs", OUT / "results"
PLAN, SUBSETS, PROTOCOL = OUT / "experiment_plan.csv", OUT / "training_subsets.csv", OUT / "protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path, text, *, freeze=False):
    if freeze and path.exists():
        if path.read_bytes() != text.encode():
            raise ValueError(f"Existing frozen file differs from this calculation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(text)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_data():
    fields = ["case_id", "set", "discharge", "NBASIS_MIN", "NBASIS_MAX", *engine.FEATURES, *TARGETS]
    raw = pd.read_csv(DATA_PATH, usecols=fields, float_precision="round_trip")
    split = pd.read_csv(SPLIT_PATH, usecols=["case_id", "split", "RMIN_LOC"], float_precision="round_trip")
    standard = raw.loc[raw["set"].eq("task4")].copy()
    assert set(standard["case_id"]) == set(split["case_id"])
    assert standard["discharge"].eq(128913).all() and standard[["NBASIS_MIN", "NBASIS_MAX"]].eq(12).all().all()
    full = standard.merge(split, on=["case_id", "RMIN_LOC"], validate="one_to_one")
    assert len(full) == len(split) and not full.duplicated(engine.FEATURES).any()
    assert full["split"].value_counts().to_dict() == {"train": 16586, "validation": 3554, "test": 3554}
    full["sample_id"] = full["case_id"]  # The engine uses this for row identities.
    return full.sort_values("case_id").set_index("case_id", drop=False)


def subset_plan(train):
    by_radius = train.groupby("RMIN_LOC", sort=True)
    counts = by_radius.size()
    assert len(counts) == 13
    quotas = {}
    for size in SIZES:
        exact = size * counts / len(train)
        quota = np.floor(exact).astype(int)
        extra = (exact - quota).sort_values(ascending=False, kind="stable").index[:size - quota.sum()]
        quota.loc[extra] += 1
        assert quota.sum() == size and (quota - exact).abs().max() < 1
        quotas[size] = quota
    quota_table = pd.DataFrame(quotas)
    assert (quota_table.diff(axis=1).iloc[:, 1:] > 0).all().all()
    frames = []
    for draw, seed in enumerate(DRAW_SEEDS, start=1):
        rng = np.random.default_rng(seed)
        for radius, frame in by_radius:
            ordered = rng.permutation(frame.sort_index()["case_id"].to_numpy())
            starts = np.repeat(SIZES, np.diff([0, *quota_table.loc[radius].to_list()]))
            frames.append(pd.DataFrame({"draw": draw, "draw_seed": seed, "case_id": ordered,
                                        "RMIN_LOC": radius, "first_size": starts}))
    membership = pd.concat(frames, ignore_index=True).sort_values(["draw", "case_id"])
    rows = []
    seen = {size: set() for size in SIZES}
    for draw in (1, 2, 3):
        m = membership.loc[membership["draw"].eq(draw)]
        assert set(m["case_id"]) == set(train["case_id"]) and m["case_id"].is_unique
        previous = set()
        for size in SIZES:
            group = m.loc[m["first_size"].le(size)]
            ids = tuple(group["case_id"])
            assert previous.issubset(ids) and len(ids) == size
            previous = set(ids)
            pd.testing.assert_series_equal(group.groupby("RMIN_LOC").size(), quotas[size], check_names=False)
            digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
            seen[size].add(digest)
            if size == SIZES[-1] and draw != 1:
                continue
            draw_name = "all" if size == SIZES[-1] else str(draw)
            for target in TARGETS:
                for width in (SCALES if size in WIDTH_SIZES else ("original",)):
                    fit_id = f"{target}_{size}_d{draw_name}_{width}"
                    rows.append({"fit_id": fit_id, "target": target, "size": size,
                                 "draw": draw_name, "width": width, "training_ids_sha256": digest})
    assert all(len(seen[n]) == (1 if n == SIZES[-1] else 3) for n in SIZES)
    plan = pd.DataFrame(rows)
    assert len(plan) == 81 and plan.fit_id.is_unique and plan.width.eq("original").sum() == 39
    return membership, plan


def prepare():
    full = load_data()
    membership, plan = subset_plan(full.loc[full["split"].eq("train")])
    settings = {
        "shot": 128913, "set": "task4", "nbasis": 12, "sizes": SIZES, "width_sizes": WIDTH_SIZES,
        "draw_seeds": DRAW_SEEDS, "network_seeds": MEMBER_SEEDS, "hidden_width_multipliers": SCALES,
        "fixed_split": {"train": 16586, "validation": 3554, "test": 3554},
        "stratification": "Permute cases within each radius and draw. Take proportional prefixes using largest remainders.",
        "width_comparison": "Change hidden width only. Depth, bottleneck, activations, preprocessing and budget are fixed.",
        "fit": "One fresh five-member ensemble per condition. Average predictions in physical units.",
        "selection": "Each member uses Task 5 validation early stopping; fit transformations on its training subset only.",
        "test": "Evaluate the predeclared original-width curves only, after every fit and width comparison is frozen.",
        "evaluation": "RMSE in gyro-Bohm units on the fitted subset, the fixed validation set, and the fixed historical test set.",
        "capacity_rule": "For each paired draw, flag a lower training RMSE and a higher validation RMSE at greater width.",
        "inputs_sha256": {"combined": sha(DATA_PATH), "split": sha(SPLIT_PATH), "specialists": sha(ARCH_PATH),
                          "engine": sha(TASK5 / "engine.py")},
    }
    write_text(SUBSETS, membership.to_csv(index=False, lineterminator="\n"), freeze=True)
    write_text(PLAN, plan.to_csv(index=False, lineterminator="\n"), freeze=True)
    write_text(PROTOCOL, json.dumps(settings, indent=2) + "\n", freeze=True)
    return plan


def cohort(full, job, membership):
    draw = 1 if job.draw == "all" else int(job.draw)
    subset = membership.loc[membership.draw.eq(draw) & membership.first_size.le(int(job["size"])), "case_id"]
    assert hashlib.sha256(("\n".join(subset) + "\n").encode()).hexdigest() == job.training_ids_sha256
    train = full.loc[subset]
    assert len(train) == int(job["size"]) and train["split"].eq("train").all()
    return train


def configuration(job):
    spec = json.loads(ARCH_PATH.read_text())["specialists"][job.target]
    cfg = dict(spec["training_config"])
    assert all(cfg[k] == v for k, v in spec.items() if k not in {"a5_run", "training_config"})
    cfg.update(case_id=job.fit_id, repeat=1, member_seeds=list(MEMBER_SEEDS),
               width=int(spec["width"] * SCALES[job.width]))
    cfg["parameters_per_member"] = engine.parameter_count(cfg)
    return cfg


def rmse(frame, ensemble, target):
    estimates = ensemble.predict(frame)["prediction"][:, 0]
    actual = frame[target].to_numpy(dtype=float)
    assert len(actual) == len(estimates) and np.isfinite(estimates).all()
    return float(np.sqrt(np.mean((actual - estimates) ** 2)))


def fit_one(fit_id):
    plan = pd.read_csv(PLAN, dtype={"draw": str}).set_index("fit_id", drop=False)
    job = plan.loc[fit_id]
    full = load_data()
    validation = full.loc[full["split"].eq("validation")]
    train = cohort(full, job, pd.read_csv(SUBSETS))
    cfg = configuration(job)
    directory = RUNS / fit_id
    with lock(directory / "fit.lock"):
        with (directory / "fit.log").open("a", buffering=1) as log, redirect_stdout(log):
            summary = engine.run_case(cfg, train, validation, directory, device="cpu", threads=1)
        ensemble = engine.load_ensemble(directory)
        result = {**job.to_dict(), "hidden_width": cfg["width"], "parameters_per_member": engine.parameter_count(cfg),
                  "n_members": len(ensemble.models), "train": rmse(train, ensemble, job.target),
                  "validation": rmse(validation, ensemble, job.target),
                  "best_epochs": [m["best_epoch"] for m in summary["member_summaries"]],
                  "checkpoint_sha256": sha(directory / "checkpoints.pt")}
        assert len(ensemble.models) == 5
        write_text(directory / "fit_result.json", json.dumps(result, indent=2) + "\n")


def run_job(fit_id):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_fit", fit_id],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "TASK6_CURVES_DIR": str(OUT)})
    if result.returncode:
        raise RuntimeError(f"{fit_id} failed\n{result.stdout[-4500:]}")
    return fit_id


def train_all(workers):
    with lock(OUT / "runner.lock"):
        plan = prepare()
        # Start time-consuming jobs first, keeping all choices independent of their scores.
        priority = plan.assign(cost=plan["size"] * plan["width"].map(SCALES).pow(2)
                               * plan["target"].map({"Q_e": 1., "Q_i": 2., "Gamma_e": 1.25}))
        ordered = priority.sort_values("cost", ascending=False, kind="stable")
        todo = [k for k in ordered.fit_id if not (RUNS / k / "fit_result.json").is_file()]
        print(f"{len(plan)-len(todo)}/{len(plan)} finished; {len(todo)} to fit", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_job, k) for k in todo]
            for i, f in enumerate(concurrent.futures.as_completed(futures), start=len(plan)-len(todo)+1):
                print(f"{i}/{len(plan)} complete: {f.result()}", flush=True)


def save_table(frame, name):
    write_text(RESULTS / name, frame.to_csv(index=False, lineterminator="\n"))


def paired_comparisons(fits):
    results = []
    for (target, size), frame in fits[fits["size"].isin(WIDTH_SIZES)].groupby(["target", "size"]):
        for lower, higher in combinations(("narrow", "original", "wide"), 2):
            a = frame.loc[frame.width.eq(lower)].set_index("draw").sort_index()
            b = frame.loc[frame.width.eq(higher)].set_index("draw").sort_index()
            assert a.index.equals(b.index) and a.training_ids_sha256.equals(b.training_ids_sha256)
            delta = b[["train", "validation"]] - a[["train", "validation"]]
            flag = (delta.train < 0) & (delta.validation > 0)
            results.append(dict(target=target, size=size, lower_width=lower, higher_width=higher,
                                lower_units=int(a.hidden_width.iloc[0]), higher_units=int(b.hidden_width.iloc[0]),
                                mean_delta_train=delta.train.mean(), mean_delta_validation=delta.validation.mean(),
                                draws_showing_overfitting=int(flag.sum()), paired_draws=len(delta),
                                mean_shows_overfitting=bool(delta.train.mean() < 0 and delta.validation.mean() > 0)))
    return pd.DataFrame(results)


def check_fits(plan):
    records = []
    for item in plan.itertuples(index=False):
        directory = RUNS / item.fit_id
        if not (directory / "fit_result.json").exists():
            raise RuntimeError(f"Waiting for training to complete: {item.fit_id}")
        result = json.loads((directory / "fit_result.json").read_text())
        assert all(result[k] == v for k, v in item._asdict().items())
        assert result["checkpoint_sha256"] == sha(directory / "checkpoints.pt")
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        assert config["status"] == summary["status"] == "complete"
        assert result["n_members"] == 5 and summary["runtime"]["device"] == "cpu"
        from_archive = pd.read_csv(directory / "metrics.csv")
        ensemble_val = from_archive.loc[from_archive["member_seed"].eq("ensemble")
                                       & from_archive["split"].eq("validation") & from_archive["target"].eq(item.target)]
        assert len(ensemble_val) == 1 and np.isclose(result["validation"], ensemble_val.rmse.item(), rtol=1e-9)
        records.append(result)
    fits = pd.DataFrame(records)
    assert len(fits) == 81 and not fits.duplicated(["target", "size", "draw", "width"]).any()
    assert np.isfinite(fits[["train", "validation"]]).all().all()
    return fits


def evaluate():
    engine.torch.set_num_threads(1)
    with lock(OUT / "runner.lock"):
        plan = prepare()
        fits = check_fits(plan)
        pairs = paired_comparisons(fits)
        capacity = (fits.loc[fits["size"].isin(WIDTH_SIZES)]
                    .groupby(["target", "size", "width"], as_index=False)
                    .agg(hidden_width=("hidden_width", "first"), parameters=("parameters_per_member", "first"),
                         draws=("fit_id", "size"), train_rmse=("train", "mean"), validation_rmse=("validation", "mean")))
        winners = capacity.loc[capacity.groupby(["target", "size"]).validation_rmse.idxmin()]
        save_table(pairs, "paired_width_comparisons.csv")
        save_table(capacity, "width_summary.csv")
        # Everything above this line uses training and validation scores only.
        freeze = {
            "test_cases": 3554, "training_counts": list(SIZES), "test_policy": "All 39 predeclared original-width fits.",
            "checkpoint_sha256": dict(zip(fits.fit_id, fits.checkpoint_sha256)),
            "paired_comparison_sha256": sha(RESULTS / "paired_width_comparisons.csv"),
            "lowest_mean_validation_rmse_by_target_and_size": winners.to_dict(orient="records"),
        }
        write_text(RESULTS / "validation_freeze.json", json.dumps(freeze, indent=2) + "\n", freeze=True)
        full = load_data()
        test = full.loc[full["split"].eq("test")]
        fits["test"] = np.nan
        for i, job in fits.loc[fits.width.eq("original")].iterrows():
            model = engine.load_ensemble(RUNS / job.fit_id)
            fits.loc[i, "test"] = rmse(test, model, job.target)
        assert np.isfinite(fits.loc[fits.width.eq("original"), "test"]).all()
        assert fits.loc[~fits.width.eq("original"), "test"].isna().all()
        fits["best_epochs"] = fits["best_epochs"].map(json.dumps)
        save_table(fits, "per_fit_rmse.csv")
        long = fits.loc[fits.width.eq("original")].melt(
            id_vars=["target", "size", "draw"], value_vars=["train", "validation", "test"],
            var_name="split", value_name="rmse",
        )
        curves = long.groupby(["target", "size", "split"], as_index=False).agg(
            mean_rmse=("rmse", "mean"), min_rmse=("rmse", "min"), max_rmse=("rmse", "max"), draws=("rmse", "size"))
        assert len(curves) == 45 and (curves.draws == np.where(curves["size"] == SIZES[-1], 1, 3)).all()
        save_table(curves, "learning_curve_summary.csv")
        print("Verified 81 ensembles, 405 newly trained networks, 39 held-out evaluations and 27 paired width comparisons.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["prepare", "train", "summarize", "_fit"])
    p.add_argument("fit_id", nargs="?")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if OUT.is_relative_to(CODE):
        p.error("Write runs under notebooks/ryan/runs/ (TASK6_CURVES_DIR), not the source folder.")
    if args.action == "prepare":
        print(f"Prepared {len(prepare())} ensemble fits.")
    elif args.action == "train":
        if args.workers < 1:
            p.error("--workers must be positive")
        train_all(args.workers)
    elif args.action == "summarize":
        evaluate()
    else:
        fit_one(args.fit_id)


if __name__ == "__main__":
    main()
