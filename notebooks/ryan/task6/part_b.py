"""Task 6 Part B: more held-out cohorts, error bars, scaling, input edges, and network capacity.

Reads the 81 learning-curve ensembles from ``learning_curves.py`` (none is changed) and adds:

``capacity``  train full-data ensembles at 4x hidden width plus an original-width rerun control
              (same 16,586 cases, seeds, and recipe as the learning curves);
``score``     score the 39 original-width ensembles on validation, test, 1,426 varied cases and
              12 profile cases at the intermediate radii, and 385 cases also run at other TGLF
              resolutions. It checks the archived test and validation RMSE first. Error bars use
              the Task 5 method, calibrated for 90% on each fit's validation predictions;
``analyze``   write the tables that answer the Part B questions (no training).

Cohorts: Team B's 1,186 varied runs at the 12 intermediate radii (contributor Ketan, from the
Task 6 course table) plus Ryan's 240 seed-180501 runs from Task 5, after the same negative-heat
rule. All errors compare with numerical TGLF for shot 128913. From the repository root::

    python notebooks/ryan/task6/part_b.py capacity --workers 6
    python notebooks/ryan/task6/part_b.py score
    python notebooks/ryan/task6/part_b.py analyze

Output goes to ``--output`` (default ``notebooks/ryan/runs/task6/part_b``).
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import learning_curves as lc  # noqa: E402  (also sets single-thread BLAS and imports the Task 5 engine)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

engine = lc.engine
OUT = Path(os.environ.get("TASK6_PART_B_DIR", lc.REPO / "notebooks/ryan/runs/task6/part_b")).resolve()
RUNS, RESULTS = OUT / "runs", OUT / "results"
RYAN_VARIED = lc.REPO / "notebooks/ryan/task5/varied_test.csv"
TEAM_B_RAW = lc.REPO / "data/Task6_DIIID_Surrogate/team_b_d3d_database.csv"
SCALES = {"original_rerun": 1.0, "xwide": 4.0}
FITS = [f"{target}_16586_dall_{label}" for label in SCALES for target in lc.TARGETS]
COVERAGE = 0.90
F = list(engine.FEATURES)
FACTORS = {"f_Te": "RLTS_1", "f_Ti": "RLTS_2", "f_n": "RLNS_1", "f_tau": "TAUS_2", "f_nu": "XNUE"}
INTERMEDIATE = [round(0.225 + 0.05 * i, 3) for i in range(12)]
TARGETS = list(lc.TARGETS)
SIZES = list(lc.SIZES)
FULL = SIZES[-1]
OUTER = 0.70  # r/a at and beyond which the outer region starts.
BOOT, BOOT_SEED = 10_000, 180630
FACTOR_RANGES = {"f_Te": (0.6, 1.4), "f_Ti": (0.6, 1.4), "f_n": (0.6, 1.4), "f_tau": (0.75, 1.25), "f_nu": (0.5, 2.0)}


def read(path, **kw):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False, **kw)


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


# ---------------------------------------------------------------- capacity
def configuration(target, scale, fit_id):
    spec = json.loads(lc.ARCH_PATH.read_text())["specialists"][target]
    cfg = dict(spec["training_config"])
    assert all(cfg[k] == v for k, v in spec.items() if k not in {"a5_run", "training_config"})
    cfg.update(case_id=fit_id, repeat=1, member_seeds=list(lc.MEMBER_SEEDS), width=int(spec["width"] * scale))
    cfg["parameters_per_member"] = engine.parameter_count(cfg)
    return cfg


def fit_one(fit_id):
    target, label = fit_id.split("_16586_dall_")
    size, draw = 16586, "all"
    assert target in lc.TARGETS and label in SCALES
    plan = lc.pd.read_csv(lc.PLAN, dtype={"draw": str}).set_index("fit_id", drop=False)
    reference = plan.loc[f"{target}_{size}_d{draw}_original"]  # Same training IDs as every full-size A6 fit.
    full = lc.load_data()
    validation = full.loc[full["split"].eq("validation")]
    train = lc.cohort(full, reference, lc.pd.read_csv(lc.SUBSETS))
    cfg = configuration(target, SCALES[label], fit_id)
    directory = RUNS / fit_id
    with lc.lock(directory / "fit.lock"):
        with (directory / "fit.log").open("a", buffering=1) as log, redirect_stdout(log):
            summary = engine.run_case(cfg, train, validation, directory, device="cpu", threads=1)
        ensemble = engine.load_ensemble(directory)
        result = {"fit_id": fit_id, "target": target, "size": size, "draw": draw, "width": label,
                  "training_ids_sha256": reference.training_ids_sha256, "hidden_width": cfg["width"],
                  "parameters_per_member": engine.parameter_count(cfg), "n_members": len(ensemble.models),
                  "train": lc.rmse(train, ensemble, target), "validation": lc.rmse(validation, ensemble, target),
                  "best_epochs": [m["best_epoch"] for m in summary["member_summaries"]],
                  "hit_epoch_limit": [m["hit_epoch_limit"] for m in summary["member_summaries"]],
                  "fit_seconds": summary["fit_seconds"],
                  "checkpoint_sha256": lc.sha(directory / "checkpoints.pt")}
        assert len(ensemble.models) == 5
        lc.write_text(directory / "fit_result.json", json.dumps(result, indent=2) + "\n")


def run_job(fit_id):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_fit", fit_id],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "TASK6_PART_B_DIR": str(OUT), "TASK6_CURVES_DIR": str(lc.OUT)})
    if result.returncode:
        raise RuntimeError(f"{fit_id} failed\n{result.stdout[-4500:]}")
    return fit_id


def train_capacity(workers):
    todo = [k for k in FITS if not (RUNS / k / "fit_result.json").is_file()]
    print(f"{len(FITS) - len(todo)}/{len(FITS)} finished; {len(todo)} to fit", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run_job, k) for k in todo]):
            print(f"complete: {future.result()}", flush=True)


# ---------------------------------------------------------------- score
def load_cases():
    full = lc.load_data().reset_index(drop=True)
    cleaned = read(lc.DATA_PATH)
    nominal = cleaned.loc[cleaned["set"].eq("answers_check")].set_index("RMIN_LOC")[F[:-1]]
    assert nominal.index.is_unique and len(nominal) == 25

    standard = full.loc[full["split"].isin(["validation", "test"]), ["case_id", "split", *F, *lc.TARGETS]]
    standard = standard.rename(columns={"split": "cohort"}).assign(source="task4_fixed_split")

    team = cleaned.loc[cleaned["set"].eq("test_varied"), ["case_id", "contributor", *F, *lc.TARGETS]]
    assert team["contributor"].eq("Ketan").all() and len(team) == 1186
    team = team.drop(columns="contributor").assign(cohort="intermediate_varied", source="team_b_ketan")

    ryan_raw = read(RYAN_VARIED)
    assert len(ryan_raw) == 240 and ryan_raw[["NBASIS_MIN", "NBASIS_MAX"]].eq(12).all().all()
    bad = ryan_raw[["Q_e", "Q_i"]].lt(0).any(axis=1) | ~np.isfinite(ryan_raw[F + list(lc.TARGETS)]).all(axis=1)
    exclusions = ryan_raw.loc[bad, ["sample_id", *F, *lc.TARGETS]].assign(reason="negative_heat_flux")
    ryan = ryan_raw.loc[~bad, ["sample_id", *F, *lc.TARGETS]].rename(columns={"sample_id": "case_id"})
    ryan = ryan.assign(case_id="ryan180501_" + ryan["case_id"].astype(str),
                       cohort="intermediate_varied", source="ryan_seed180501")

    profile = cleaned.loc[cleaned["set"].eq("answers_check") & cleaned["RMIN_LOC"].isin(INTERMEDIATE),
                          ["case_id", *F, *lc.TARGETS]].assign(cohort="intermediate_profile", source="profile_answers")
    assert len(profile) == 12

    nb = cleaned.loc[cleaned["set"].eq("nbasis_check"), ["case_id", *F, *lc.TARGETS]]
    nb = nb.assign(cohort="nbasis", source="team_b_ketan_nbasis12")

    cases = pd.concat([standard, team, ryan, profile, nb], ignore_index=True)
    assert cases["case_id"].is_unique
    assert np.isfinite(cases[F + list(lc.TARGETS)]).all().all()
    assert set(cases.loc[cases.cohort.str.startswith("intermediate"), "RMIN_LOC"]) == set(INTERMEDIATE)
    # None of the extra cohorts repeats a training input vector.
    train_keys = set(map(tuple, full.loc[full.split.eq("train"), F].to_numpy()))
    extra = cases.loc[~cases.cohort.isin(["validation", "test"]), F].to_numpy()
    assert not any(tuple(row) in train_keys for row in extra)

    # Sampling factors relative to the unperturbed profile at the same radius.
    local = nominal.reindex(cases["RMIN_LOC"]).to_numpy()
    for i, (factor, column) in enumerate(FACTORS.items()):
        cases[factor] = cases[column].to_numpy() / local[:, i]
    stored = pd.concat([cleaned[["case_id", *FACTORS]], ryan_raw.assign(
        case_id="ryan180501_" + ryan_raw["sample_id"].astype(str))[["case_id", *FACTORS]]])
    check = cases[["case_id", *FACTORS]].merge(stored, on="case_id", suffixes=("", "_stored"))
    factor_error = max(float((check[f] / check[f + "_stored"] - 1).abs().max()) for f in FACTORS
                       if check[f + "_stored"].notna().any())
    assert factor_error < 1e-4, factor_error  # Inputs carry six significant digits.
    return full, cases, exclusions, factor_error


def nbasis_pairs(cases):
    raw = read(TEAM_B_RAW)
    raw = raw.loc[raw["set"].isin(["nbasis_check", "nbasis_default"])].copy()
    raw["basis"] = raw["NBASIS_MIN"].astype(int).astype(str) + "/" + raw["NBASIS_MAX"].astype(int).astype(str)
    wide = raw.pivot_table(index=F, columns="basis", values=list(lc.TARGETS), aggfunc="first")
    wide.columns = [f"{t}_nb{b.replace('/', '_')}" for t, b in wide.columns]
    labeled = cases.loc[cases.cohort.eq("nbasis"), ["case_id", *F, *lc.TARGETS]]
    pairs = labeled.merge(wide.reset_index(), on=F, how="left", validate="one_to_one")
    for t in lc.TARGETS:
        np.testing.assert_array_equal(pairs[t].to_numpy(), pairs[f"{t}_nb12_12"].to_numpy())
    assert len(pairs) == 385
    return pairs


def conformal_multiplier(residual_ratio):
    n = residual_ratio.size
    rank = math.ceil((n + 1) * COVERAGE)
    return float(np.partition(residual_ratio, rank - 1)[rank - 1]), rank


def score(full, cases):
    fits = read(lc.RESULTS / "per_fit_rmse.csv", dtype={"draw": str})
    fits = fits.loc[fits["width"].eq("original")]
    assert len(fits) == 39
    frame = cases.assign(sample_id=cases["case_id"])
    rows, multipliers, checks = [], [], []
    for job in fits.itertuples(index=False):
        directory = lc.RUNS / job.fit_id
        assert lc.sha(directory / "checkpoints.pt") == job.checkpoint_sha256
        ensemble = engine.load_ensemble(directory)
        members = ensemble.predict(frame)["members"][:, :, 0]
        mean, spread = members.mean(axis=0), members.std(axis=0, ddof=1)
        floor = max(1e-8, 0.01 * float(ensemble.raw_training_std[0]))
        scale = np.maximum(spread, floor)
        truth = frame[job.target].to_numpy(dtype=float)
        part = pd.DataFrame({"fit_id": job.fit_id, "target": job.target, "size": job.size, "draw": job.draw,
                             "cohort": frame["cohort"].to_numpy(), "case_id": frame["case_id"].to_numpy(),
                             "truth": truth, "prediction": mean, "spread": spread, "scale": scale})
        for split in ("validation", "test"):  # Must reproduce the frozen A6 scores exactly.
            s = part.loc[part.cohort.eq(split)]
            recomputed = float(np.sqrt(np.mean((s.truth - s.prediction) ** 2)))
            archived = float(getattr(job, split))
            checks.append(dict(fit_id=job.fit_id, split=split, archived=archived, recomputed=recomputed))
            assert np.isclose(recomputed, archived, rtol=1e-9, atol=0), (job.fit_id, split, recomputed, archived)
        val = part.loc[part.cohort.eq("validation")]
        q, rank = conformal_multiplier(((val.truth - val.prediction).abs() / val.scale).to_numpy())
        part["half_width"] = q * part["scale"]
        multipliers.append(dict(fit_id=job.fit_id, target=job.target, size=job.size, draw=job.draw,
                                calibration_n=len(val), rank=rank, multiplier=q, floor=floor))
        rows.append(part)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(multipliers), pd.DataFrame(checks)


def score_all():
    lc.engine.torch.set_num_threads(4)
    full, cases, exclusions, factor_error = load_cases()
    pairs = nbasis_pairs(cases)
    predictions, multipliers, checks = score(full, cases)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases.to_csv(RESULTS / "cases.csv.gz", index=False)
    predictions.to_csv(RESULTS / "predictions.csv.gz", index=False, float_format="%.10g")
    multipliers.to_csv(RESULTS / "multipliers.csv", index=False)
    exclusions.to_csv(RESULTS / "exclusions.csv", index=False)
    pairs.to_csv(RESULTS / "nbasis_pairs.csv", index=False)
    checks.to_csv(RESULTS / "archived_score_checks.csv", index=False)
    manifest = {
        "script": "notebooks/ryan/task6/part_b.py score",
        "fits_scored": int(predictions.fit_id.nunique()),
        "cohort_counts": cases.groupby(["cohort", "source"]).size().rename("n").reset_index().to_dict("records"),
        "excluded": {"ryan_seed180501_negative_heat": int(len(exclusions))},
        "max_relative_factor_difference_vs_stored": factor_error,
        "max_abs_difference_vs_archived_rmse": float((checks.recomputed - checks.archived).abs().max()),
        "interval": f"mean +/- q * max(member SD, 1% of training SD); q = {COVERAGE:.0%} split-conformal on validation",
        "inputs_sha256": {"combined": lc.sha(lc.DATA_PATH), "split": lc.sha(lc.SPLIT_PATH),
                          "per_fit_rmse": lc.sha(lc.RESULTS / "per_fit_rmse.csv"),
                          "ryan_varied": lc.sha(RYAN_VARIED), "team_b_raw": lc.sha(TEAM_B_RAW)},
        "runtime": {"python": platform.python_version(), "torch": engine.torch.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
    }
    (RESULTS / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("fits_scored", "cohort_counts", "excluded",
                                              "max_abs_difference_vs_archived_rmse")}, indent=1))



# ---------------------------------------------------------------- analyze
def load():
    pred = read(RESULTS / "predictions.csv.gz", dtype={"draw": str})
    cases = read(RESULTS / "cases.csv.gz")
    pred = pred.merge(cases[["case_id", "source", "RMIN_LOC", *FACTOR_RANGES]], on="case_id", validate="many_to_one")
    pred["error"] = pred.prediction - pred.truth
    pred["inside"] = pred.error.abs() <= pred.half_width
    pred["region"] = np.where(pred.RMIN_LOC >= OUTER - 1e-9, "outer", "inner")
    fits = read(lc.RESULTS / "per_fit_rmse.csv", dtype={"draw": str})
    return pred, cases, fits


def cohort_scores(pred):
    def summary(g):
        rel = (g.error.abs() / g.truth.abs()).replace(np.inf, np.nan)
        return pd.Series({"n": len(g), "rmse": rms(g.error), "coverage": g.inside.mean(),
                          "mean_half_width": g.half_width.mean(), "median_relative_error": rel.median(),
                          "share_within_5pct": (rel < 0.05).mean()})
    return pred.groupby(["fit_id", "target", "size", "draw", "cohort"]).apply(summary, include_groups=False).reset_index()


def curves(scores, fits):
    train = fits.loc[fits.width.eq("original"), ["fit_id", "target", "size", "draw", "train"]]
    train = train.rename(columns={"train": "rmse"}).assign(cohort="train")
    long = pd.concat([scores[["fit_id", "target", "size", "draw", "cohort", "rmse"]], train], ignore_index=True)
    return long.groupby(["target", "size", "cohort"], as_index=False).agg(
        mean_rmse=("rmse", "mean"), min_rmse=("rmse", "min"), max_rmse=("rmse", "max"), draws=("rmse", "size"))


def power_fit(sizes, values):
    x, y = np.log(np.asarray(sizes, float)), np.log(np.asarray(values, float))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return -slope, float(np.exp(intercept)), float(r2)


def scaling(curve_table, region_table):
    rows = []
    windows = {"all sizes (1k-16.6k)": SIZES, "4k-16.6k": SIZES[2:]}
    sources = [(curve_table.rename(columns={"mean_rmse": "rmse"}), "cohort", ["validation", "test", "intermediate_varied"]),
               (region_table, "group", sorted(region_table.group.unique()))]
    for table, key, groups in sources:
        for target in TARGETS:
            for group in groups:
                s = table.loc[table.target.eq(target) & table[key].eq(group)].set_index("size").rmse
                for window, sizes in windows.items():
                    b, a, r2 = power_fit(sizes, s.loc[sizes])
                    multiple = 2 ** (1 / b) if b > 0 else np.inf
                    rows.append(dict(target=target, curve=group, window=window, exponent=b, prefactor=a, r2=r2,
                                     fitted_full_rmse=a * FULL ** -b, observed_full_rmse=s.loc[FULL],
                                     multiple_to_halve=multiple,
                                     runs_to_halve=FULL * multiple if np.isfinite(multiple) else np.inf))
    return pd.DataFrame(rows)


def region_curves(pred):
    groups = {("test", "inner"): "training radii, r/a < 0.70", ("test", "outer"): "training radii, r/a >= 0.70",
              ("intermediate_varied", "inner"): "intermediate radii, r/a < 0.70",
              ("intermediate_varied", "outer"): "intermediate radii, r/a >= 0.70"}
    p = pred.loc[pred.cohort.isin(["test", "intermediate_varied"])]
    per_fit = p.groupby(["target", "size", "draw", "cohort", "region"]).error.apply(rms).rename("rmse").reset_index()
    per_fit["group"] = [groups[(c, r)] for c, r in zip(per_fit.cohort, per_fit.region)]
    return per_fit.groupby(["target", "size", "group"], as_index=False).agg(
        rmse=("rmse", "mean"), min_rmse=("rmse", "min"), max_rmse=("rmse", "max"))


def by_radius(pred):
    p = pred.loc[pred["size"].eq(FULL)]
    out = []
    for (target, cohort, r), g in p.groupby(["target", "cohort", "RMIN_LOC"]):
        rel = (g.error.abs() / g.truth.abs()).replace(np.inf, np.nan)
        out.append(dict(target=target, cohort=cohort, r_over_a=r, n=len(g), rmse=rms(g.error),
                        rms_flux=rms(g.truth), mean_half_width=g.half_width.mean(), coverage=g.inside.mean(),
                        median_relative_error=rel.median(), mean_signed_error=g.error.mean()))
    return pd.DataFrame(out)


def radius_scale():
    """RMS training flux at each of the 13 training radii; used to put errors on a common scale."""
    full = lc.load_data()
    train = full.loc[full.split.eq("train")]
    return {t: train.groupby("RMIN_LOC")[t].apply(rms) for t in TARGETS}


def by_input(pred, scale):
    p = pred.loc[pred["size"].eq(FULL) & pred.cohort.eq("test")].copy()
    p["s"] = [scale[t].loc[r] for t, r in zip(p.target, p.RMIN_LOC)]
    p["scaled_error"], p["scaled_half_width"] = p.error / p.s, p.half_width / p.s
    rows = []
    for factor, (lo, hi) in FACTOR_RANGES.items():
        log = factor == "f_nu"
        edges = np.geomspace(lo, hi, 6) if log else np.linspace(lo, hi, 6)
        p["bin"] = np.clip(np.digitize(p[factor], edges[1:-1]), 0, 4)
        for region, sub in [("all radii", p), ("r/a < 0.70", p.loc[p.region.eq("inner")]),
                            ("r/a >= 0.70", p.loc[p.region.eq("outer")])]:
            for (target, b), g in sub.groupby(["target", "bin"]):
                rows.append(dict(factor=factor, region=region, target=target, bin=int(b) + 1,
                                 lower=edges[b], upper=edges[b + 1],
                                 center=float(np.sqrt(edges[b] * edges[b + 1]) if log else (edges[b] + edges[b + 1]) / 2),
                                 n=len(g), rmse=rms(g.error), scaled_rmse=rms(g.scaled_error),
                                 scaled_mean_half_width=g.scaled_half_width.mean(), coverage=g.inside.mean()))
    table = pd.DataFrame(rows)
    edge = table.assign(position=np.where(table.bin.isin([1, 5]), "edge bins", "middle bins"))
    edge = edge.groupby(["factor", "region", "target", "position"]).apply(
        lambda g: pd.Series({"n": g.n.sum(), "scaled_rmse": np.sqrt(np.average(g.scaled_rmse ** 2, weights=g.n)),
                             "coverage": np.average(g.coverage, weights=g.n)}), include_groups=False).reset_index()
    return table, edge


def tglf_resolution(pred):
    pairs = read(RESULTS / "nbasis_pairs.csv")
    surrogate = pred.loc[pred["size"].eq(FULL) & pred.cohort.eq("nbasis"), ["case_id", "target", "prediction"]]
    rows, by_r = [], []
    for target in TARGETS:
        s = pairs.merge(surrogate.loc[surrogate.target.eq(target)], on="case_id", validate="one_to_one")
        reference = s[f"{target}_nb12_12"]
        comparisons = {"surrogate (full-data A6 model)": s.prediction,
                       "TGLF, NBASIS 16/16": s[f"{target}_nb16_16"], "TGLF, NBASIS 2/4": s[f"{target}_nb2_4"]}
        for name, other in comparisons.items():
            diff = other - reference
            rel = (diff.abs() / reference.abs()).replace(np.inf, np.nan)
            rows.append(dict(target=target, comparison=name, n=int(diff.notna().sum()), rmse=rms(diff.dropna()),
                             median_relative_difference=rel.median(), share_within_5pct=(rel < 0.05).mean()))
            for r, g in pd.DataFrame({"r": s.RMIN_LOC, "d": diff, "rel": rel}).groupby("r"):
                by_r.append(dict(target=target, comparison=name, r_over_a=r, n=len(g), rmse=rms(g.d.dropna()),
                                 median_relative_difference=g.rel.median()))
    return pd.DataFrame(rows), pd.DataFrame(by_r)


def validation_predictions(fit_dirs):
    out = {}
    for fit_id, directory in fit_dirs.items():
        p = read(directory / "predictions.csv", dtype={"member_seed": str})
        p = p.loc[p.member_seed.eq("ensemble") & p.split.eq("validation")].sort_values("sample_id")
        out[fit_id] = p
    first = next(iter(out.values()))
    for p in out.values():
        assert (p.sample_id.to_numpy() == first.sample_id.to_numpy()).all()
    return out, first.r_over_a.to_numpy()


def bootstrap_index(strata, n_boot, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for start in range(0, n_boot, 500):
        k = min(500, n_boot - start)
        yield np.concatenate([g[rng.integers(0, len(g), size=(k, len(g)))] for g in groups], axis=1)


def paired_delta(err2_low, err2_high, strata):
    """Mean over draws of RMSE(high) - RMSE(low) on validation, with radius-stratified resampling."""
    point = np.mean([np.sqrt(h.mean()) - np.sqrt(l.mean()) for l, h in zip(err2_low, err2_high)])
    deltas = []
    for idx in bootstrap_index(strata, BOOT, BOOT_SEED):
        deltas.append(np.mean([np.sqrt(h[idx].mean(axis=1)) - np.sqrt(l[idx].mean(axis=1))
                               for l, h in zip(err2_low, err2_high)], axis=0))
    deltas = np.concatenate(deltas)
    lo, hi = np.percentile(deltas, [5, 95])
    return point, lo, hi


def capacity(fits):
    extra = {json.loads((d / "fit_result.json").read_text())["fit_id"]: d
             for d in sorted(RUNS.iterdir()) if (d / "fit_result.json").is_file()}
    results = [json.loads((d / "fit_result.json").read_text()) for d in extra.values()]
    rerun = {r["target"]: r for r in results if r["width"] == "original_rerun"}
    control = []
    for target, r in rerun.items():
        saved = fits.loc[fits.fit_id.eq(f"{target}_{FULL}_dall_original")].iloc[0]
        control.append(dict(target=target, saved_validation=saved.validation, rerun_validation=r["validation"],
                            saved_train=saved.train, rerun_train=r["train"],
                            identical_validation=bool(saved.validation == r["validation"])))
    dirs = {row.fit_id: lc.RUNS / row.fit_id for row in fits.itertuples()}
    dirs.update({k: v for k, v in extra.items() if k.endswith("xwide")})
    preds, strata = validation_predictions(dirs)
    err2 = {k: ((p.prediction - p.truth) ** 2).to_numpy() for k, p in preds.items()}

    table = fits[["fit_id", "target", "size", "draw", "width", "hidden_width", "parameters_per_member", "train",
                  "validation", "best_epochs"]].copy()
    table["members_kept_warmup_weights"] = table.best_epochs.map(lambda s: sum(e == 0 for e in json.loads(s)))
    for r in results:
        if r["width"] == "xwide":
            table.loc[len(table)] = dict(fit_id=r["fit_id"], target=r["target"], size=FULL, draw="all", width="xwide",
                                         hidden_width=r["hidden_width"], parameters_per_member=r["parameters_per_member"],
                                         train=r["train"], validation=r["validation"], best_epochs=json.dumps(r["best_epochs"]),
                                         members_kept_warmup_weights=sum(e == 0 for e in r["best_epochs"]))
    summary = table.groupby(["target", "size", "width"], as_index=False).agg(
        hidden_width=("hidden_width", "first"), parameters_per_member=("parameters_per_member", "first"),
        draws=("fit_id", "size"), train_rmse=("train", "mean"), validation_rmse=("validation", "mean"),
        members_kept_warmup_weights=("members_kept_warmup_weights", "sum"))

    comparisons = []
    for target in TARGETS:
        for size in lc.WIDTH_SIZES:
            widths = ["narrow", "original", "wide"] + (["xwide"] if size == FULL else [])
            draws = ["all"] if size == FULL else ["1", "2", "3"]
            for low, high in zip(widths[:-1], widths[1:]):
                ids = lambda w: [f"{target}_{size}_d{d}_{w}" for d in draws]
                point, lo, hi = paired_delta([err2[k] for k in ids(low)], [err2[k] for k in ids(high)], strata)
                comparisons.append(dict(target=target, size=size, lower=low, higher=high, draws=len(draws),
                                        delta_validation_rmse=point, ci90_low=lo, ci90_high=hi))
    return summary, pd.DataFrame(comparisons), pd.DataFrame(control)


def save(frame, name):
    frame.to_csv(RESULTS / name, index=False, float_format="%.6g")


def analyze():
    pred, cases, fits = load()
    scores = cohort_scores(pred)
    curve = curves(scores, fits)
    regions = region_curves(pred)
    fits_table = scaling(curve, regions)
    radius = by_radius(pred)
    inputs, edges = by_input(pred, radius_scale())
    resolution, resolution_r = tglf_resolution(pred)
    width, deltas, control = capacity(fits)
    for frame, name in [(scores, "cohort_scores.csv"), (curve, "learning_curves_by_cohort.csv"),
                        (regions, "learning_curves_by_region.csv"), (fits_table, "scaling_fits.csv"),
                        (radius, "error_by_radius.csv"), (inputs, "error_by_input.csv"),
                        (edges, "edge_vs_middle.csv"), (resolution, "tglf_resolution.csv"),
                        (resolution_r, "tglf_resolution_by_radius.csv"), (width, "width_capacity.csv"),
                        (deltas, "width_paired_deltas.csv"), (control, "rerun_control.csv")]:
        save(frame, name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["capacity", "score", "analyze", "_fit"])
    p.add_argument("fit_id", nargs="?")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if OUT.is_relative_to(CODE):
        p.error("Write output under notebooks/ryan/runs/ (TASK6_PART_B_DIR), not the source folder.")
    if args.action == "_fit":
        fit_one(args.fit_id)
    elif args.action == "capacity":
        train_capacity(args.workers)
    elif args.action == "score":
        score_all()
    else:
        analyze()


if __name__ == "__main__":
    main()
