"""Task 5 architecture search: one shared ensemble versus a separate ensemble per flux.

Stages: ``controls`` and ``screen`` train the 114 one-change configurations in
``search_matrix.csv``; ``interactions`` combines each output's best changes; ``confirmation``
retrains the finalists with three fresh seed groups; ``final`` freezes the choice on
validation and only then scores the test set and the 12 profile cases. ``all`` runs every
stage, and completed compatible runs are reused. Every ensemble has five networks.

``--root`` (default ``notebooks/ryan/runs/task5/search``, ignored by Git) receives the split,
checkpoints, and results. The code refuses to write into this source folder.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2] / "notebooks/ryan/runs/task5/search"
TARGETS = ["Q_e", "Q_i", "Gamma_e"]
MODES = ["joint", *TARGETS]
PREFIX = {"joint": "joint", "Q_e": "qe", "Q_i": "qi", "Gamma_e": "gamma"}
INT_FIELDS = {"width", "depth", "bottleneck_width", "ensemble_size", "warmup_epochs",
              "main_epochs", "batch_size", "patience_epochs", "eval_every_epochs",
              "parameters_per_member"}
FLOAT_FIELDS = {"warmup_lr", "main_lr", "weight_decay", "gradient_clip_norm"}
META_FIELDS = {"case_id", "baseline_id", "phase", "family", "changed_factor",
               "baseline_value", "candidate_value", "parameters_per_member", "status",
               "source_case_id", "parent_cases", "repeat"}
REPEATS = {1: "142|143|144|145|146", 2: "242|243|244|245|246", 3: "342|343|344|345|346"}


def _root(root=None):
    root = Path(root or ROOT).resolve()
    matrix = root / "experiment_matrix.csv"
    if root != CODE and not matrix.exists():
        # A new experiment directory starts from the committed initial matrix.
        root.mkdir(parents=True, exist_ok=True)
        matrix.write_bytes((CODE / "search_matrix.csv").read_bytes())
    return root


def _guard(path):
    """Keep run output out of the source folder."""
    if Path(path).resolve().is_relative_to(CODE):
        raise PermissionError(f"{path} is inside the source folder. Use --root under notebooks/ryan/runs/.")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def _write_json(path, value):
    _guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path, frame):
    _guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_matrix(path):
    frame = pd.read_csv(path, keep_default_na=False)
    for name in INT_FIELDS & set(frame):
        frame[name] = pd.to_numeric(frame[name]).astype(int)
    for name in FLOAT_FIELDS & set(frame):
        frame[name] = pd.to_numeric(frame[name]).astype(float)
    return frame


def _config_identity(row):
    return json.dumps({k: v for k, v in row.items() if k not in META_FIELDS}, sort_keys=True)


def parameter_count(row):
    widths = [6] + [int(row["width"])] * int(row["depth"]) + [int(row["bottleneck_width"])]
    widths += [3 if row["output_mode"] == "joint" else 1]
    return sum((a + 1) * b for a, b in zip(widths, widths[1:]))


def validate_matrix(root=None):
    """Validate all declared one-factor changes and return the initial matrix."""
    root = _root(root)
    matrix = read_matrix(root / "experiment_matrix.csv")
    if matrix.case_id.duplicated().any():
        raise ValueError("Duplicate case IDs")
    expected = {"joint": 34, "Q_e": 27, "Q_i": 27, "Gamma_e": 26}
    if matrix.output_mode.value_counts().to_dict() != expected:
        raise ValueError("The approved initial matrix must contain 114 configurations")
    rows = {r["case_id"]: r for r in matrix.to_dict("records")}
    seen = set()
    fields = set(matrix) - META_FIELDS
    for row in rows.values():
        if not re.fullmatch(r"[A-Za-z0-9_]+", row["case_id"]):
            raise ValueError("Case IDs must be safe directory names")
        identity = _config_identity(row)
        if identity in seen:
            raise ValueError(f"Duplicate configuration: {row['case_id']}")
        seen.add(identity)
        if int(row["ensemble_size"]) != 5 or row["member_seeds"] != "42|43|44|45|46":
            raise ValueError("Initial configurations require the five reference seeds")
        if parameter_count(row) != int(row["parameters_per_member"]):
            raise ValueError(f"Incorrect parameter count: {row['case_id']}")
        if row["phase"] == "screening":
            base = rows[row["baseline_id"]]
            changed = {k for k in fields if base[k] != row[k]}
            if changed != {row["changed_factor"]}:
                raise ValueError(f"Undeclared changes: {row['case_id']}: {changed}")
            name = row["changed_factor"]
            # Values in the descriptive columns are strings, including numbers.
            if str(base[name]) != str(row["baseline_value"]) or str(row[name]) != str(row["candidate_value"]):
                raise ValueError(f"Incorrect change annotation: {row['case_id']}")
    return matrix


@contextmanager
def _run_lock(root):
    _guard(root / ".runner.lock")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another experiment runner is using this directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _selection(root):
    path = root / "results" / "selection.json"
    result = _json(path)
    current = _hash(root / "experiment_matrix.csv")
    if result and result.get("initial_matrix_sha256") != current:
        raise RuntimeError("The initial matrix changed after selection began")
    return result or {"initial_matrix_sha256": current, "created_at": _now()}


def _save_selection(root, value):
    _write_json(root / "results" / "selection.json", value)


def _run_status(folder):
    failure = _json(folder / "failure.json")
    if failure:
        return {"status": "failed", "reason": failure.get("reason", "Unknown failure")}
    summary = _json(folder / "summary.json")
    if summary.get("status") == "complete":
        return {**summary, "status": "complete"}
    config = _json(folder / "config.json")
    status = config.get("status", summary.get("status", "pending"))
    if status == "complete" and (folder / "metrics.csv").exists():
        return {**summary, "status": "complete"}
    return {**summary, "status": status}


def consolidate(root=None):
    """Collect per-run outputs without opening any source test targets."""
    root = _root(root)
    metrics, radial, statuses = [], [], []
    for folder in sorted((root / "runs").glob("*/repeat_*")):
        if not folder.is_dir():
            continue
        repeat = int(folder.name.split("_")[-1])
        state = _run_status(folder)
        statuses.append({"case_id": folder.parent.name, "repeat": repeat,
                         "status": state["status"], "reason": state.get("reason", ""),
                         "fit_seconds": state.get("fit_seconds", np.nan),
                         "peak_process_rss_mb": state.get("peak_process_rss_mb", np.nan)})
        if state["status"] == "complete":
            for name, bucket in [("metrics.csv", metrics), ("radius_metrics.csv", radial)]:
                path = folder / name
                if path.exists():
                    table = pd.read_csv(path, dtype={"member_seed": str})
                    table["status"] = "complete"
                    bucket.append(table)
        elif state["status"] == "failed":
            metrics.append(pd.DataFrame([{"case_id": folder.parent.name, "repeat": repeat,
                                          "member_seed": "ensemble", "split": "validation",
                                          "target": "", "status": "failed", "reason": state.get("reason", "")}]))
    status_table = pd.DataFrame(statuses, columns=["case_id", "repeat", "status", "reason", "fit_seconds", "peak_process_rss_mb"])
    _write_csv(root / "results" / "run_status.csv", status_table)
    for name, bucket in [("metrics.csv", metrics), ("radius_metrics.csv", radial)]:
        if bucket:
            _write_csv(root / "results" / name, pd.concat(bucket, ignore_index=True))
    return status_table


def _run_batch(matrix, root=None, *, device="cpu", threads=2, repeats=(0,), retry_failed=False):
    from data import prepare_data, load_split
    from engine import run_case
    root = _root(root)
    with _run_lock(root):
        prepare_data(root)
        train = load_split("train", root)
        validation = load_split("validation", root)
        for row in matrix.to_dict("records"):
            for repeat in repeats:
                config = dict(row, repeat=int(repeat))
                if repeat:
                    config["member_seeds"] = REPEATS[repeat]
                folder = root / "runs" / config["case_id"] / f"repeat_{repeat}"
                prior = _run_status(folder)
                if prior["status"] == "failed" and not retry_failed:
                    continue
                if retry_failed and (folder / "failure.json").exists():
                    (folder / "failure.json").unlink()
                print(f"{_now()} {config['case_id']} repeat {repeat}", flush=True)
                try:
                    outcome = run_case(config, train, validation, folder, device=device, threads=threads)
                    # Engine returns its complete summary; retain a stable orchestration view.
                    if isinstance(outcome, dict):
                        summary = {k: v for k, v in outcome.items() if isinstance(v, (str, int, float, bool, type(None), list, dict))}
                        try:
                            json.dumps(summary, allow_nan=False)
                            _write_json(folder / "summary.json", summary)
                        except (TypeError, ValueError):
                            pass  # Engine owns its detailed summary when arrays are returned.
                except (KeyboardInterrupt, SystemExit):
                    consolidate(root)
                    raise
                except Exception as exc:
                    # A mismatch on a previously completed run must never overwrite its identity.
                    if prior["status"] == "complete":
                        raise
                    folder.mkdir(parents=True, exist_ok=True)
                    _write_json(folder / "failure.json", {"status": "failed", "case_id": config["case_id"],
                                "repeat": repeat, "reason": f"{type(exc).__name__}: {exc}", "at": _now()})
                    print(f"FAILED {config['case_id']}: {type(exc).__name__}: {exc}", flush=True)
                consolidate(root)
        return consolidate(root)


def controls(root=None, **kwargs):
    matrix = validate_matrix(root)
    return _run_batch(matrix[matrix.phase == "control"], root, **kwargs)


def screen(root=None, **kwargs):
    matrix = validate_matrix(root)
    # Controls run first even though the matrix is arranged by output mode.
    ordered = pd.concat([matrix[matrix.phase == "control"], matrix[matrix.phase != "control"]])
    return _run_batch(ordered, root, **kwargs)


def _require_terminal(root, matrix, repeats=(0,)):
    missing = []
    for case_id in matrix.case_id:
        for repeat in repeats:
            status = _run_status(root / "runs" / case_id / f"repeat_{repeat}")["status"]
            if status not in {"complete", "failed"}:
                missing.append(f"{case_id}/{repeat}")
    if missing:
        raise RuntimeError(f"Finish the preceding stage first ({len(missing)} pending; e.g. {missing[:3]})")


def _scores(root, matrix, *, target=None, repeats=(0,)):
    """Rank complete ensembles only; require every requested repeat and target."""
    ranked = []
    for row in matrix.to_dict("records"):
        requested = [target] if target else (TARGETS if row["output_mode"] == "joint" else [row["output_mode"]])
        values = []
        for repeat in repeats:
            folder = root / "runs" / row["case_id"] / f"repeat_{repeat}"
            if _run_status(folder)["status"] != "complete":
                break
            table = pd.read_csv(folder / "metrics.csv", dtype={"member_seed": str})
            table = table[(table.member_seed == "ensemble") & (table.split == "validation")]
            subset = table[table.target.isin(requested)]
            if len(subset) != len(requested) or set(subset.target) != set(requested):
                break
            metric = "nrmse" if target is None and row["output_mode"] == "joint" else "rmse"
            if not np.isfinite(subset[metric].to_numpy(float)).all():
                break
            values.append(float(subset[metric].mean()))
        if len(values) == len(repeats):
            ranked.append({"case_id": row["case_id"], "score": float(np.mean(values)),
                           "scores_by_repeat": values, "parameters_per_member": int(row["parameters_per_member"])})
    return sorted(ranked, key=lambda r: (r["score"], r["parameters_per_member"], r["case_id"]))


def _winner(root, matrix, **kwargs):
    ranked = _scores(root, matrix, **kwargs)
    if not ranked:
        raise RuntimeError("No complete eligible ensemble in a required candidate group")
    return ranked[0]["case_id"]


def build_interactions(root=None):
    root = _root(root)
    matrix = validate_matrix(root)
    _require_terminal(root, matrix)
    selection = _selection(root)
    path = root / "matrix_interactions.csv"
    if "interactions" in selection:
        if not path.exists() or _hash(path) != selection["interactions"]["matrix_sha256"]:
            raise RuntimeError("Frozen interaction matrix is missing or changed")
        return read_matrix(path)
    rows = {r["case_id"]: r for r in matrix.to_dict("records")}
    known = {_config_identity(r): r["case_id"] for r in rows.values()}
    groups = {"structure": ["width", "depth", "bottleneck_width", "bottleneck_activation"],
              "activation": ["hidden_activation"], "input": ["input_transform"], "target": ["target_transform"]}
    additions, parents, grid = [], {}, []
    for mode in MODES:
        mode_matrix = matrix[matrix.output_mode == mode]
        base = rows[f"{PREFIX[mode]}_base"]
        if _run_status(root / "runs" / base["case_id"] / "repeat_0")["status"] != "complete":
            raise RuntimeError(f"Reference failed: {base['case_id']}")
        choices = {name: _winner(root, mode_matrix[mode_matrix.family.isin(families)])
                   for name, families in groups.items()}
        parents[mode] = choices
        for bits in itertools.product([0, 1], repeat=4):
            candidate = dict(base)
            changed, parent_ids = [], []
            for bit, name in zip(bits, groups):
                if bit:
                    parent = rows[choices[name]]
                    field = parent["changed_factor"]
                    candidate[field] = parent[field]
                    changed.append(field)
                    parent_ids.append(parent["case_id"])
            identity = _config_identity(candidate)
            case_id = known.get(identity)
            if case_id is None:
                case_id = f"{PREFIX[mode]}_combo_{''.join(map(str, bits))}"
                candidate.update(case_id=case_id, baseline_id=base["case_id"], phase="interaction",
                                 family="combination", changed_factor="|".join(changed), baseline_value="",
                                 candidate_value="", parent_cases="|".join(parent_ids), status="planned")
                candidate["parameters_per_member"] = parameter_count(candidate)
                additions.append(candidate)
                known[identity] = case_id
            grid.append({"output_mode": mode, **dict(zip(groups, bits)), "case_id": case_id})
    result = pd.DataFrame(additions, columns=[*matrix.columns, "parent_cases"])
    _write_csv(path, result)
    _write_csv(root / "results" / "interaction_grid.csv", pd.DataFrame(grid))
    selection["interactions"] = {"frozen_at": _now(), "parents": parents, "grid": grid,
                                   "matrix_sha256": _hash(path), "new_configurations": len(result)}
    _save_selection(root, selection)
    return result


def run_interactions(root=None, **kwargs):
    return _run_batch(build_interactions(root), root, **kwargs)


def build_confirmation(root=None):
    root = _root(root)
    initial = validate_matrix(root)
    interactions = build_interactions(root)
    _require_terminal(root, interactions)
    selection = _selection(root)
    path = root / "matrix_confirmation.csv"
    if "confirmation" in selection:
        if not path.exists() or _hash(path) != selection["confirmation"]["matrix_sha256"]:
            raise RuntimeError("Frozen confirmation matrix is missing or changed")
        return read_matrix(path)
    all_rows = pd.concat([initial, interactions], ignore_index=True).fillna("")
    row_by_id = {r["case_id"]: r for r in all_rows.to_dict("records")}
    joint = all_rows[all_rows.output_mode == "joint"]
    roles = {"joint_reference": "joint_base", "joint_balanced": _winner(root, joint)}
    for target in TARGETS:
        roles[f"joint_best_{target}"] = _winner(root, joint, target=target)
        roles[f"single_reference_{target}"] = f"{PREFIX[target]}_base"
        roles[f"single_best_{target}"] = _winner(root, all_rows[all_rows.output_mode == target])
    capacity = dict(row_by_id[roles["joint_balanced"]])
    budget = sum(int(row_by_id[roles[f"single_best_{t}"]]["parameters_per_member"]) for t in TARGETS)
    upper = 1
    while parameter_count(dict(capacity, width=upper)) < budget:
        upper *= 2
    capacity["width"] = min(range(1, upper + 1), key=lambda w: (abs(parameter_count(dict(capacity, width=w)) - budget), w))
    capacity.update(case_id="joint_capacity_control", phase="confirmation", family="capacity_control",
                    changed_factor="width", baseline_id=roles["joint_balanced"], baseline_value="", candidate_value="")
    capacity["parameters_per_member"] = parameter_count(capacity)
    row_by_id[capacity["case_id"]] = capacity
    roles["capacity_control"] = capacity["case_id"]
    confirmed, identities, role_ids = [], {}, {}
    for role, original in roles.items():
        row = dict(row_by_id[original])
        row.update(main_epochs=600, phase="confirmation", status="planned", source_case_id=original)
        identity = _config_identity(row)
        if identity not in identities:
            row["case_id"] = "confirm_" + original
            identities[identity] = row["case_id"]
            confirmed.append(row)
        role_ids[role] = identities[identity]
    result = pd.DataFrame(confirmed).fillna("")
    if len(result) > 12:
        raise AssertionError("Confirmation budget exceeded")
    _write_csv(path, result)
    selection["confirmation"] = {"frozen_at": _now(), "roles": role_ids,
                                   "matrix_sha256": _hash(path), "repeats": REPEATS,
                                   "parameter_match_target_per_member_group": budget,
                                   "parameter_match_achieved_per_member": capacity["parameters_per_member"]}
    _save_selection(root, selection)
    return result


def run_confirmation(root=None, **kwargs):
    return _run_batch(build_confirmation(root), root, repeats=(1, 2, 3), **kwargs)


def freeze_final_selection(root=None):
    root = _root(root)
    matrix = build_confirmation(root)
    _require_terminal(root, matrix, (1, 2, 3))
    selection = _selection(root)
    if "final" in selection:
        return selection["final"]
    joint = matrix[matrix.output_mode == "joint"]
    roles = selection["confirmation"]["roles"]
    # The capacity control is a diagnostic, except when it deduplicates with an ordinary candidate.
    normal_ids = {value for key, value in roles.items() if key != "capacity_control"}
    competitive_joint = joint[joint.case_id.isin(normal_ids)]
    best_joint = _winner(root, competitive_joint, repeats=(1, 2, 3))
    systems = {"shared": {target: best_joint for target in TARGETS},
               "output_specific_shared": {target: _winner(root, competitive_joint, target=target, repeats=(1, 2, 3)) for target in TARGETS},
               "separate": {target: _winner(root, matrix[matrix.output_mode == target], repeats=(1, 2, 3)) for target in TARGETS},
               "shared_reference": {target: roles["joint_reference"] for target in TARGETS},
               "separate_reference": {target: roles[f"single_reference_{target}"] for target in TARGETS},
               "capacity_control": {target: roles["capacity_control"] for target in TARGETS}}
    chosen = sorted({case_id for system in systems.values() for case_id in system.values()})
    for case_id in chosen:
        for repeat in (1, 2, 3):
            if _run_status(root / "runs" / case_id / f"repeat_{repeat}")["status"] != "complete":
                raise RuntimeError(f"Final evaluation requires complete reference/control: {case_id}/{repeat}")
    rankings = {"joint_mean_nrmse": _scores(root, competitive_joint, repeats=(1, 2, 3))}
    for target in TARGETS:
        rankings[f"joint_{target}"] = _scores(root, competitive_joint, target=target, repeats=(1, 2, 3))
        rankings[f"single_{target}"] = _scores(root, matrix[matrix.output_mode == target], repeats=(1, 2, 3))
    final = {"frozen_at": _now(), "systems": systems, "evaluated_case_ids": chosen,
             "rankings": rankings, "selection_split": "validation", "test_opened": False}
    selection["final"] = final
    _save_selection(root, selection)
    return final


class SelectedSystem:
    """A frozen deployment that calls each selected five-member ensemble once."""

    def __init__(self, mapping, ensembles):
        self.mapping = dict(mapping)
        self.ensembles = ensembles

    def predict(self, frame):
        predictions = {case_id: model.predict(frame)["prediction"] for case_id, model in self.ensembles.items()}
        columns = {target: predictions[case_id][:, list(self.ensembles[case_id].targets).index(target)]
                   for target, case_id in self.mapping.items()}
        return pd.DataFrame(columns, index=frame.index)[TARGETS]


def load_selected_system(system="single_output", repeat=1, root=None, *, device="cpu"):
    """Return the selected deployment with ``predict(X) -> Q_e, Q_i, Gamma_e``.

    ``single_output`` is an alias for the selection table's ``separate`` system.
    No model selection or fitting occurs when loading a deployment.
    """
    from engine import load_ensemble
    root = _root(root)
    if repeat not in REPEATS:
        raise ValueError("Select confirmation repeat 1, 2, or 3")
    selection = _selection(root)
    if "final" not in selection:
        raise RuntimeError("Freeze final model selection before loading a deployment")
    system = {"single_output": "separate", "joint": "shared"}.get(system, system)
    systems = selection["final"]["systems"]
    if system not in systems:
        raise ValueError(f"Unknown system {system!r}; choose {sorted(systems)}")
    mapping = systems[system]
    ensembles = {case_id: load_ensemble(root / "runs" / case_id / f"repeat_{repeat}", device=device)
                 for case_id in sorted(set(mapping.values()))}
    return SelectedSystem(mapping, ensembles)


def _error_metrics(truth, prediction, scale):
    residual = prediction - truth
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denom = float(np.sum((truth - np.mean(truth)) ** 2))
    return {"rmse": rmse, "nrmse": rmse / scale, "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)), "r2": 1 - float(np.sum(residual ** 2)) / denom if denom > 0 else np.nan,
            "n_cases": len(truth)}


def _evaluate_systems(root, final, frame, split, device):
    from engine import load_ensemble
    predictions, metrics, radial, costs = [], [], [], []
    for repeat in (1, 2, 3):
        loaded = {case_id: load_ensemble(root / "runs" / case_id / f"repeat_{repeat}", device=device)
                  for case_id in final["evaluated_case_ids"]}
        predicted = {case_id: ensemble.predict(frame)["prediction"] for case_id, ensemble in loaded.items()}
        for system_name, mapping in final["systems"].items():
            unique = sorted(set(mapping.values()))
            for target, case_id in mapping.items():
                ensemble = loaded[case_id]
                index = list(ensemble.targets).index(target)
                yhat = predicted[case_id][:, index]
                y = frame[target].to_numpy(float)
                std = float(np.asarray(ensemble.raw_training_std)[index])
                meta = {"system": system_name, "repeat": repeat, "split": split, "target": target, "case_id": case_id}
                measures = _error_metrics(y, yhat, std)
                measures["negative_heat_rate"] = float(np.mean(yhat < 0)) if target in TARGETS[:2] else np.nan
                metrics.append({**meta, **measures})
                for radius in sorted(frame.RMIN_LOC.unique()):
                    mask = frame.RMIN_LOC.to_numpy() == radius
                    radial.append({**meta, "r_over_a": float(radius), **_error_metrics(y[mask], yhat[mask], std)})
                predictions.append(pd.DataFrame({**meta, "sample_id": frame.sample_id.to_numpy(),
                                                  "r_over_a": frame.RMIN_LOC.to_numpy(), "truth": y, "prediction": yhat}))
            # Measure the complete system, reusing each shared ensemble only once.
            timings = {}
            for label, batch in [("single", frame.iloc[:1]), ("batch", frame.iloc[:1100])]:
                durations = []
                for _ in range(7):
                    started = time.perf_counter()
                    for case_id in unique:
                        loaded[case_id].predict(batch)
                    durations.append(time.perf_counter() - started)
                timings[f"inference_{label}_seconds"] = float(np.median(durations[2:]))
                timings[f"inference_{label}_cases"] = len(batch)
            costs.append({"system": system_name, "repeat": repeat, "split": split,
                          "unique_ensembles": len(unique), "networks": 5 * len(unique),
                          "parameters_total": sum(parameter_count(loaded[c].config) * 5 for c in unique), **timings})
    for name, table in [("system_predictions.csv", pd.concat(predictions, ignore_index=True)),
                         ("system_metrics.csv", pd.DataFrame(metrics)),
                         ("system_radius_metrics.csv", pd.DataFrame(radial)),
                         ("system_costs.csv", pd.DataFrame(costs))]:
        path = root / "results" / name
        if path.exists():
            previous = pd.read_csv(path)
            table = pd.concat([previous[previous.split != split], table], ignore_index=True)
        _write_csv(path, table)


def evaluate_final(root=None, *, device="cpu", threads=2, external=None):
    """Open held-out targets only after configuration selection has been frozen."""
    from data import load_split, load_profile_test
    from engine import evaluate
    import torch
    root = _root(root)
    final = freeze_final_selection(root)
    with _run_lock(root):
        torch.set_num_threads(threads)
        # The only source test reads in this module are below the selection gate.
        datasets = {"test": load_split("test", root), "profile_test": load_profile_test(root)}
        external_identity = None
        if external:
            path = Path(external).resolve()
            frame = pd.read_csv(path)
            required = {"RLTS_1", "RLTS_2", "RLNS_1", "TAUS_2", "XNUE", "RMIN_LOC", *TARGETS}
            if not required <= set(frame) or not np.isfinite(frame[list(required)].to_numpy(float)).all():
                raise ValueError("External evaluation requires finite six inputs and all three flux answers")
            if "sample_id" not in frame:
                frame["sample_id"] = [f"external_{i:06d}" for i in range(len(frame))]
            if frame.sample_id.duplicated().any():
                raise ValueError("External sample IDs must be unique")
            datasets["external"] = frame
            external_identity = {"path": str(path), "sha256": _hash(path), "rows": len(frame)}
        selection = _selection(root)
        selection["final"]["test_opened"] = True
        selection["final"].setdefault("evaluation_started_at", _now())
        if external_identity:
            selection["final"]["external_dataset"] = external_identity
        _save_selection(root, selection)
        for split, frame in datasets.items():
            for case_id in final["evaluated_case_ids"]:
                for repeat in (1, 2, 3):
                    evaluate(root / "runs" / case_id / f"repeat_{repeat}", frame, split, device=device)
            _evaluate_systems(root, final, frame, split, device)
        selection = _selection(root)
        selection["final"]["evaluation_completed_at"] = _now()
        selection["final"]["evaluated_datasets"] = {key: len(value) for key, value in datasets.items()}
        _save_selection(root, selection)
        consolidate(root)
        return selection["final"]


def run_all(root=None, *, device="cpu", threads=2, retry_failed=False, external=None):
    options = dict(device=device, threads=threads, retry_failed=retry_failed)
    screen(root, **options)
    run_interactions(root, **options)
    run_confirmation(root, **options)
    return evaluate_final(root, device=device, threads=threads, external=external)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "status", "controls", "screen", "interactions", "confirmation", "final", "all"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--external", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = _root(args.root)
    options = dict(device=args.device, threads=args.threads, retry_failed=args.retry_failed)
    if args.stage == "prepare":
        from data import prepare_data
        validate_matrix(root)
        result = prepare_data(root)
    elif args.stage == "status":
        status = consolidate(root)
        result = {"counts": status.status.value_counts().to_dict(), "runs": len(status)}
    elif args.stage == "controls":
        result = controls(root, **options)
    elif args.stage == "screen":
        result = screen(root, **options)
    elif args.stage == "interactions":
        result = run_interactions(root, **options)
    elif args.stage == "confirmation":
        result = run_confirmation(root, **options)
    elif args.stage == "final":
        result = evaluate_final(root, device=args.device, threads=args.threads, external=args.external)
    else:
        result = run_all(root, **options, external=args.external)
    if isinstance(result, pd.DataFrame):
        print(result.to_string(index=False))
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
