"""Check saved Task 2 metrics against case-level predictions and data splits."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

TASK_DIR = Path(__file__).resolve().parents[1]
TARGETS = ("Q_i", "Q_e", "Gamma_e")
MODELS = {"Mean baseline": "_mean", "Selected NN": "_NN"}
CASE_COUNTS = {"train": 3496, "validation": 750, "test": 750, "SAT3": 43}


def read_rows(directory, name):
    with (directory / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def by_case(directory, name):
    rows = read_rows(directory, name)
    cases = {row["case"]: row for row in rows}
    if len(cases) != len(rows):
        raise ValueError(f"{name}: duplicate case IDs")
    return cases


def close(label, actual, expected):
    if not math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"{label}: found {actual}, expected {expected:.10g}")


def check_results(directory):
    summary = json.loads((directory / "summary.json").read_text())
    scan = by_case(directory, "scan_table.csv")
    sat3 = by_case(directory, "sat3_table.csv")
    split = by_case(directory, "split_assignments.csv")
    excluded = by_case(directory, "excluded_cases.csv")
    if len(scan) != 5000 or scan.keys() != split.keys():
        raise ValueError("Expected identical sets of 5,000 cases in scan_table.csv and split_assignments.csv")
    held = {case for case, row in scan.items() if min(float(row[t]) for t in ("Q_i", "Q_e")) <= 0}
    if len(held) != 4 or excluded.keys() != held:
        raise ValueError("excluded_cases.csv must identify the four cases with nonpositive heat flux")
    ids = {group: {case for case in scan.keys() - held if split[case]["split"] == group}
           for group in ("train", "validation", "test")}
    counts = {group: len(cases) for group, cases in ids.items()}
    counts["SAT3"] = len(sat3)
    if counts != CASE_COUNTS or summary["split_sizes"] != counts:
        raise ValueError(f"Case counts: found {counts}, summary reports {summary['split_sizes']}, expected {CASE_COUNTS}")
    scales = {t: statistics.pstdev(float(scan[c][t]) for c in ids["train"]) for t in TARGETS}

    def metric_key(row):
        return (row["dataset"], row["model"], row["output"])

    metric_rows = read_rows(directory, "metrics.csv")
    csv_rows = {metric_key(row): row for row in metric_rows}
    json_rows = {metric_key(row): row for row in summary["metrics"]}
    if (len(metric_rows) != 12 or len(summary["metrics"]) != 12 or len(csv_rows) != 12
            or csv_rows.keys() != json_rows.keys()):
        raise ValueError("Expected the same 12 dataset/model/output combinations in metrics.csv and summary.json")
    for dataset, name, cases, truth in [
        ("Random test", "test_predictions.csv", ids["test"], scan),
        ("SAT3", "sat3_predictions.csv", set(sat3), sat3),
    ]:
        rows = by_case(directory, name)
        if rows.keys() != cases:
            raise ValueError(f"{name}: prediction case IDs differ from the {dataset} cases")
        for target in TARGETS:
            y = [float(truth[c][target]) for c in rows]
            for c in rows:
                close(f"{name} {c} true {target}", rows[c][target], float(truth[c][target]))
            for model, suffix in MODELS.items():
                key = (dataset, model, target)
                mse = statistics.mean((float(row[target + suffix]) - value) ** 2
                                      for row, value in zip(rows.values(), y))
                rmse = math.sqrt(mse)
                expected = {"n": len(y), "RMSE": rmse, "R2": 1 - mse / statistics.pvariance(y),
                            "nRMSE": rmse / scales[target]}
                for field, value in expected.items():
                    for source, row in [("metrics.csv", csv_rows[key]), ("summary.json", json_rows[key])]:
                        close(f"{source} {dataset} / {model} / {target} / {field}", row[field], value)
        negative = [row for row in rows.values() if float(row["Gamma_e"]) < 0]
        signs = {"negative_particle_cases": len(negative),
                 "negative_particle_sign_correct": sum(float(row["Gamma_e_NN"]) < 0 for row in negative),
                 "negative_predicted_heat_values": sum(float(row[t + "_NN"]) <= 0
                                                       for row in rows.values() for t in ("Q_i", "Q_e"))}
        if summary["signed_checks"][dataset] != signs:
            raise ValueError(f"summary.json: {dataset} sign counts differ from predictions; expected {signs}")

    curves = read_rows(directory, "learning_curves.csv")
    sizes = [250, 500, 1000, 2000, counts["train"]]
    expected_groups = {(rep, size, model) for rep in range(3) for size in sizes for model in MODELS}
    groups = {(int(row["repeat"]), int(row["training_n"]), row["model"]) for row in curves}
    if groups != expected_groups or len(curves) != len(expected_groups):
        raise ValueError("learning_curves.csv: expected each combination of three repeats, five sizes and two models")
    for row in curves:
        mean = statistics.mean(float(row[t]) for t in TARGETS)
        close(f"learning_curves.csv {row['model']} / {row['training_n']} / repeat {row['repeat']}",
              row["mean_nRMSE"], mean)
    print(f"Verified {counts} cases, all 12 metric rows, particle-flux signs, and {len(curves)} learning-curve rows.")


def check_notebook(path):
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        if cell["execution_count"] is None:
            raise ValueError(f"{path.name} cell {index}: no recorded execution")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error" or "Traceback" in "".join(output.get("text", "")):
                raise ValueError(f"{path.name} cell {index}: recorded error")
    print(f"Verified recorded cell executions in {path.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=TASK_DIR / "results",
                        help="Directory containing the saved Task 2 tables and summary")
    parser.add_argument("--notebook", type=Path, help="Also verify execution counts and errors in this executed notebook")
    args = parser.parse_args()
    try:
        check_results(args.results_dir)
        if args.notebook:
            check_notebook(args.notebook)
    except (OSError, KeyError, ValueError, TypeError, ArithmeticError) as error:
        sys.exit(f"Validation failed: {error}")
