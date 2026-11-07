"""Reproduce the starter-scan analysis from the 43 course SAT3 cases."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tglf_data import INPUTS, load_cases
from nonlinear_tools import (
    baseline_scans, dependency_screen, matched_rectangles,
    nested_model_comparison, univariate_sweep,
)

FEATURES = ["XNUE", "RLTS_2", "RLNS_2", "MASS_2", "TAUS_2",
            "magnetic_shear", "Q_LOC", "RMIN_LOC", "KAPPA_LOC", "DRMAJDX_LOC"]
TARGETS = ["Q_i", "Q_e", "Gamma_e"]
FLUXES = ["Gamma_e", "Gamma_i", "Q_e", "Q_i", "Pi_e", "Pi_i"]


def starter_cases(directory):
    frame = load_cases(directory)
    frame["letter"] = frame.index.to_series().str.rsplit("_", n=1).str[0]
    isotope = frame.index.to_series().str.rsplit("_", n=1).str[-1]
    if len(frame) != 43 or frame.letter.nunique() != 26 or "a_D" not in frame.index:
        raise ValueError("Expected the 43 SAT3 starter cases in 26 letter groups, including a_D")
    if not ((isotope == frame.species_type_2) & isotope.isin(["H", "D", "T"])).all():
        raise ValueError("Case name and ion species disagree")
    positive = ["MASS_1", "MASS_2", "AS_1", "AS_2", "TAUS_1", "TAUS_2", "XNUE", "Q_e", "Q_i"]
    if frame[positive].le(0).any().any() or frame.Q_LOC.eq(0).any():
        raise ValueError("Invalid starter input or nonpositive heat flux")
    for first, second in [("RLTS_1", "RLTS_2"), ("RLNS_1", "RLNS_2"), ("Gamma_e", "Gamma_i")]:
        if not frame[first].equals(frame[second]):
            raise ValueError(f"Revisit the starter-scan coordinates: {first} differs from {second}")
    frame["magnetic_shear"] = frame.Q_PRIME_LOC * frame.RMIN_LOC**2 / frame.Q_LOC**2
    return frame


def quality_flags(frame):
    """Reproduce the exploratory flux, isolation and duplicate-input screens."""
    flux = frame[FLUXES].copy()
    for name in ("Q_i", "Q_e"):
        flux[name] = np.log10(flux[name])
    q1, q3 = flux.quantile(.25), flux.quantile(.75)
    span = q3 - q1
    flags = ((flux.lt(q1 - 3 * span) | flux.gt(q3 + 3 * span)) & span.gt(0)).add_prefix("review_")
    coords = frame[FEATURES].copy()
    coords["XNUE"] = np.log10(coords.XNUE)
    coords = (coords - coords.loc["a_D"]) / (coords.max() - coords.min())
    xy = coords.to_numpy()
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    q1, q3 = np.quantile(nearest, [.25, .75])
    flags["review_isolated_inputs"] = nearest > q3 + 1.5 * (q3 - q1)
    flags["review_duplicate_inputs"] = frame.duplicated(list(INPUTS), keep=False)
    flags["quality_flag"] = np.where(flags.any(axis=1), "review", "ok")
    return flags


def isotope_exponents(frame):
    rows = []
    for letter, group in frame.groupby("letter"):
        if len(group) != 3 or set(group.species_type_2) != {"H", "D", "T"}:
            continue
        if group[[name for name in INPUTS if name != "MASS_2"]].nunique().gt(1).any():
            raise ValueError(f"Isotope triplet {letter} differs in other inputs")
        values = group.set_index("species_type_2")[["Q_i", "Q_e"]]
        for target in values:
            rows.append({
                "letter": letter, "target": target,
                "beta_H_to_D": np.log(values.at["D", target] / values.at["H", target]) / np.log(2),
                "beta_D_to_T": np.log(values.at["T", target] / values.at["D", target]) / np.log(1.5),
            })
    return pd.DataFrame(rows)


def write_figures(frame, scores, ranking, output):
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for target, ax in zip(TARGETS, axes):
        table = scores[scores.target == target]
        ax.barh(table.model, table.letter_MAE)
        ax.set(title=target, xlabel="Letter-balanced MAE (gyro-Bohm)")
        ax.invert_yaxis()
    fig.savefig(figure_dir / "starter_model_errors.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    axes.flat[-1].set_visible(False)
    for row, ax in zip(ranking.head(5).itertuples(), axes.flat):
        for isotope, group in frame.groupby("species_type_2"):
            ax.scatter(group[row.variable_1], group[row.variable_2], label=isotope, s=24)
        ax.set(xlabel=row.variable_1, ylabel=f"{row.variable_2} (gyro-Bohm)",
               title=f"Spearman r = {row.spearman:.2f}")
        if row.variable_1 == "XNUE":
            ax.set_xscale("log")
    axes.flat[0].legend(title="Ion")
    fig.savefig(figure_dir / "starter_correlations.png", dpi=150)
    plt.close(fig)


def analyze(directory, output):
    frame = starter_cases(directory)
    flags = quality_flags(frame)
    groups = frame.letter.to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    frame.join(flags).to_csv(output / "sat3_tglf_table.csv")
    print(f"Loaded {len(frame)} cases. Quality flags: {flags.quality_flag.value_counts().to_dict()}", flush=True)

    screen_inputs = FEATURES + ["Q_PRIME_LOC"]
    deps = dependency_screen(frame, screen_inputs + TARGETS + ["Pi_e", "Pi_i"], groups, screen_inputs)
    deps.to_csv(output / "dependency_screen.csv", index=False)
    ranking = deps.loc[deps.variable_1.isin(FEATURES) & deps.variable_2.isin(TARGETS)].copy()
    ranking["absolute_spearman"] = ranking.spearman.abs()
    ranking = ranking.sort_values(["absolute_spearman", "variable_1", "variable_2"],
                                  ascending=[False, True, True], kind="stable")
    ranking.to_csv(output / "input_flux_ranking.csv", index=False)

    scans = baseline_scans(frame, FEATURES)
    pd.concat([cases.assign(scan_input=name) for name, cases in scans.items()]).to_csv(output / "baseline_scans.csv")
    rectangles, coverage = matched_rectangles(frame, FEATURES, TARGETS)
    rectangles.to_csv(output / "matched_interactions.csv", index=False)
    coverage.to_csv(output / "interaction_coverage.csv", index=False)
    isotope_exponents(frame).to_csv(output / "isotope_exponents.csv", index=False)

    univariate, univariate_predictions = univariate_sweep(frame, FEATURES, TARGETS, groups)
    univariate.to_csv(output / "one_input_model_scores.csv", index=False)
    univariate_predictions.to_csv(output / "one_input_predictions.csv", index=False)
    scores, predictions, tuning = nested_model_comparison(frame, FEATURES, TARGETS, groups)
    scores.to_csv(output / "model_scores.csv", index=False)
    predictions.to_csv(output / "held_out_predictions.csv", index=False)
    tuning.to_csv(output / "ridge_tuning.csv", index=False)
    write_figures(frame, scores, ranking, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sat3-data", type=Path, required=True, help="SAT3_dataset/data directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated tables and figures")
    args = parser.parse_args()
    analyze(args.sat3_data, args.output)
