"""One question per figure for Task 3; fitting and calibration stay in task3_workflow."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from task3_workflow import TARGETS, OOD_SETS, INPUT_FRACTIONS, bounds

BLUE = "#17658A"
ORANGE = "#C55A28"
INK = "#203342"
GRAY = "#6B7780"
PALE = "#DDE9EF"
LABELS = ["Ion heat flux", "Electron heat flux", "Particle flux"]


def _style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#C5CED4")
    ax.tick_params(colors=INK, labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#E4E9ED", linewidth=.7)


def _heading(fig, title, subtitle):
    fig.suptitle(title, x=.025, y=.98, ha="left", fontsize=17, fontweight="bold", color=INK)
    fig.text(.025, .98 - .38 / fig.get_figheight(), subtitle, ha="left", va="top",
             fontsize=10.5, color=GRAY)


def _finish(fig, caption, output_dir, name):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}.png", dpi=200, facecolor="white", bbox_inches="tight")
        fig.savefig(output_dir / f"{name}.svg", facecolor="white", bbox_inches="tight")
    return fig, caption


def calibration_figure(prediction, truth, calibration, output_dir=None):
    """Compare one-SD coverage with calibrated coverage, on validation data only."""
    error = np.abs(prediction["mean"] - truth[TARGETS].to_numpy())
    scale = np.maximum(prediction["std"], calibration["floor"])
    before = 100 * (error <= scale).mean(axis=0)
    after = 100 * (error <= scale * calibration["scaled_multiplier"]).mean(axis=0)
    target = 100 * (1 - calibration["alpha"])
    fig, ax = plt.subplots(figsize=(9.6, 5.1))
    fig.subplots_adjust(left=.10, right=.97, bottom=.15, top=.70)
    _heading(fig, "Validation coverage before and after interval calibration",
             f"How often do the intervals contain the TGLF answer? All {len(truth):,} validation cases.")
    x = np.arange(3)
    for offset, values, color, label in [(-.18, before, PALE, "Before: one SD (with floor)"),
                                         (.18, after, BLUE, "After: calibrated interval")]:
        bars = ax.bar(x + offset, values, width=.32, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=5, color=INK, fontsize=11)
    ax.axhline(target, color=ORANGE, linestyle="--", linewidth=1.4, label=f"{target:g}% target")
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 105), yticks=[0, 25, 50, 75, 100],
           ylabel="TGLF answers inside the interval (%)")
    _style(ax)
    fig.legend(loc="upper center", bbox_to_anchor=(.52, .83), ncol=3, frameon=False, fontsize=10)
    caption = (
        "**Figure 1. Calibration widens intervals to match observed errors.** Before calibration, "
        f"one floor-adjusted SD covers {before[0]:.1f}%, {before[1]:.1f}%, and {before[2]:.1f}% "
        f"of validation answers. After calibration, coverage is about **{target:g}%** for each flux. "
        "These are the cases used to choose the widths; the next figure checks separate test cases."
    )
    return _finish(fig, caption, output_dir, "01_calibration")


def test_figure(prediction, truth, calibration, test_metrics, output_dir=None, sample_size=50, seed=2026):
    """Show a fixed random sample; compute coverage from every test case."""
    # Select before looking at errors, with the same cases in all three panels.
    # Sorting only changes display order and uses predictions, not TGLF answers.
    n = len(truth)
    sample = np.random.default_rng(seed).choice(n, size=min(sample_size, n), replace=False)
    lower, upper = bounds(prediction, calibration)
    actual = truth[TARGETS].to_numpy()
    coverage = 100 * ((actual >= lower) & (actual <= upper)).mean(axis=0)
    np.testing.assert_allclose(coverage, 100 * test_metrics.loc[TARGETS, "coverage"].to_numpy())
    level = 100 * (1 - calibration["alpha"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    fig.subplots_adjust(left=.065, right=.985, bottom=.18, top=.69, wspace=.25)
    _heading(fig, "Do the 90% prediction intervals contain the TGLF answers?",
             f"{len(sample)} randomly selected test cases shown; coverage below uses all {n:,} test cases.")
    handles = [Line2D([], [], color=BLUE, linewidth=2, label="Mean prediction"),
               Patch(facecolor=PALE, label=f"{level:g}% prediction interval"),
               Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=5, label="Actual TGLF")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .84), ncol=3, frameon=False)
    export = truth[TARGETS].iloc[sample].rename(columns={t: f"{t}_true" for t in TARGETS}).copy()
    for j, ax in enumerate(axes):
        order = np.argsort(prediction["mean"][sample, j], kind="stable")
        selected = sample[order]
        x = np.arange(1, len(sample) + 1)
        ax.fill_between(x, lower[selected, j], upper[selected, j], color=PALE, zorder=1)
        ax.plot(x, prediction["mean"][selected, j], color=BLUE, linewidth=1.9, zorder=3)
        ax.scatter(x, actual[selected, j], s=20, color=ORANGE, edgecolors="white", linewidths=.35, zorder=4)
        ax.set(title=f"{LABELS[j]}\n{coverage[j]:.1f}% covered on all {n:,} cases",
               xlabel="Cases ordered by prediction", xlim=(.3, len(sample) + .7),
               xticks=[1, len(sample)//2, len(sample)])
        ax.margins(y=.08)
        _style(ax)
        target = TARGETS[j]
        export[f"{target}_prediction"] = prediction["mean"][sample, j]
        export[f"{target}_lower"] = lower[sample, j]
        export[f"{target}_upper"] = upper[sample, j]
        export[f"{target}_plot_order"] = np.argsort(order) + 1
    axes[0].set_ylabel("Flux (gyro-Bohm)")
    fig.text(.065, .035, "A TGLF dot outside the blue band is a missed interval. The horizontal axis is case order, not time.",
             fontsize=10, color=GRAY)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        export.to_csv(Path(output_dir) / "02_displayed_test_cases.csv", index_label="case")
    caption = (
        "**Figure 2. Predictions with 90% prediction intervals.** The blue line is the 15-network mean, "
        "the shaded band is the calibrated interval, and each orange dot is a TGLF answer. "
        "A dot outside the band is a miss. Across the **full test set**, coverage is "
        f"**{coverage[0]:.1f}%, {coverage[1]:.1f}%, and {coverage[2]:.1f}%**. "
        f"For readability, all panels show the same {len(sample)} cases sampled without replacement "
        f"using seed {seed}, sorted separately by each flux's prediction. The connected line and band "
        "only guide the eye between separate cases. Bounds remain unclipped, including negative bounds."
    )
    return _finish(fig, caption, output_dir, "02_test_reliability")


def ood_figure(ood_metrics, output_dir=None):
    """One metric: coverage for every outside-range group and output."""
    names = ["Electron temperature gradient", "Ion temperature gradient", "Density gradient",
             "Temperature ratio", "Collisionality", "Safety factor", "Elongation",
             "Geometry (DRMAJDX_LOC)", "Minor radius"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    fig.subplots_adjust(left=.235, right=.98, bottom=.15, top=.74, wspace=.18)
    _heading(fig, "Prediction interval coverage outside the training range",
             "Each row changes one input beyond its training range. Dashed line: 90% target.")
    for ax, target in zip(axes, TARGETS):
        part = ood_metrics.loc[ood_metrics.target.eq(target)].set_index("dataset").loc[OOD_SETS]
        values = part["Coverage (%)"].to_numpy()
        ax.axvline(90, color=GRAY, linestyle="--", linewidth=1.2)
        ax.scatter(values, np.arange(len(OOD_SETS)), s=55,
                   color=[ORANGE if v < 90 else BLUE for v in values], zorder=3)
        ax.set(title=LABELS[TARGETS.index(target)], xlim=(0, 105), xticks=[0, 30, 60, 90],
               xlabel="TGLF answers covered (%)", yticks=np.arange(len(OOD_SETS)), yticklabels=names)
        _style(ax, "x")
    axes[0].invert_yaxis()
    worst = ood_metrics.loc[ood_metrics["Coverage (%)"].idxmin()]
    flux_name = LABELS[TARGETS.index(worst.target)].lower()
    input_name = names[OOD_SETS.index(worst.dataset)]
    caption = (
        "**Figure 3. Check coverage when an input leaves the training range.** Orange dots fall below "
        "the 90% target. The lowest coverage is "
        f"**{worst['Coverage (%)']:.0f}% for {flux_name}** in the **{input_name}** group. "
        "Treat these predictions with caution. Each dot uses only 30 cases, so one additional miss "
        "changes coverage by 3.3 percentage points. Error and disagreement comparisons are available below."
    )
    return _finish(fig, caption, output_dir, "03_outside_range")


def forward_figure(mc_prediction, paired, truth, output_dir=None):
    """Compare output distributions with one histogram panel per flux."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.1))
    fig.subplots_adjust(left=.065, right=.985, bottom=.16, top=.70, wspace=.25)
    _heading(fig, "DIII-D flux distributions as the inputs vary",
             "Output distributions as the DIII-D inputs vary within their stated uncertainty.")
    mean_bias = []
    for j, (ax, target) in enumerate(zip(axes, TARGETS)):
        mc = mc_prediction["mean"][:, j]
        actual = truth[target].to_numpy()
        bins = np.histogram_bin_edges(np.r_[mc, actual], bins=22)
        for values, color in [(mc, BLUE), (actual, ORANGE)]:
            counts, _ = np.histogram(values, bins=bins, density=True)
            ax.stairs(counts, bins, color=color, linewidth=2)
            ax.stairs(counts, bins, color=color, fill=True, alpha=.09)
        mean_bias.append(100 * (paired["mean"][:, j].mean() / actual.mean() - 1))
        ax.set(title=LABELS[j], xlabel="Flux (gyro-Bohm)", ylim=(0, None))
        _style(ax)
    axes[0].set_ylabel("Probability density")
    fig.legend(handles=[Line2D([], [], color=BLUE, linewidth=2, label="Surrogate: 10,000 input draws"),
                        Line2D([], [], color=ORANGE, linewidth=2, label="TGLF: 300 supplied runs")],
               loc="upper center", bbox_to_anchor=(.52, .84), ncol=2, frameon=False)
    caption = (
        "**Figure 4. Compare the location, width, and shape of the distributions.** "
        "On the same 300 inputs, the predicted mean particle flux differs from the TGLF mean "
        f"by **{mean_bias[2]:+.0f}%** "
        f"(ion heat: {mean_bias[0]:+.1f}%; electron heat: {mean_bias[1]:+.1f}%). "
        "This spread comes from varying the inputs; it is different from the prediction intervals in Figure 2."
    )
    return _finish(fig, caption, output_dir, "04_forward_uncertainty")


def sensitivity_figure(sensitivity, output_dir=None):
    """Show flux SD from varying each input, highlighting only the largest bar."""
    order = list(INPUT_FRACTIONS)
    names = ["Electron temperature gradient", "Ion temperature gradient", "Density gradient",
             "Temperature ratio", "Collisionality", "Safety factor", "Q_PRIME_LOC"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharey=True)
    fig.subplots_adjust(left=.255, right=.985, bottom=.16, top=.74, wspace=.23)
    _heading(fig, "Which input uncertainty changes the flux most?",
             "Vary one input at a time. A longer bar means more variation in the predicted flux.")
    leaders = []
    for ax, target, title in zip(axes, TARGETS, LABELS):
        part = sensitivity.loc[sensitivity.target.eq(target)].set_index("input").loc[order]
        values = part["std"].to_numpy()
        best = int(np.argmax(values))
        leaders.append(names[best].lower())
        ax.barh(np.arange(len(order)), values, height=.6,
                color=[ORANGE if k == best else PALE for k in range(len(order))])
        ax.set(title=title, xlabel="Flux SD (gyro-Bohm)", xlim=(0, values.max()*1.10),
               yticks=np.arange(len(order)), yticklabels=names)
        _style(ax, "x")
    axes[0].invert_yaxis()
    other_leaders = (f"electron heat and particle flux both respond most to the **{leaders[1]}**. "
                     if leaders[1] == leaders[2] else
                     f"electron heat responds most to the **{leaders[1]}**, and particle flux "
                     f"to the **{leaders[2]}**. ")
    caption = (
        "**Figure 5. Orange marks the largest effect.** Ion heat flux responds most to the "
        f"**{leaders[0]}**; " + other_leaders + "These priorities apply to the supplied measurement "
        "error bars near this reference case, according to the surrogate. The bars are not additive "
        "shares of total uncertainty, and the TGLF mismatch in Figure 4 limits their interpretation."
    )
    return _finish(fig, caption, output_dir, "05_input_sensitivity")
