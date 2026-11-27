"""Make every data figure in the final Q1 report from committed result tables.

One shared style keeps the figures consistent: a sans-serif face, hairline gray grids, thin
marks with light washes, and the two-color scheme of the data-coverage figure. Blue marks the
subject of a figure, gray the comparison or reference, and a dark navy step of the same blue
a third series; lines also differ by marker shape. Thin near-black lines mark references
(equal error, nominal coverage, R^2 = 0). No figure retrains anything. From the repository root::

    python quarter_1_report/make_figures.py

Writes PDFs to ``quarter_1_report/final_figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RYAN = REPO / "notebooks/ryan"
OUT = REPO / "quarter_1_report/final_figures"

# Roles (validated light-mode categorical slots 1-3 plus ink and a muted gray).
BLUE, LIGHT_BLUE, NAVY = "#2a78d6", "#6da7ec", "#0d366b"
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal steps 250 -> 700
FLUXES = ["Q_i", "Q_e", "Gamma_e"]
FLUX_LABEL = {"Q_i": r"Ion heat flux $Q_i$", "Q_e": r"Electron heat flux $Q_e$",
              "Gamma_e": r"Electron particle flux $\Gamma_e$"}
WIDTH = 6.5  # inches, the text width of the report

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "savefig.dpi": 300, "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def legend_row(fig, handles, y=1.0):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(handles),
               handlelength=1.6, columnspacing=1.6)


def dot(color, label, marker="o", line=False):
    return Line2D([], [], color=color, marker=marker, linestyle="-" if line else "none",
                  markersize=5, markeredgecolor="white", markeredgewidth=0.6, label=label)


def read(path, **kw):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False, **kw)


# ----------------------------------------------------------------------------- Figure 2
def coverage():
    """Task 4: the DIII-D database's inputs against the scan_db design ranges."""
    team = read(RYAN / "task4/d3d_team_database.csv")
    scan = read(REPO / "notebooks/sbhatnagar/task2/data/scan_db_table.csv",
                usecols=["RMIN_LOC", "RLTS_1", "RLTS_2", "RLNS_1", "TAUS_2", "XNUE"])
    scan = scan.rename(columns={"RMIN_LOC": "r_over_a"})
    panels = [("r_over_a", r"Minor radius $r/a$", (0.25, 0.75)),
              ("RLTS_1", r"Electron temp. gradient $a/L_{T_e}$", (1.5, 3.5)),
              ("RLTS_2", r"Ion temp. gradient $a/L_{T_i}$", (1.5, 3.5)),
              ("RLNS_1", r"Density gradient $a/L_n$", (1.0, 3.0)),
              ("TAUS_2", r"Temperature ratio $T_i/T_e$", (0.5, 1.5)),
              ("XNUE", r"Collision frequency $\nu_{ei}$", (0.01, 1.0))]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.6))
    for ax, (key, title, (low, high)) in zip(axes.flat, panels):
        values = pd.concat([team[key], scan[key]])
        if key == "XNUE":
            edges = np.geomspace(values.min() * 0.9, values.max() * 1.1, 31)
            ax.set_xscale("log")
        elif key == "r_over_a":
            edges = np.linspace(0.175, 0.825, 14)
        else:
            edges = np.linspace(values.min(), values.max(), 31)
        ax.axvspan(low, high, color="#f0efec", zorder=0, lw=0)
        for data, color in [(scan, MUTED), (team, BLUE)]:
            pct = 100 * np.histogram(data[key], bins=edges)[0] / len(data)
            ax.stairs(pct, edges, fill=True, color=color, alpha=0.12, lw=0)
            ax.stairs(pct, edges, color=color, lw=1.3)
        outside = (~team[key].between(low, high)).mean()
        ax.set_title(title, fontsize=8)
        ax.text(0.98, 0.94, f"{100 * outside:.0f}% outside", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=SECONDARY)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(bottom=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cases (%)")
    legend_row(fig, [Patch(facecolor=BLUE, alpha=0.6, label=f"DIII-D #128913 database ({len(team):,})"),
                     Patch(facecolor=MUTED, alpha=0.6, label=f"scan_db ({len(scan):,})"),
                     Patch(facecolor="#e6e5e0", label="scan_db design range")], y=0.99)
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig_coverage")


# ----------------------------------------------------------------------------- Figure 3
def generic_limits():
    """Task 3: the scan_db ensemble outside its ranges and at the DIII-D reference case."""
    ood = read(RYAN / "task3/results/ood_comparison.csv")
    paired = read(RYAN / "task3/results/d3d_paired_predictions.csv")
    names = {"ood_RLTS_1": r"$a/L_{T_e}$", "ood_RLTS_2": r"$a/L_{T_i}$", "ood_RLN": r"$a/L_n$",
             "ood_TAUS_2": r"$T_i/T_e$", "ood_XNUE": r"$\nu_{ei}$", "ood_Q_LOC": r"$q$",
             "ood_KAPPA_LOC": r"$\kappa$", "ood_DRMAJDX_LOC": r"$\Delta'$", "ood_RMIN_LOC": r"$r/a$"}
    order = list(names)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.5), gridspec_kw={"height_ratios": [1, 1.05]})
    for ax, flux in zip(axes[0], FLUXES):
        part = ood.loc[ood.target.eq(flux)].set_index("dataset").loc[order]
        y = np.arange(len(order))[::-1]
        ax.axvline(1, color=INK, lw=0.8)
        ax.hlines(y, 1, part.rmse_vs_test, color=BLUE, lw=1.2, alpha=0.45)
        ax.scatter(part.rmse_vs_test, y, s=22, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y, [names[d] for d in order])
        ax.set_xlim(0.2, 20)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(FLUX_LABEL[flux])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("RMSE outside range / RMSE in range")
    axes[0, 0].set_ylabel("Input pushed beyond its scan_db range")
    for ax, flux in zip(axes[1], FLUXES):
        truth, pred = paired[f"{flux}_true"], paired[f"{flux}_prediction"]
        low, high = min(truth.min(), pred.min()), max(truth.max(), pred.max())
        pad = 0.05 * (high - low)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=INK, lw=0.8)
        ax.scatter(truth, pred, s=9, color=BLUE, alpha=0.6, edgecolor="none")
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal")
        ratio = pred.mean() / truth.mean()
        ax.text(0.04, 0.95, f"mean prediction / TGLF = {ratio:.2f}", transform=ax.transAxes,
                va="top", fontsize=7, color=SECONDARY)
        ax.set_xlabel("TGLF (gyro-Bohm)")
    axes[1, 0].set_ylabel("Generic surrogate (gyro-Bohm)")
    fig.text(0.0, 1.0, "a", fontweight="bold", fontsize=10, va="bottom")
    fig.text(0.0, 0.47, "b", fontweight="bold", fontsize=10, va="bottom")
    fig.tight_layout(h_pad=2.2)
    save(fig, "fig_generic_limits")


# ----------------------------------------------------------------------------- Figure 4
def specialists():
    """Task 5: test RMSE of the shared and per-flux systems on 1,100 held-out cases."""
    summary = read(RYAN / "task5/results/system_summary.csv")
    pooled = read(RYAN / "task5/results/pooled_scores.csv")
    rows = [("shared_reference", "Joint baseline (Task 2 design)", MUTED),
            ("separate_reference", "Baseline, one ensemble per flux", MUTED),
            ("shared", "Tuned joint ensemble", MUTED),
            ("separate", "Tuned per-flux specialists", BLUE),
            ("pooled_15", "Specialists, 15 networks pooled", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.95), sharey=True)
    for ax, flux in zip(axes, FLUXES):
        test = summary.loc[summary.split.eq("test") & summary.target.eq(flux)].set_index("system")
        pool = pooled.loc[pooled.split.eq("test") & pooled.target.eq(flux) & pooled.model.eq("pooled_15"), "rmse"].item()
        y = np.arange(len(rows))[::-1]
        for yi, (key, _, color) in zip(y, rows):
            mean = pool if key == "pooled_15" else test.loc[key, "rmse_mean"]
            if key != "pooled_15":  # SD across the three independently trained repeats
                sd = test.loc[key, "rmse_sd"]
                ax.hlines(yi, mean - sd, mean + sd, color=color, lw=1.6, alpha=0.6)
            ax.scatter(mean, yi, s=30, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.axvline(test.loc["shared_reference", "rmse_mean"], color=MUTED, lw=0.8, zorder=1)
        ax.set_yticks(y, [label for _, label, _ in rows])
        ax.set_title(FLUX_LABEL[flux])
        ax.set_xlabel("Test RMSE (gyro-Bohm)")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.12)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_specialists")


# ----------------------------------------------------------------------------- Figure 5
def intervals_by_radius():
    """Task 6 Part B: error and 90% interval coverage by radius for the full-data specialists."""
    radius = read(RYAN / "task6/results/error_by_radius.csv")
    cohorts = [("test", "Held-out cases at the 13 training radii", MUTED, "o"),
               ("intermediate_varied", "Varied cases at 12 radii between them", BLUE, "s")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.5), sharex=True)
    for col, flux in enumerate(FLUXES):
        for cohort, _, color, marker in cohorts:
            part = radius.loc[radius.target.eq(flux) & radius.cohort.eq(cohort)].sort_values("r_over_a")
            axes[0, col].plot(part.r_over_a, 100 * part.coverage, color=color, marker=marker,
                              markeredgecolor="white", markeredgewidth=0.6)
            axes[1, col].plot(part.r_over_a, part.rmse / part.rms_flux, color=color, marker=marker,
                              markeredgecolor="white", markeredgewidth=0.6)
        axes[0, col].axhline(90, color=INK, lw=0.8)
        axes[0, col].set_ylim(30, 102)
        axes[0, col].set_title(FLUX_LABEL[flux])
        axes[1, col].set_xlabel(r"Minor radius $r/a$")
        axes[1, col].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("90% interval coverage (%)")
    axes[1, 0].set_ylabel("RMSE / RMS flux")
    legend_row(fig, [dot(color, label, marker=marker, line=True) for _, label, color, marker in cohorts], y=0.995)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig_radius")


# ----------------------------------------------------------------------------- Figure 6
def learning_curves():
    """Task 6: RMSE against training-set size, and validation RMSE against hidden width."""
    curves = read(RYAN / "task6/results/learning_curves_by_cohort.csv")
    width = read(RYAN / "task6/results/width_capacity.csv")
    cohorts = [("train", "Training subset", MUTED, "o"), ("validation", "Validation", LIGHT_BLUE, "D"),
               ("test", "Held-out test", BLUE, "s"), ("intermediate_varied", "Intermediate radii", NAVY, "^")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.9))
    for col, flux in enumerate(FLUXES):
        ax = axes[0, col]
        for cohort, _, color, marker in cohorts:
            part = curves.loc[curves.target.eq(flux) & curves.cohort.eq(cohort)].sort_values("size")
            ax.fill_between(part["size"], part.min_rmse, part.max_rmse, color=color, alpha=0.12, lw=0)
            ax.plot(part["size"], part.mean_rmse, color=color, marker=marker, markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1000, 2000, 4000, 8000, 16586], ["1k", "2k", "4k", "8k", "16.6k"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(FLUX_LABEL[flux])
        ax.set_xlabel("Training runs")
        ax = axes[1, col]
        for color, size in zip(BLUE_RAMP[1:], [1000, 4000, 16586]):
            part = width.loc[width.target.eq(flux) & width["size"].eq(size)].sort_values("hidden_width")
            ax.plot(part.hidden_width, part.validation_rmse, color=color, marker="o", markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.annotate(f"{size / 1000:.3g}k runs", (part.hidden_width.iloc[-1], part.validation_rmse.iloc[-1]),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.5, color=SECONDARY)
        ax.set_xscale("log", base=2)
        ax.set_xlim(22, 1400)
        ax.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Hidden units per full-width layer")
    axes[0, 0].set_ylabel("RMSE (gyro-Bohm)")
    axes[1, 0].set_ylabel("Validation RMSE")
    legend_row(fig, [dot(color, label, marker=marker, line=True) for _, label, color, marker in cohorts], y=0.995)
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig_learning")


# ----------------------------------------------------------------------------- Figure 7
def transfer():
    """Task 7: R² of the frozen shot-128913 models on shot 193870, by training size."""
    scores = read(RYAN / "task7/results/transfer_scores.csv")
    task6 = scores.loc[scores.family.eq("Task 6") & scores.width.eq("original")]
    cohorts = [("128913_test", "Shot 128913, held-out test", MUTED, "o"),
               ("193870_varied", "Shot 193870, 517 perturbed cases", BLUE, "s"),
               ("193870_profile", "Shot 193870, 13 profile points", NAVY, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.1), sharey=True)
    for ax, flux in zip(axes, FLUXES):
        ax.axhline(0, color=INK, lw=0.8)
        for cohort, _, color, marker in cohorts:
            part = task6.loc[task6.target.eq(flux) & task6.cohort.eq(cohort)]
            stats = part.groupby("train_size").r2.agg(["mean", "min", "max"]).reset_index()
            floor = -1.25
            ax.fill_between(stats.train_size, stats["min"].clip(lower=floor), stats["max"], color=color,
                            alpha=0.12, lw=0)
            ax.plot(stats.train_size, stats["mean"].clip(lower=floor), color=color, marker=marker, markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.5)
            for size, value in zip(stats.train_size, stats["mean"]):
                if value < floor:  # off scale: draw a marker at the floor and print the value
                    ax.scatter(size, floor, marker="v", s=28, color=color, zorder=4)
                    ax.annotate(f"{value:.1f}", (size, floor), xytext=(6, 2), textcoords="offset points",
                                fontsize=6.5, color=SECONDARY)
        ax.set_xscale("log")
        ax.set_xticks([1000, 2000, 4000, 8000, 16586], ["1k", "2k", "4k", "8k", "16.6k"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-1.35, 1.05)
        ax.set_title(FLUX_LABEL[flux])
        ax.set_xlabel("Training runs (shot 128913)")
    axes[0].set_ylabel(r"$R^2$ against TGLF")
    legend_row(fig, [dot(color, label, marker=marker, line=True) for _, label, color, marker in cohorts], y=0.99)
    fig.tight_layout()
    save(fig, "fig_transfer")


# ----------------------------------------------------------------------------- Figure 8
def cgyro():
    """CGYRO check: TGLF and the generic surrogate against nonlinear CGYRO on 43 SAT3 cases."""
    rows = read(RYAN / "cgyro_check/results/cgyro_check.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3))
    for ax, flux in zip(axes, FLUXES):
        part = rows.loc[rows.target.eq(flux)]
        low = min(part[["tglf", "cgyro", "surrogate"]].min())
        high = max(part[["tglf", "cgyro", "surrogate"]].max())
        heat = flux != "Gamma_e"
        if heat:
            lims = (low / 1.3, high * 1.3)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            pad = 0.05 * (high - low)
            lims = (low - pad, high + pad)
        ax.plot(lims, lims, color=INK, lw=0.8)
        ax.scatter(part.cgyro, part.tglf, s=16, facecolor="none", edgecolor=MUTED, linewidth=0.9, zorder=3)
        ax.scatter(part.cgyro, part.surrogate, s=10, color=BLUE, edgecolor="white", linewidth=0.4, zorder=4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        if heat:
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())
        ax.set_title(FLUX_LABEL[flux])
        ax.set_xlabel("Nonlinear CGYRO (gyro-Bohm)")
    axes[0].set_ylabel("Prediction (gyro-Bohm)")
    legend_row(fig, [dot(MUTED, "TGLF (SAT2)", marker="o"), dot(BLUE, "Generic surrogate (trained on TGLF)")], y=0.98)
    for handle in fig.legends[0].legend_handles[:1]:  # open circle, drawn in the reference gray
        handle.set_markerfacecolor("none")
        handle.set_markeredgecolor(MUTED)
        handle.set_markeredgewidth(0.9)
    fig.tight_layout()
    save(fig, "fig_cgyro")


if __name__ == "__main__":
    coverage()
    generic_limits()
    specialists()
    intervals_by_radius()
    learning_curves()
    transfer()
    cgyro()
    print(f"Wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT.relative_to(REPO)}")
