#!/usr/bin/env python
"""Task 4 figures: DIII-D flux profile (Step 1) and input histograms vs scan_db.

    python plot_d3d.py
Reads step1_check.csv and d3d_database.csv (this folder) and Sweekrit's committed scan_db table.
Writes graphics/d3d_profile_fluxes.png and graphics/d3d_vs_scan_db_inputs.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCAN = HERE.parents[1] / "sbhatnagar" / "task2" / "data" / "scan_db_table.csv"
OUT = HERE / "graphics"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.8, "axes.axisbelow": True})
OUT.mkdir(exist_ok=True)

# Step 1: flux profile (three panels, one scale each)
p = pd.read_csv(HERE / "step1_check.csv")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (k, label, c) in zip(axes, [("Q_e", "$Q_e$", BLUE), ("Q_i", "$Q_i$", ORANGE),
                                    ("Gamma_e", r"$\Gamma_e$", AQUA)]):
    ax.plot(p.r_over_a, p[f"{k}_ours"], color=c, lw=2, marker="o", ms=5)
    ax.set(title=f"{label} (gyro-Bohm)", xlabel="r/a")
    if k == "Gamma_e":
        ax.axhline(0, color="#52514e", lw=0.8)
fig.suptitle("DIII-D #128913 TGLF flux profile")
fig.tight_layout()
fig.savefig(OUT / "d3d_profile_fluxes.png", dpi=150)

# Step 2: where the DIII-D database sits inside the scan_db box
me = pd.read_csv(HERE / "d3d_database.csv").rename(columns={"r_over_a": "RMIN_LOC"})
scan = pd.read_csv(SCAN)
feats = {  # names from the Task 1 and Task 4 handouts
    "RLTS_1": "Electron temperature gradient ($a/L_{T_e}$)",
    "RLTS_2": "Ion temperature gradient ($a/L_{T_i}$)",
    "RLNS_1": "Density gradient ($a/L_n$)",
    "TAUS_2": "Ion-to-electron temperature ratio ($T_i/T_e$)",
    "XNUE": "Electron collision frequency",
    "RMIN_LOC": "Minor radius ($r/a$)",
}
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (f, name) in zip(axes.flat, feats.items()):
    lo, hi = min(me[f].min(), scan[f].min()), max(me[f].max(), scan[f].max())
    bins = np.geomspace(lo, hi, 40) if f == "XNUE" else np.linspace(lo, hi, 40)
    ax.hist(scan[f], bins=bins, density=True, color=BLUE, alpha=0.55, label=f"scan_db (n={len(scan)})")
    ax.hist(me[f], bins=bins, density=True, color=ORANGE, alpha=0.55, label=f"DIII-D, seed 3 (n={len(me)})")
    ax.set(yticks=[])
    ax.set_title(name, fontsize=11)
    if f == "XNUE":
        ax.set_xscale("log")
axes.flat[0].legend(frameon=False, fontsize=9)
fig.suptitle("Input distributions: DIII-D database vs scan_db (normalized)")
fig.tight_layout()
fig.savefig(OUT / "d3d_vs_scan_db_inputs.png", dpi=150)
print("wrote", *sorted(p.name for p in OUT.glob("d3d_*.png")))
