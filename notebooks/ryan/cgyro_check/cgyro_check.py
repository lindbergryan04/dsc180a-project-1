"""Compare the scan_db surrogate's error with TGLF's own difference from nonlinear CGYRO.

Optional extension in the Tasks 5-7 handout. For each of the 43 SAT3 cases (Dudding et al.
2022) there is a nonlinear CGYRO simulation and a TGLF (SAT2) run made from the same inputs by
pyrokinetics. The surrogate is the Task 2 five-network ensemble trained on scan_db (TGLF only);
its saved predictions on these cases are in ``sat3_surrogate_predictions.csv``.

CGYRO fluxes are the sums over toroidal modes and both fields (phi, A_par) in
``bin.cgyro.ky_flux``, averaged over the second half of each simulation. Both codes report
gyro-Bohm units with the same reference: deuterium mass, electron temperature, and B_unit
(CGYRO ``m_norm`` is deuterium for all three isotopes, and the TGLF inputs use the default
``UNITS = GYRO``). The script checks the CGYRO reference mass and species order per case.

From the repository root, after extracting the SAT3 archive (``DATA.md``)::

    python notebooks/ryan/cgyro_check/cgyro_check.py

Writes ``cgyro_check.csv`` (one row per case and flux) and ``cgyro_summary.csv`` to
``--output`` (default ``notebooks/ryan/runs/cgyro_check``).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "notebooks/ryan"))
from tglf_data import load_cases  # noqa: E402

TARGETS = ["Q_i", "Q_e", "Gamma_e"]
BLOCKS = 4  # equal time blocks in the averaging window, for a rough averaging uncertainty


def cgyro_header(folder):
    """Grid sizes from out.cgyro.grids and the species table from out.cgyro.info."""
    grids = (folder / "out.cgyro.grids").read_text().split()
    n_n, n_species, n_field = (int(grids[i]) for i in range(3))
    lines = (folder / "out.cgyro.info").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "m/m_norm" in line) + 1
    species = []
    for line in lines[start:start + n_species]:
        fields = line.split()
        species.append({"z": int(fields[1]), "mass": float(fields[4])})
    return n_n, n_species, n_field, species


def cgyro_fluxes(folder):
    """Time-averaged particle and heat flux per species, summed over modes and fields."""
    n_n, n_species, n_field, species = cgyro_header(folder)
    time = np.loadtxt(folder / "out.cgyro.time", ndmin=2)[:, 0]
    raw = np.fromfile(folder / "bin.cgyro.ky_flux", dtype="float32")
    per_step = n_species * 3 * n_field * n_n
    extra = raw.size // per_step - len(time)
    # Like pygacode, read one flux record per time entry; a restart can leave one extra record.
    if raw.size % per_step or extra not in (0, 1) or np.any(np.diff(time) <= 0):
        raise ValueError(f"{folder}: flux file does not match the grid and time records")
    raw = raw[:per_step * len(time)]
    # Layout as in pygacode: (species, moment, field, toroidal mode, time), Fortran order.
    flux = raw.reshape((n_species, 3, n_field, n_n, len(time)), order="F").sum(axis=(2, 3))
    window = time >= time[-1] / 2
    ion = next(i for i, s in enumerate(species) if s["z"] > 0)
    electron = next(i for i, s in enumerate(species) if s["z"] < 0)
    if abs(species[electron]["mass"] - 2.724e-4) > 1e-6:
        raise ValueError(f"{folder}: CGYRO reference mass is not deuterium")
    series = {"Q_i": flux[ion, 1], "Q_e": flux[electron, 1], "Gamma_e": flux[electron, 0]}
    result = {"t_end": time[-1], "window_start": time[window][0], "extra_flux_records": extra}
    for target, values in series.items():
        blocks = np.array_split(values[window], BLOCKS)
        means = np.array([b.mean() for b in blocks])
        result[target] = values[window].mean()
        result[f"{target}_block_se"] = means.std(ddof=1) / np.sqrt(BLOCKS)
    return result


def summarize(rows):
    def stats(a, b):
        diff = a - b
        rel = (diff.abs() / b.abs()).replace(np.inf, np.nan)
        return {"rmse": float(np.sqrt((diff ** 2).mean())), "median_abs_rel_diff": float(rel.median()),
                "mean_signed_diff": float(diff.mean())}
    out = []
    for target, group in rows.groupby("target", sort=False):
        for name, a, b in [("surrogate vs TGLF", group.surrogate, group.tglf),
                           ("TGLF vs CGYRO", group.tglf, group.cgyro),
                           ("surrogate vs CGYRO", group.surrogate, group.cgyro)]:
            out.append({"target": target, "comparison": name, "n": len(group), **stats(a, b)})
    return pd.DataFrame(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sat3-data", type=Path,
                        default=REPO / "data/Task1_TGLF_SAT3_dataset/SAT3_dataset/data")
    parser.add_argument("--output", type=Path, default=REPO / "notebooks/ryan/runs/cgyro_check")
    args = parser.parse_args(argv)
    tglf = load_cases(args.sat3_data)
    surrogate = pd.read_csv(HERE / "sat3_surrogate_predictions.csv").set_index("case")
    if len(tglf) != 43 or set(tglf.index) != set(surrogate.index):
        raise ValueError("Expected the same 43 SAT3 cases in the data and the surrogate predictions")
    rows = []
    for case in sorted(tglf.index):
        cg = cgyro_fluxes(args.sat3_data / case / "nonlinear")
        for target in TARGETS:
            rows.append({"case": case, "isotope": tglf.loc[case, "species_type_2"], "target": target,
                         "tglf": tglf.loc[case, target], "cgyro": cg[target],
                         "cgyro_block_se": cg[f"{target}_block_se"], "surrogate": surrogate.loc[case, target],
                         "cgyro_window": f"{cg['window_start']:.0f}-{cg['t_end']:.0f}",
                         "extra_flux_records": cg["extra_flux_records"]})
    rows = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "cgyro_check.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(args.output / "cgyro_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
