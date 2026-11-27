"""Make and collect DIII-D #128913 TGLF cases (Task 4 database and Task 5 varied test).

Every case starts from a supplied ``input.tglf`` at one radius and scales five local inputs
by random factors (Task 4 handout): ``RLTS_1`` and ``RLTS_2`` by U(0.6, 1.4) each, both density
gradients by one shared U(0.6, 1.4), ``TAUS_2`` by U(0.75, 1.25), and ``XNUE`` by a factor
that is uniform in log10 between 0.5 and 2. Geometry and all other settings stay fixed.
Cases cycle through the radii in order, and draws come from NumPy ``Generator(PCG64(seed))``.

Subcommands (run from the repository root):

``check-profile``
    After TGLF has run in the supplied profile folders, compare all 39 fluxes with
    ``d3d_profile_answers.dat`` to three significant figures (Task 4, Step 1).
``plan``
    Write ``cases.csv`` and one ``input.tglf`` per case. Ryan's Task 4 database is seed 180401
    with 1,000 cases; the Task 5 varied test is seed 180501 with 20 cases at each of the 12
    intermediate radii (``--varied-test``).
``collect``
    After TGLF has run, add ``Q_e``, ``Q_i``, ``Gamma_e`` and the output hash to ``cases.csv``.

TGLF itself runs with the course helper, e.g.
``python data/Task4_DIIID_Database/task4_d3d/tools/run_tglf_parallel.py <output>/data -nproc 8``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tarfile

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from d3d_profile import set_values  # noqa: E402  (same helper as the course task4_d3d/tools)

FACTORS = ["f_Te", "f_Ti", "f_n", "f_tau", "f_nu"]
CHANGED = {"RLTS_1", "RLTS_2", "RLNS_1", "RLNS_2", "TAUS_2", "XNUE"}
LIMITS = {"RLTS_1": (0.6, 1.4), "RLTS_2": (0.6, 1.4), "RLNS_1": (0.6, 1.4), "RLNS_2": (0.6, 1.4),
          "TAUS_2": (0.75, 1.25), "XNUE": (0.5, 2.0)}
OWNER = "Ryan Lindberg"


def read_values(text):
    """input.tglf text -> {KEY: number}, accepting Fortran D exponents."""
    values = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(\S+)", line)
        if match:
            try:
                values[match.group(1).upper()] = float(match.group(2).replace("D", "E").replace("d", "e"))
            except ValueError:
                pass
    return values


def configuration_hash(text):
    """Hash of every assignment, insensitive to order and number formatting (for deduplication)."""
    normalized = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        key = key.upper()
        if key in normalized:
            raise ValueError(f"Duplicate input key: {key}")
        try:
            number = float(value.replace("D", "E").replace("d", "e"))
            if not math.isfinite(number):
                raise ValueError("non-finite input")
            normalized[key] = (0.0 if number == 0 else number).hex()
        except ValueError:
            if value.upper() not in {"T", "F", ".TRUE.", ".FALSE."}:
                raise ValueError(f"Unexpected value for {key}: {value}")
            normalized[key] = "T" if value.upper() in {"T", ".TRUE."} else "F"
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def load_profiles(source):
    """Return [(radius name, input.tglf text)] sorted by r/a, from a folder or a .tar.gz archive."""
    source = Path(source)
    texts = {}
    if source.is_dir():
        for path in source.glob("*/tglf_sat2/input.tglf"):
            texts[path.parents[1].name] = path.read_text()
    else:
        with tarfile.open(source, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith("/tglf_sat2/input.tglf"):
                    texts[Path(member.name).parts[-3]] = archive.extractfile(member).read().decode()
    if len(texts) < 2 or not all(re.fullmatch(r"r[0-9]+", name) for name in texts):
        raise ValueError(f"Expected radius folders r###/tglf_sat2/input.tglf in {source}")
    return sorted(texts.items(), key=lambda item: read_values(item[1])["RMIN_LOC"])


def plan_cases(profiles, seed, n_cases, *, varied_test=False):
    """Draw the factors and build each case's input text; returns (table, {case: text})."""
    rng = np.random.Generator(np.random.PCG64(seed))
    rows, texts = [], {}
    for k in range(n_cases):
        radius, original = profiles[k % len(profiles)]
        base = read_values(original)
        if base["RLNS_1"] != base["RLNS_2"]:
            raise ValueError(f"{radius}: baseline density gradients must match.")
        factors = dict(zip(FACTORS, [rng.uniform(0.6, 1.4), rng.uniform(0.6, 1.4), rng.uniform(0.6, 1.4),
                                     rng.uniform(0.75, 1.25), 10.0 ** rng.uniform(np.log10(0.5), np.log10(2.0))]))
        updates = {"RLTS_1": base["RLTS_1"] * factors["f_Te"], "RLTS_2": base["RLTS_2"] * factors["f_Ti"],
                   "RLNS_1": base["RLNS_1"] * factors["f_n"], "RLNS_2": base["RLNS_1"] * factors["f_n"],
                   "TAUS_2": base["TAUS_2"] * factors["f_tau"], "XNUE": base["XNUE"] * factors["f_nu"]}
        text = set_values(original, updates)  # writes six significant figures
        actual = read_values(text)
        if actual.keys() != base.keys() or any(actual[key] != base[key] for key in base if key not in CHANGED):
            raise ValueError(f"{radius}: a fixed input changed.")
        for key, (low, high) in LIMITS.items():
            ratio = actual[key] / base[key]  # six-figure rounding moves an endpoint by at most ~5e-6
            if not low * (1 - 5e-6) <= ratio <= high * (1 + 5e-6):
                raise ValueError(f"{radius}: {key} is outside the Task 4 factor range.")
        case = f"c{k + 1:04d}"
        texts[case] = text
        digest = hashlib.sha256(text.encode()).hexdigest()
        if varied_test:
            rows.append({"sample_id": f"test_seed{seed}_{case}", "case_id": f"test_seed{seed}_{case}", "case": case,
                         "seed": seed, "owner": OWNER, "radius": radius, **factors, **actual, "input_sha256": digest})
        else:
            rows.append({"case_id": f"seed{seed}_{case}", "case": case, "owner": OWNER, "seed": seed,
                         "radius": radius, "r_over_a": float(base["RMIN_LOC"]), **factors, **actual,
                         "input_sha256": digest, "configuration_sha256": configuration_hash(text)})
    table = pd.DataFrame(rows)
    if table["input_sha256"].duplicated().any():
        raise ValueError("Duplicate input configurations.")
    counts = table.groupby("radius").size()
    if counts.max() - counts.min() > 1:
        raise ValueError("Cases are not evenly distributed across radii.")
    return table, texts


def read_fluxes(path):
    """Two-species out.tglf.gbflux -> electron heat, ion heat, electron particle flux."""
    values = [float(text.replace("D", "E").replace("d", "e")) for text in Path(path).read_text().split()]
    if len(values) != 8 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{path}: expected eight finite two-species flux values")
    # Blocks: particle flux (e, i), heat flux (e, i), momentum flux, energy exchange.
    return {"Q_e": values[2], "Q_i": values[3], "Gamma_e": values[0]}


def check_profile(profile_dir):
    answers = []
    for line in (profile_dir / "d3d_profile_answers.dat").read_text().splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0].startswith("r"):
            answers.append([fields[0], *map(float, fields[1:])])
    answers = pd.DataFrame(answers, columns=["radius", "r_over_a", "Q_e", "Q_i", "Gamma_e"])
    rows = []
    for row in answers.itertuples(index=False):
        computed = read_fluxes(profile_dir / row.radius / "tglf_sat2/out.tglf.gbflux")
        for flux, value in computed.items():
            expected = getattr(row, flux)
            rows.append({"radius": row.radius, "flux": flux, "computed": value, "reference": expected,
                         "agrees_3_sig_figs": float(f"{value:.3g}") == float(f"{expected:.3g}")})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if not table.agrees_3_sig_figs.all():
        raise SystemExit(f"{(~table.agrees_3_sig_figs).sum()} fluxes disagree. Stop before generating cases.")
    print(f"PASS: all {len(table)} fluxes agree with the supplied answers to three significant figures.")


def write_plan(profile, seed, n_cases, per_radius, varied_test, output):
    profiles = load_profiles(profile)
    n_cases = n_cases or per_radius * len(profiles)
    table, texts = plan_cases(profiles, seed, n_cases, varied_test=varied_test)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty. Choose a new --output directory.")
    for case, text in texts.items():
        path = output / "data" / case / "tglf_sat2" / "input.tglf"
        path.parent.mkdir(parents=True)
        path.write_text(text)
    table.to_csv(output / "cases.csv", index=False)
    print(f"Wrote {len(table)} cases at {table.radius.nunique()} radii to {output}")


def collect(output):
    table = pd.read_csv(output / "cases.csv", float_precision="round_trip")
    rows = []
    for case in table["case"]:
        folder = output / "data" / case / "tglf_sat2"
        flux = folder / "out.tglf.gbflux"
        rows.append({**read_fluxes(flux), "output_sha256": hashlib.sha256(flux.read_bytes()).hexdigest()})
    result = pd.concat([table, pd.DataFrame(rows)], axis=1)
    result.to_csv(output / "cases_with_fluxes.csv", index=False)
    print(f"Collected {len(result)} cases; {(result[['Q_e', 'Q_i']] <= 0).any(axis=1).sum()} have a "
          "nonpositive heat flux (kept and flagged downstream).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-profile", help="compare the 13 profile runs with the supplied answers")
    check.add_argument("--profile", type=Path, required=True, help="task4_d3d/d3d_profile")
    plan = sub.add_parser("plan", help="write cases.csv and input decks")
    plan.add_argument("--profile", type=Path, required=True, help="profile folder or .tar.gz archive")
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--cases", type=int, help="total cases (Task 4: 1000)")
    plan.add_argument("--per-radius", type=int, default=20, help="cases per radius when --cases is omitted")
    plan.add_argument("--varied-test", action="store_true", help="use the Task 5 varied-test table layout")
    plan.add_argument("--output", type=Path, required=True)
    gather = sub.add_parser("collect", help="add TGLF fluxes to cases.csv")
    gather.add_argument("--output", type=Path, required=True, help="folder written by plan")
    args = parser.parse_args(argv)
    if args.command == "check-profile":
        check_profile(args.profile)
    elif args.command == "plan":
        write_plan(args.profile, args.seed, args.cases, args.per_radius, args.varied_test, args.output)
    else:
        collect(args.output)


if __name__ == "__main__":
    main()
