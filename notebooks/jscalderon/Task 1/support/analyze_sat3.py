"""SAT3 parsing and matched-case checks recovered from Create Sat3 dataset slides.

The original chat ran this analysis inline. This saved version uses portable
paths and the Python standard library; it retains the original flux mapping.
"""
import argparse
import json
import math
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[1] / "SAT3_dataset.tar")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {}
    with tarfile.open(args.archive) as archive:
        for member in archive:
            parts = member.name.split("/")
            if len(parts) == 5 and parts[3] == "tglf_sat2" and parts[4] in ("input.tglf.gen", "out.tglf.gbflux"):
                cases.setdefault(parts[2], {})[parts[4]] = archive.extractfile(member).read().decode()
    (args.output_dir / "raw_cases.json").write_text(json.dumps(cases, indent=2))
    parsed = {}
    flux_keys = ["Gamma_e", "Gamma_i", "Q_e", "Q_i", "skip_e", "skip_i", "Pi_e", "Pi_i"]
    for case, files in sorted(cases.items()):
        settings = {}
        for line in files["input.tglf.gen"].splitlines():
            value, key = line.split()
            try:
                value = float(value)
            except ValueError:
                pass
            settings[key] = value
        values = [float(value) for value in files["out.tglf.gbflux"].split()]
        assert int(settings["NS"]) == 2 and len(values) == 4 * int(settings["NS"])
        settings.update(zip(flux_keys, values))
        parsed[case] = settings
    assert len(parsed) == 43
    assert all(math.isfinite(row[key]) for row in parsed.values() for key in flux_keys)
    assert all(row[key] > 0 for row in parsed.values() for key in ("Q_e", "Q_i"))
    (args.output_dir / "parsed.json").write_text(json.dumps(parsed, indent=2))
    groups = [([f"{letter}_D" for letter in "bcad"], {"RLTS_1", "RLTS_2"}),
              ([f"{letter}_D" for letter in "ghai"], {"Q_PRIME_LOC"})]
    for cases_in_group, expected in groups:
        changing = {key for key in parsed[cases_in_group[0]] if key not in flux_keys
                    and len({parsed[case][key] for case in cases_in_group}) > 1}
        assert changing == expected, (cases_in_group, changing)
    for letter in "abcdefjx":
        rows = [parsed[f"{letter}_{isotope}"] for isotope in "HDT"]
        changing = {key for key in rows[0] if key not in flux_keys and len({row[key] for row in rows}) > 1}
        assert changing == {"MASS_2"}, (letter, changing)
        print(letter, "T/H ion heat-flux increase (%):", 100 * (rows[2]["Q_i"] / rows[0]["Q_i"] - 1))
    print("Verified:", len(parsed), "cases; matched gradient scans; eight isotope triplets")
    print("Temperature-gradient Q_i factor:", parsed["d_D"]["Q_i"] / parsed["b_D"]["Q_i"])
    print("Safety-factor-gradient Q_i factor:", parsed["i_D"]["Q_i"] / parsed["g_D"]["Q_i"])
    print("Safety-factor-gradient Q_e factor:", parsed["i_D"]["Q_e"] / parsed["g_D"]["Q_e"])


if __name__ == "__main__":
    main()
