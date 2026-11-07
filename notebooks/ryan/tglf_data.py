"""Read course TGLF/SAT2 inputs and species-resolved gyro-Bohm fluxes."""
from math import isclose, isfinite
from pathlib import Path

import pandas as pd

SCALAR_PARAMETERS = ("NS", "XNUE", "RMIN_LOC", "KAPPA_LOC", "Q_LOC", "Q_PRIME_LOC", "DRMAJDX_LOC")
SPECIES_PARAMETERS = ("MASS", "ZS", "TAUS", "AS", "RLTS", "RLNS")
INPUTS = (*SCALAR_PARAMETERS, *(f"{name}_{s}" for name in SPECIES_PARAMETERS for s in (1, 2)))


def _number(text):
    value = float(text.replace("D", "E").replace("d", "e"))
    if not isfinite(value):
        raise ValueError(f"non-finite value {text!r}")
    return value


def _read_case(case):
    directory = case / "tglf_sat2"
    inputs = {}
    for line in (directory / "input.tglf.gen").read_text().splitlines():
        fields = line.split("#", 1)[0].split("!", 1)[0].split()
        if not fields:
            continue
        value, name = fields
        if name in INPUTS:
            if name in inputs:
                raise ValueError(f"duplicate input {name}")
            inputs[name] = _number(value)

    missing = set(INPUTS) - inputs.keys()
    if missing:
        raise ValueError(f"missing inputs: {sorted(missing)}")
    if inputs["NS"] != 2 or inputs["ZS_1"] >= 0 or inputs["ZS_2"] <= 0:
        raise ValueError("expected two species ordered as electron, then ion")
    if min(inputs["MASS_1"], inputs["MASS_2"]) <= 0:
        raise ValueError("active species must have positive mass")

    isotope = "other_ion"
    if isclose(inputs["ZS_2"], 1):
        for mass, label in ((0.5, "H"), (1.0, "D"), (1.5, "T")):
            if isclose(inputs["MASS_2"], mass, rel_tol=1e-6):
                isotope = label
                break
    record = {"case": case.name, **inputs, "species_type_1": "electron", "species_type_2": isotope}

    values = [_number(text) for text in (directory / "out.tglf.gbflux").read_text().split()]
    if len(values) != 8:
        raise ValueError(f"expected four flux blocks for two species, found {len(values)} entries")
    for block, quantity in enumerate(("Gamma", "Q", "Pi", "S")):
        for species, suffix in ((1, "e"), (2, "i")):
            record[f"{quantity}_{species}"] = record[f"{quantity}_{suffix}"] = values[2 * block + species - 1]
    return record


def load_cases(data_dir):
    """Return a case-indexed table for a case collection or a single case directory."""
    root = Path(data_dir)
    if (root / "tglf_sat2").is_dir():
        cases = [root]
    else:
        cases = sorted(case for case in root.iterdir() if case.is_dir() and not case.name.startswith("."))
    if not cases:
        raise ValueError(f"No case directories in {root}")
    records = []
    for case in cases:
        try:
            records.append(_read_case(case))
        except (OSError, ValueError) as exc:
            raise ValueError(f"{case}: {exc}") from exc
    return pd.DataFrame.from_records(records).set_index("case")
