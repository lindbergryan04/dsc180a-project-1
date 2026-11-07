import os
import sys
from pathlib import Path
import pandas as pd


def parse_tglf_gen(file_path):
    params = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                val_str, name = parts[0], parts[1]
                try:
                    val = float(val_str)
                    if val.is_integer():
                        val = int(val)
                except ValueError:
                    val = val_str
                params[name] = val
    return params


def infer_species_type(zs, mass):
    if zs < 0 or abs(mass - 0.0002724486) < 1e-4:
        return "electron"
    if abs(mass - 0.5) < 1e-2:
        return "H"
    if abs(mass - 1.0) < 1e-2:
        return "D"
    if abs(mass - 1.5) < 1e-2:
        return "T"
    return f"ion_{mass}"


def parse_case(case_dir):
    case_path = Path(case_dir)
    case_name = case_path.name
    tglf_dir = case_path / "tglf_sat2"
    gen_file = tglf_dir / "input.tglf.gen"
    flux_file = tglf_dir / "out.tglf.gbflux"

    if not gen_file.exists():
        raise FileNotFoundError(f"Missing input.tglf.gen in {tglf_dir}")
    if not flux_file.exists():
        raise FileNotFoundError(f"Missing out.tglf.gbflux in {tglf_dir}")

    raw_params = parse_tglf_gen(gen_file)
    if "NS" not in raw_params:
        raise KeyError(f"NS parameter missing in {gen_file}")

    ns = int(raw_params["NS"])
    if ns <= 0:
        raise ValueError(f"Invalid NS={ns} in {gen_file}")

    case_dict = {
        "case": case_name,
        "NS": ns
    }

    scalar_keys = [
        "XNUE",
        "RMIN_LOC",
        "DRMAJDX_LOC",
        "KAPPA_LOC",
        "Q_LOC",
        "Q_PRIME_LOC"
    ]
    for key in scalar_keys:
        if key not in raw_params:
            raise KeyError(f"Missing scalar parameter {key} in {gen_file}")
        case_dict[key] = raw_params[key]

    species_prefixes = ["MASS_", "ZS_", "TAUS_", "AS_", "RLTS_", "RLNS_"]
    for i in range(1, ns + 1):
        for prefix in species_prefixes:
            key = f"{prefix}{i}"
            if key not in raw_params:
                raise KeyError(f"Missing parameter {key} in {gen_file}")
            case_dict[key] = raw_params[key]

        zs = case_dict[f"ZS_{i}"]
        mass = case_dict[f"MASS_{i}"]
        stype = infer_species_type(zs, mass)
        case_dict[f"species_type_{i}"] = stype

    with open(flux_file, "r") as f:
        flux_vals = [float(x) for x in f.read().strip().split()]

    expected_flux_count = 4 * ns
    if len(flux_vals) != expected_flux_count:
        raise ValueError(f"Expected {expected_flux_count} flux values in {flux_file}, found {len(flux_vals)}")

    gamma_block = flux_vals[0:ns]
    q_block = flux_vals[ns:2 * ns]
    pi_block = flux_vals[3 * ns:4 * ns]

    for i in range(1, ns + 1):
        case_dict[f"Gamma_{i}"] = gamma_block[i - 1]
        case_dict[f"Q_{i}"] = q_block[i - 1]
        case_dict[f"Pi_{i}"] = pi_block[i - 1]

        stype = case_dict[f"species_type_{i}"]
        if stype == "electron":
            case_dict["Gamma_e"] = gamma_block[i - 1]
            case_dict["Q_e"] = q_block[i - 1]
            case_dict["Pi_e"] = pi_block[i - 1]
        else:
            case_dict["Gamma_i"] = gamma_block[i - 1]
            case_dict["Q_i"] = q_block[i - 1]
            case_dict["Pi_i"] = pi_block[i - 1]

    return case_dict


def load_all_cases(data_dir):
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Directory {data_path} does not exist")

    case_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])
    parsed_cases = []
    errors = []

    for case_dir in case_dirs:
        try:
            case_dict = parse_case(case_dir)
            parsed_cases.append(case_dict)
        except Exception as e:
            errors.append((case_dir.name, str(e)))

    if errors:
        sys.stderr.write(f"Errors occurred in {len(errors)} cases:\n")
        for cname, err in errors:
            sys.stderr.write(f"  {cname}: {err}\n")

    return pd.DataFrame(parsed_cases), errors


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    dataset_path = Path("SAT3_dataset/data")
    df, errs = load_all_cases(dataset_path)
    print(f"Parsed {len(df)} cases successfully with {len(errs)} errors.\n")

    summary_columns = [
        "case",
        "species_type_1",
        "species_type_2",
        "Gamma_1",
        "Gamma_2",
        "Q_1",
        "Q_2",
        "Pi_1",
        "Pi_2"
    ]
    print(df[summary_columns].to_string(index=False))
