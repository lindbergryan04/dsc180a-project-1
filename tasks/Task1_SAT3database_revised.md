# SAT3 Starter Database

Dataset structure and Task 1

## Overview

The starter database has 43 tokamak plasma cases. Each case has gyrokinetic (CGYRO) linear/nonlinear simulations and a matching quasilinear (TGLF) prediction, so the same plasma state is described at two levels of fidelity. For your first task you only need the TGLF files: one input file and one output file per case.

- **Where it came from:** The open SAT3 CGYRO dataset (Dudding et al., Zenodo record 14505760). It was built to calibrate TGLF’s newest saturation rule, SAT3.
- **What we added:** Each nonlinear CGYRO input was converted to a TGLF input (using pyrokinetics), and TGLF was run on every case with saturation rule SAT2 (`SAT_RULE = 2`).
- **How to get it:** Download `SAT3_dataset` from the course Google Drive (about 400 MB) and unpack it. Don’t edit the files; work on copies in your own folder.

## Dataset structure

Every case is one folder under `data/`, and every case folder has the same three subfolders.

```
SAT3_dataset/
├── tools/              scripts used to build the database (not needed)
└── data/
    ├── a_D/
    │   ├── linear/
    │   │   ├── 0/      one linear CGYRO run per wavenumber ky
    │   │   ├── 1/
    │   │   └── ... 38/
    │   ├── nonlinear/  the nonlinear (turbulence) CGYRO run
    │   └── tglf_sat2/  the TGLF run for the same plasma  <-- Task 1
    ├── a_H/
    ├── a_T/
    └── ... (43 case folders)
```

**Case names** have the form `<letter1>_<letter2>`:

- **letter1:** is one point in a parameter scan around a standard reference case. Each letter changes some plasma parameters (gradients, geometry, collisions, …). Finding out which ones is part of Task 1.
- **letter2**: Changes a different parameter (main ion species): `H` (hydrogen), `D` (deuterium) or `T` (tritium).

| **Letters** | **Isotopes** | **Cases** |
| --- | --- | --- |
| a, b, c, d, e, f, j, x | H, D, T | 24 |
| y | H, T | 2 |
| g, h, i, k–w, z | D only | 17 |
| **Total** |  | **43** |

**The three subfolders:**

- `nonlinear/`: the expensive, high-fidelity turbulence simulation. Its time-averaged fluxes are the “truth” that TGLF tries to reproduce. Used in later tasks.
- `linear/0` … `linear/38`: cheap linear CGYRO runs, one per wavenumber. They give growth rates and frequencies of the unstable modes. Also for later tasks.
- **`tglf_sat2/`: the TGLF prediction for the same plasma.**

## Files you need now

For Task 1 you read exactly two files per case, both in `data/<case>/tglf_sat2/`. Together they make one row of a regression dataset: plasma parameters in, fluxes out. Ignore the other files in that folder for now.

| **File** | **What it is** | **Role** |
| --- | --- | --- |
| `input.tglf.gen` | Every TGLF input parameter for this run, including the defaults TGLF filled in | Inputs |
| `out.tglf.gbflux` | The predicted turbulent fluxes, normalized to gyro-Bohm units | Outputs |

### input.tglf.gen

A plain-text list with one parameter per line: a value and a parameter name. Open one in a text editor before you write anything to read it. Species are numbered: `_1`, `_2`, etc.  `NS` gives the number of species. [Do not read these from input.tglf – this is not comprehensive, since some parameters are set to defaults.]

The parameters that we are presently considering are:

| **Parameter** | **Meaning** |
| --- | --- |
| `NS` | Number of species |
| `MASS_1, MASS_2, …` | Mass (for each species); 0.5=H, 1=D, 1.5=T, ~2.72e-4 (electron) |
| `ZS_1, ZS_2, …` | Charge (for each species); positive=ion, negative=electron |
| `TAUS_1, TAUS_2, …` | Temperature (for each species) |
| `AS_1, AS_2, …` | Density (for each species) |
| `RLTS_1`, `RLTS_2, …` | Temperature gradient (for each species) |
| `RLNS_1`, `RLNS_2, …` | Density gradient  (for each species) |
| `XNUE` | Electron collision frequency |
| `RMIN_LOC` | Minor radius r/a |
| `DRMAJDX_LOC` | Derivative of major Radius dR_0/dr |
| `KAPPA_LOC` | Plasma shape parameter (elongation) |
| `Q_LOC`, `Q_PRIME_LOC` | Safety factor q and its radial derivative (magnetic shear) |

Many other lines are other parameters and numerical settings. Part of Task 1 is finding which parameters actually vary.

### out.tglf.gbflux

A single list of numbers: the “ transport fluxes” for each species.  Its length depends on the number of species, so **read NS from `input.tglf.gen` first**. The file holds 4 × NS numbers in four blocks of NS values each (one value per species, in species order):

| **Block** | **Quantity** | **Use** |
| --- | --- | --- |
| 1 | Particle flux Γ | Keep |
| 2 | Energy (heat) flux Q | Keep |
| **3** | **Toroidal momentum flux Π** | **Skip** |
| **4** | **Exchange term S** | **Skip** |

Right now every case has NS = 2, so each file has 8 numbers:

```
Gamma_1 Gamma_2 Q_1 Q_2  Pi_1  Pi_2 S_1 S_2
```

Here `_1` is for species 1 and `_2` is for species 2. Write your reader for general NS.

## Task 1: build the database and explore it

The goal is a clean table with one row per case, plasma inputs plus TGLF fluxes, and a first look at what is in it. Every later task, including training a surrogate model, starts from this table.

### Step 1: parse every case into a dictionary

1. Loop over all case folders in `data/`.
2. For each case, read `input.tglf.gen` into a dictionary with case name and parameter name → value.  You should focus only on parameters in the Table here for now.
3. Add an inferred parameter ‘species_type’ (ion-type (‘H’,’D’,’T’) or ‘electron’) based on sign of ZS_.
4. Using `NS` from input.tglf.gen, then read `out.tglf.gbflux` and split it into the four blocks described above. Keep Γ, Q and Π for each species.  Add to dictionary.

Your script should report any case that is missing a file or fails to parse, rather than skipping it silently.

### Step 2: find features that matter

1. Compare its inputs with the most common value of each column. Make a small table: case → which parameter(s) it changes, and by how much.

### Step 3: plotting summary statistics

Make some plots, each with labeled axes, e.g.

1. **Histograms** of each varying input and of Q_i, Q_e and Γ_e. Try a log axis for the heat fluxes: are they spread over orders of magnitude?
2. **Correlation matrix** (heat map) of the varying inputs and the fluxes. Which inputs correlate most strongly with Q_i? Which inputs are correlated with each other?
3. **Flux vs. drive:** Q_i against `RLTS_` and Q_e against `RLTS_1`, colored by ion species.
4. **Isotope effect:** for the letters that have H, D and T, plot Q_i against ion mass, one line per letter. Does heavier fuel give more or less transport in TGLF?
5. **Electron vs. ion heating:** histogram of Q_e / Q_i. Which cases are dominated by electron transport?

### Step 4: check the data quality

Look for anything suspicious: negative or zero heat fluxes, extreme outliers, NaN values, or cases whose inputs fall far from the rest.

## Deliverables

- `sat3_tglf_table.csv`: 43 rows, readable column names, a quality-flag column.
- Loading script or notebook.
- Results: letter → parameter table, plots, and a paragraph on what surprised you.

### Tips

- 43 cases is a small dataset. Keep that in mind when you read a correlation; one outlier can make or break it.
- The H/D/T triplets are the cleanest experiment in the dataset: only the ion mass changes.

## References

- H. G. Dudding et al., SAT3 CGYRO dataset, Zenodo record 14505760 (2024).
- H. G. Dudding et al., “A new quasilinear saturation rule for tokamak turbulence with application to the isotope scaling of transport,” Nuclear Fusion 62, 096005 (2022).
