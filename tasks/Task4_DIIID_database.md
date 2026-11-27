# Task 4: Building a DIII-D TGLF database

## Goal

This week you run TGLF yourselves on a real DIII-D plasma experiment and build your own database around that plasma. In the following weeks you will train a surrogate on it and compare it with your `scan_db` surrogate.

## The data

We have provided equilibrium data for DIII-D experiment #128913 at 13 radii (r/a = 0.20 to 0.80) spanning the plasma.  Some inputs for magnetic “shaping” have been expanded from previous (`scan_db`) for more realistic shape.  The data still has 1 electron species and 1 ion species (deuterium).  The outputs of interest are still Q_i, Q_e, and Gamma_e.

```
d3d_profile/
├── r020/tglf_sat2/input.tglf    RMIN_LOC: r/a = 0.20
├── r025/tglf_sat2/input.tglf    RMIN_LOC: r/a = 0.25
├── ...
├── r075/tglf_sat2/input.tglf    RMIN_LOC: r/a = 0.75
└── r080/tglf_sat2/input.tglf    RMIN_LOC: r/a = 0.80
```

Everything comes in `task4_d3d.tar.gz`. After you unpack it, run every command in this task from inside the `task4_d3d` folder. It contains `d3d_profile/` (above, plus our answers in `d3d_profile/d3d_profile_answers.dat`) and `tools/` (the scripts for Step 2).

## Step 1: Run the DIII-D profile and check your answers

1. Run TGLF on all 13 radii (see “Running TGLF” at the end).
2. Read each `out.tglf.gbflux` with your Task 1 code and compare with the values in the table in d3d_profile/d3d_profile_answers.dat. They should agree to at least 3 digits. **If the answers at any radius do not agree, then stop and ask us before going forward** (i.e. something is wrong with the installation or the input).
3. Note how many seconds one TGLF run takes on your computer, and how many cores you have. You need this to plan Step 2.
4. Plot Q_i, Q_e and Gamma_e against r/a. This is the DIII-D flux profile your surrogates will be tested on.

## Step 2: Build the DIII-D database

Now make new TGLF cases that look like this DIII-D plasma, and run TGLF on them. **How you write your code is up to you, but every case must be built with the same recipe**, given in the table below, so that all the databases are comparable. Each input is set in one of four ways (the physical name first, then the name used in input.tglf):

| **What it is** | **TGLF name** | **How it is set in each new case** |
| --- | --- | --- |
| **1. Chosen by you** |  |  |
| Radius of the case (minor radius r/a) | `RMIN_LOC` | one of the 13 radii, r020 … r080 (spread your cases evenly over them) |
| **2. Taken from the DIII-D input at that radius, unchanged** |  |  |
| Safety factor q | `Q_LOC` | value at that radius |
| Magnetic shear (how fast q changes with radius) | `Q_PRIME_LOC` | value at that radius |
| Elongation κ, and how it changes with radius | `KAPPA_LOC`, `S_KAPPA_LOC` | value at that radius |
| Triangularity δ, and how it changes with radius | `DELTA_LOC`, `S_DELTA_LOC` | value at that radius |
| Major radius R0/a, and the Shafranov shift | `RMAJ_LOC`, `DRMAJDX_LOC` | value at that radius |
| **3. Value at that radius × a random factor** |  |  |
| Electron temperature gradient a/L_Te | `RLTS_1` | × f_Te, between 0.6 and 1.4 |
| Ion temperature gradient a/L_Ti | `RLTS_2` | × f_Ti, between 0.6 and 1.4 (drawn separately from f_Te) |
| Density gradient a/L_n (the same for electrons and ions) | `RLNS_1` = `RLNS_2` | × f_n, between 0.6 and 1.4 (one factor for both) |
| Ion-to-electron temperature ratio T_i/T_e | `TAUS_2` | × f_tau, between 0.75 and 1.25 |
| Electron collision frequency | `XNUE` | × f_nu, between 0.5 and 2 (draw log10(f_nu) uniformly) |
| **4. The same in every case (and the same as in scan_db)** |  |  |
| Species: electrons and deuterium (mass, charge) | `NS`, `ZS_1`, `ZS_2`, `MASS_1`, `MASS_2`, … | unchanged |
| Plasma pressure β, rotation, impurity content | `BETAE`, `VEXB_SHEAR`, `ZEFF`, … | unchanged |
| TGLF model settings | `SAT_RULE`, … | unchanged |

So each case is the real DIII-D plasma at one of the 13 radii, with the temperature and density gradients changed by up to ±40%, and T_i/T_e and the collision frequency changed as well. These are the quantities a transport code changes when it predicts the density and temperature profiles, so this is the data a surrogate for DIII-D needs.

**Example.** For a case at r050, start from `d3d_profile/r050/tglf_sat2/input.tglf`. If it has `RLTS_2` = 2.50 and you draw f_Ti = 1.15, your case gets `RLTS_2` = 1.15 × 2.50 = 2.88. Do the same for the rest of group 3 (with its own factors), and leave everything else as it is.

### What to run

**Each team member make 500-1000 cases**, spread evenly over the 13 radii, so a team has at least 2000 cases. Before you start, estimate the run time from Step 1: (number of cases) × (seconds per run) ÷ (number of cores). For example, 1000 cases at 10 s per run on 8 cores take about 20 minutes, and about 150 MB of disk.

**Make sure your cases are different from your teammates’.** For example, if you draw random numbers with a seed, give each team member a different seed (e.g. 1, 2, 3, 4).

However you organize your code, the steps are:

1. Write a script that makes your cases with the recipe above. For each case: choose one of the 13 radii, start from its input.tglf, multiply the group 3 inputs by their five random factors, and write the new input.tglf in its own folder (e.g. `d3d_mydb_1/data/c0001/tglf_sat2/input.tglf`). Record the radius and the factors you drew.
2. Check one case before running them all: compare a new input.tglf with the input.tglf of its radius (e.g. with `diff`). Only the six group 3 inputs (`RLTS_1`, `RLTS_2`, `RLNS_1`, `RLNS_2`, `TAUS_2`, `XNUE`) should differ, and they should be inside the ranges listed in `d3d_profile_answers.dat` for that radius.
3. Run TGLF on all your cases.

**Helper scripts.** We provide scripts in `tools/` that do most of this:

- `d3d_profile.py` reads the 13 radii and lets you change inputs in their input.tglf text
- `make_db_template.py` is a template for the case-making script with three TODOs to fill in
- `run_tglf_parallel.py` runs TGLF on many cases at once.

You can these scripts as they are, change them, or write your own. `tools/README.txt` explains how to use them.

### Collect the results

1. Build a table of your cases with your Task 1 code (point it at your case folder, e.g. `d3d_mydb_1/data`), and add the radius and the random factors you recorded.
2. Combine the tables of all team members into one database. Check that no two cases are the same.
3. Plot the input histograms and compare with `scan_db` to see where the DIII-D database sits inside the `scan_db` box.

### Looking ahead: your DIII-D surrogate

In the coming weeks you will train surrogate models on your team’s database.

- **Features:** the five inputs you varied (`RLTS_1`, `RLTS_2`, `RLNS_1`, `TAUS_2`, `XNUE`) plus the radius r/a. The geometry inputs change only with radius in this database, so r/a stands in for them.
- **Outputs:** Q_e, Q_i and Gamma_e.
- **Testing:** we will give you test cases (input.tglf files) for which your surrogate has to predict the fluxes.
- **How many cases are enough?** You will train on growing parts of your database (e.g. 500, 1000, 2000 cases, all of it) and check whether the test error is still going down.

## Running TGLF

- TGLF runs in a folder that contains `input.tglf`. From the folder above it:
   1. >> `tglf -e tglf_sat2`
   2. The outputs appear next to the input.
- For Step 1 you can run the 13 radii one at a time, or all at once with:

```
>> python tools/run_tglf_parallel.py d3d_profile -nproc 4
```

- A case is finished when `out.tglf.gbflux` exists.
