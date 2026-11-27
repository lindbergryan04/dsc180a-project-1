# TGLF Surrogate Models

Task 2: Build and test a surrogate for a TGLF database

## Goal

TGLF takes seconds per case; a trained surrogate takes microseconds. In this task you build one surrogate that predicts the TGLF fluxes Q_i, Q_e and Gamma_e from the plasma inputs, compare it with a simple baseline, and test it on the SAT3 cases from Task 1.

## The new database

The SAT3 database was a scan: one parameter changed at a time around a reference case. The new database scan_db has 5000 cases and starts from the same reference case (a_D) but changes **all inputs at once, at random**, over the same ranges.

| **Input** | **Range** | **Notes** |
| --- | --- | --- |
| RLTS_1, RLTS_2 | 1.5 – 3.5 | Electron and ion temperature gradients, varied separately |
| RLNS_1 = RLNS_2 | 1 – 3 | Density gradient (equal for both species) |
| TAUS_2 | 0.5 – 1.5 | Temperature ratio T_i/T_e |
| XNUE | 0.01 – 1 | Collision frequency, sampled on a log scale |
| Q_LOC | 1.5 – 2.5 | Safety factor q |
| Q_PRIME_LOC | varies | Magnetic shear term; depends on Q_LOC and RMIN_LOC |
| KAPPA_LOC | 1 – 2 | Elongation (plasma shape) |
| DRMAJDX_LOC | −0.5 – 0 | Shift of the flux surfaces |
| RMIN_LOC | 0.25 – 0.75 | Minor radius r/a |
| MASS_2 | 0.5, 1, 1.5 | Ion mass: H, D or T |

Each case is a folder `data/r0001/tglf_sat2` … `data/r5000/tglf_sat2` using the same two files as before (input.tglf.gen for inputs and out.tglf.gbflux for outputs), so your Task 1 scripts should be able to build the table.

## Step 1: prepare the data

1. Build the table of 5,000 cases with your Task 1 code. Check the input histograms.  They should now be roughly flat across each range, unlike the SAT3 scan.
2. Look at the outputs and range of scales. Decide whether to predict Q directly or log10(Q).  Catch errors (e.g. Q < 0).  Particle fluxes Γ can be positive or negative.
3. Choose your features, e.g. use log10(XNUE), drop columns that never change or duplicate another.
4. **Split the data once, at the start:** 70% training, 15% validation, 15% test, with a **fixed random seed**. Use the validation set to choose model settings.

## Step 2: baselines

Every surrogate must beat some type of baseline that you choose, e.g.:

- Mean: always predict the average flux.
- Linear/polynomial regression on all features.

## Step 3: build a surrogate

- Choose some type, e.g. a tree ensemble (random forest, gradient boosting), a Gaussian process or a neural network, and explain your choice.
- Tune the model on the **validation set**.

## Step 4: statistics for the surrogate

1. **Accuracy:** test-set RMSE and R² for the baseline and the surrogate, for each output, in one table. Parity plots (predicted vs. true) for both.
2. **Speed:** training time, and prediction time per case. Compare with TGLF’s seconds per case.
3. **Learning curve:** retrain the baseline and the surrogate on 250, 500, 1,000, 2,000 and all ~3,500 training cases, and plot test error against training size. Does the surrogate keep improving with more data? Does the baseline? Would generating more TGLF cases help?

## Step 5: test on the SAT3 cases

The 43 SAT3 cases were made a different way (a scan, not random sampling), so they are a true outside test.

1. Predict all 43 with your surrogate and compare error with test set error.
2. Scans might be helpful, e.g. for the SAT3 temperature-gradient scan (cases b, c, a, d) and the isotope triplets (H, D, T), could plot TGLF’s flux and your surrogate’s prediction on the same axes to see if they give the same trends.

### Tips

- **Fix the random seed for the split and for every model, so results are reproducible.**
- Might want to scale the inputs (`StandardScaler`) for linear models, Gaussian processes and neural networks.
- Smooth, physically sensible trends matter as much as a low average error.
