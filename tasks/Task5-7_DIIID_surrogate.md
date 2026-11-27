# Tasks 5–7: A DIII-D TGLF Surrogate with Error Bars

November 9 – 30

## Goal

In Task 4 your team built a database of TGLF runs around one DIII-D plasma. In the next three weeks you turn it into a surrogate that predicts the heat and particle fluxes from the inputs, with an error bar on every prediction. Then you ask two questions that matter for anyone building such a surrogate:

- How many TGLF runs are enough?
- Does it work for a different DIII-D discharge, and if not, what does it need to learn to be more general?

| **Week of** | **Part** |
| --- | --- |
| Nov 9 | A. Surrogates with error bars |
| Nov 16 | B. How many TGLF runs are enough and when does the model overfit? |
| Nov 23 | C. Predicting a second DIII-D discharge |
| Nov 30 | Final Presentations |

## The data

| **Data** | **What it is** |
| --- | --- |
| Your team’s DIII-D database | All team members’ cases from Task 4, combined into one table. This is your training data. |
| `d3d_profile_test/` | TGLF inputs (`input.tglf`) at 12 new radii between the 13 profile radii, with their TGLF answers (`d3d_profile_test_answers.dat`). Check your predictions against the answers, or run TGLF on them yourself. |
| `d3d_profile_second/` | A second DIII-D L-mode discharge at the same 13 radii, with its TGLF answers. |

**Inputs (features):** the five inputs you varied (`RLTS_1`, `RLTS_2`, `RLNS_1`, `TAUS_2`, `XNUE`) plus the radius r/a (`RMIN_LOC`). In this database, the shape and q inputs change only with the radius, so r/a stands in for them.

**Outputs:** Q_e, Q_i and Gamma_e. Remove any bad cases (e.g. negative Q).

## Part A: Surrogates with error bars

1. Split your database into a training set and a validation set.
2. Fit a baseline and a surrogate model for each output.
3. Give every prediction an error bar, with a method that you choose.
4. Check whether your error bars are honest on the validation set.
5. Make a varied test set: at each of the 12 test radii in `d3d_profile_test`, make 20+ cases with the same random variations of the five inputs as in Task 4 (same factor ranges), and run TGLF on them. Your Task 4 script does this if you point it at `d3d_profile_test` and give it a new seed and output folder. Never train on these cases.
6. Predict Q_e, Q_i and Gamma_e, with error bars, for the 12 test radii and for your varied test set.
7. Questions to answer: How much better is the surrogate model than the baseline? Are your error bars too small, too large, or about right? Which output is hardest to predict, and where (which radii)?

## Part B: How many TGLF runs are enough and when does the model overfit?

1. Make a learning curve: train on growing random subsets of your database and measure the training error, the validation error and the error on the test cases.
2. Find where the model overfits: for a few database sizes.
3. Find where the data is too thin: how do the error and the error bar depend on the radius, and on each of the five varied inputs? Are the edges of the ranges worse?

**Questions to answer:** Is the error still going down at the full size of your database? Roughly how many more TGLF runs would you need to halve it? How large a network can your data support?

Keep in mind that TGLF itself is not exact: changing its numerical settings changes the fluxes by about 5% in this plasma. A surrogate error at that level is about as good as the data allows.

## Part C: Predicting a second DIII-D discharge

Can the surrogate work beyond the plasma it was built from but same type of scenario? We give you a second DIII-D discharge at the same 13 radii.

1. Predict the fluxes of the second discharge with your surrogate from Part A, and compare with the answers at each radius. Where does it work, and where does it fail?
2. Wider ranges: compare the second discharge’s inputs with the ranges in your database. Widen the ranges of your five inputs where needed and predict again. What improved, and what didn’t?
3. Add q and the shear: In your new cases, also vary the safety factor q and the magnetic shear, and add them as features. The template has an optional block for this (see `tools/README.txt`). Retrain and predict again.

**Questions to answer:** Why did (or didn’t) widening the ranges fix it? What does a surrogate need to know to work for a new discharge? Did your error bars warn you where the predictions were wrong?

## Optional Extensions

- **Ion-scale vs. electron-scale turbulence.** TGLF also writes the flux for each wavenumber k_y. The script `tglf_flux_spectrum.py` separates the flux into an “ion-scale” and “electron-scale” component. How much of Q_e comes from electron-scale turbulence (k_y > 1), and how does that change with radius? Does separating the flux in this way help?
- **CGYRO check.** We can give you the fluxes from the original nonlinear CGYRO simulations behind the SAT3 database (Task 1). Compare them with your `scan_db` surrogate and with TGLF itself. Is your surrogate’s error small compared with the difference between TGLF and CGYRO?
