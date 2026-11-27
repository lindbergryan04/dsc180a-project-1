# Task 3: Uncertainty for TGLF surrogates

## Goal

In this task you give your surrogate an error bar for every prediction, check that the error bars are honest, see what happens outside the training data, and use the surrogate to answer a question that would require many TGLF runs.

Start from your Task 2 surrogate and the same train / validation / test split.

## Step 1: add an error bar to each prediction

Extend your surrogate so that it returns a prediction and an uncertainty (e.g. a standard deviation or a 90% interval) for every case. Any method is fine. Some ideas:

- Train several copies of your model (different random seeds or different subsets of the training data) and use the spread of their predictions.
- Use a model that gives uncertainty directly, such as a Gaussian process.
- Predict a low and a high value (e.g. the 5th and 95th percentiles) instead of a single value.
- Use the errors on the validation set to set the size of the interval.

## Step 2: check the error bars

On the test set:

1. **Coverage:** what fraction of the true TGLF values fall inside your 90% intervals?
2. Does uncertainty track error? Plot the actual error |prediction − TGLF| against the predicted uncertainty. When the model says it is unsure, is it really less accurate?
3. Where is the model unsure? Which regions of input space have the largest uncertainty?

## Step 3: outside the training data

Your surrogate was trained on a fixed range of each input (see the Task 2 table). The database `scan_db_outside` holds TGLF cases that lie outside those ranges. It has one set per input: in each set, only that one input is outside its training range, and all other inputs are inside their usual ranges.

```
scan_db_outside/
├── ood_RLTS_1/
│   ├── samples.csv          the sampled input values for each case
│   └── data/
│       ├── r01/tglf_sat2/   input.tglf.gen, out.tglf.gbflux (as in scan_db)
│       ├── ...
│       └── r30/tglf_sat2/
├── ood_RLTS_2/
└── ... (9 sets, 30 cases each)
```

| **Set** | **Training range** | **Outside range** |
| --- | --- | --- |
| `ood_RLTS_1` | 1.5 – 3.5 | 3.5 – 5.0 |
| `ood_RLTS_2` | 1.5 – 3.5 | 3.5 – 5.0 |
| `ood_RLN` | 1 – 3 | 3 – 4.5 |
| `ood_TAUS_2` | 0.5 – 1.5 | 1.5 – 2.5 |
| `ood_XNUE` | 0.01 – 1 | 1 – 5 |
| `ood_Q_LOC` | 1.5 – 2.5 | 2.5 – 4.0 |
| `ood_KAPPA_LOC` | 1 – 2 | 2 – 2.5 |
| `ood_DRMAJDX_LOC` | −0.5 – 0 | −0.75 – −0.5 |
| `ood_RMIN_LOC` | 0.25 – 0.75 | 0.75 – 0.9 |

Your Task 1 code builds a table for each set (point it at `scan_db_outside/ood_RLTS_2/data`, etc.). Predict every set with your surrogate:

- How large are the errors compared with your test set?
- Does your uncertainty grow for these cases? Would your error bars have warned a user not to trust the prediction?

## Step 4: forward uncertainty quantification

Measured plasma inputs are never exact. If the measured temperature gradient is RLTS = 3.0 ± 0.3, how uncertain does that make the predicted heat flux? Answering this with TGLF would take many runs.

The reference case is a DIII-D plasma at r/a = 0.5, converted to the inputs of your database (all other TGLF settings are the same as in `scan_db`). Its TGLF result is Q_e = 3.63, Q_i = 4.31, Gamma_e = 0.387. The data are in `data_d3d`:

```
data_d3d/
├── d3d_db/tglf_sat2/        the reference case: input.tglf.gen, out.tglf.gbflux
└── d3d_uq/                  300 TGLF runs within the error bars
    ├── samples.csv          the sampled input values for each run
    └── data/
        ├── u001/tglf_sat2/  input.tglf.gen, out.tglf.gbflux
        ├── ...
        └── u300/tglf_sat2/
```

Your Task 1 code reads both (point it at `data_d3d/d3d_db` and `data_d3d/d3d_uq/data`). The inputs and their error bars:

| **Input** | **Value** | **Error bar** |
| --- | --- | --- |
| `RLTS_1` | 2.560 | ±10% |
| `RLTS_2` | 1.934 | ±10% |
| `RLNS_1 = RLNS_2` | 1.179 | ±10% (keep them equal) |
| `TAUS_2` | 0.796 | ±10% |
| `XNUE` | 0.0899 | ±20% |
| `Q_LOC` | 1.722 | ±5% |
| `Q_PRIME_LOC` | 4.606 | ±10% |
| `KAPPA_LOC` | 1.299 | fixed |
| `DRMAJDX_LOC` | −0.0836 | fixed |
| `RMIN_LOC` | 0.500 | fixed |
| `MASS_2` | 1.0 (D) | fixed |

Sample each input uniformly within its error bar, e.g. RLTS_2 between 1.741 and 2.127. Then:

1. Draw many random input sets (e.g. 10,000), each input sampled within its error bar.
2. Run all of them through your surrogate.
3. Plot the distribution of the predicted Q_i, Q_e and Gamma_e, and report the mean and spread.
4. Compare with TGLF: `data_d3d/d3d_uq` holds 300 TGLF runs sampled the same way (inputs in `samples.csv`). Does your surrogate’s distribution match TGLF’s, not just its average?

**Questions to answer:**

- A 10% uncertainty in the inputs gives what uncertainty in the fluxes?
- Which input’s uncertainty matters most? (Try giving only one input an error bar at a time.)
- How long did this take with your surrogate?

### Tips

- Fix the random seeds so your results are reproducible.
- Use the validation set, not the test set, to tune anything about your uncertainty method.

## Appendix: Amendment about the TGLF output file out.tglf.gbflux

The file holds 4 × NS numbers: four blocks of NS values, one value per species in species order. We use only the first two blocks.

| **Block** | **Quantity** | **Use** |
| --- | --- | --- |
| 1 | Particle flux Γ | Keep |
| 2 | Heat flux Q | Keep |
| 3 | Momentum flux Π | Not used (≈ 0: no rotation in these cases) |
| 4 | Energy exchange S between species | Not used (but ok to keep) |

For NS = 2 the eight numbers are: `Gamma_1 Gamma_2 Q_1 Q_2 Pi_1 Pi_2 S_1 S_2`.

Note: Earlier handouts called block 4 “Pi”; it is the energy exchange S. Gamma and Q were always correct.
