# CGYRO check: surrogate error versus TGLF's difference from nonlinear CGYRO

The optional extension in the Tasks 5–7 handout asks whether a surrogate's error is small
compared with the difference between TGLF and nonlinear CGYRO. The 43 SAT3 cases
(Dudding et al. 2022) have both: a nonlinear CGYRO simulation and a TGLF (SAT2) run made
from the same inputs. The surrogate is the Task 2 five-network ensemble, trained on
`scan_db` (TGLF only) and never fitted to these cases.

```sh
python notebooks/ryan/cgyro_check/cgyro_check.py
```

Inputs: the SAT3 archive extracted to `data/Task1_TGLF_SAT3_dataset/SAT3_dataset/` and
`sat3_surrogate_predictions.csv`. That file holds the saved Task 2 ensemble's predictions
for the 43 cases, from the verified saved models in the portable Task 2 package
(`lindbergryan04/DSC180A-TGLF-Surrogate`, `results/sat3/predictions.csv`, model
`nn_ensemble_equal5`). `results/` holds the committed output.

**CGYRO fluxes.** For each case the script reads `bin.cgyro.ky_flux` (float32; species ×
moment × field × toroidal mode × time), sums over modes and both fields, and averages over
the second half of the run (time ≥ half the final time, about 500 a/c_s). `cgyro_block_se`
is the standard error of four block means in that window, a rough averaging uncertainty
(median 2–3% of the flux). One run (`t_D`) has one flux record more than its time file; as
in `pygacode`, the extra record is ignored.

**Units.** Both codes report gyro-Bohm units with the same reference: deuterium mass (CGYRO
`m_norm` is deuterium for H, D and T cases alike), electron temperature, and `B_unit` (the
TGLF inputs use the default `UNITS = GYRO`). The script checks the CGYRO reference mass.

The comparison is limited to these 43 idealized scan cases around one reference condition
(r/a 0.25–0.75, elongation 1–2, safety factor 1.5–2.5).
It shows how large TGLF's own model error is for these cases; it does not measure the
surrogate's accuracy for DIII-D shot 128913.
