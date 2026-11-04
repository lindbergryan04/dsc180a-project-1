# Draft notes and evidence

These notes support review of the checkpoint draft. They are not part of the
paper. Numerical statements were transcribed from saved notebook outputs;
no notebook cells, training scripts, or project analysis were executed.
Cell and output indices below are **zero-based**.

## Scope and interpretation

- The project centers on exploring and developing surrogate methods, comparing
  design choices, and learning which approaches to retain or set aside.
  Uncertainty is an evaluation component. DIII-D is a case study.
- The report keeps the title Ryan accepted. The numerical comparisons currently
  come from Ryan's supplied notebooks. Add the other members' evidence before
  presenting a complete group comparison.
- Ryan reports that the group has tried gradient boosted trees and individual
  networks in addition to forests and ensembles. The supplied Task 1 notebook
  establishes a particular random-forest experiment. It does not establish the
  results of the group's other tree-model or single-network experiments.
- No method is declared generally unsuitable. The forest and spline findings
  concern the small starter dataset and the tested configurations.
- Task 1's nonlinear exploration studies relationships in TGLF outputs. It is
  not an evaluation against nonlinear CGYRO fluxes. All reported reference
  fluxes use the supplied TGLF SAT2 outputs.

## Numerical and methodological evidence

Paths in this table are relative to `../notebooks/ryan/`.

| Report content | Notebook and location | Evidence or qualification |
| --- | --- | --- |
| 43 starter cases, basic validity checks | `task1/01_load_cases.ipynb`, cells 9, 14, 48 | 43 loaded, no loading failures; 19 retained input columns, 12 varying and 7 constant; 32 OK and 11 review flags, no basic validity failures. All 43 retained. Varying columns are not independent physical variables. |
| Coupled gradients and isotope triplets | `task1/01_load_cases.ipynb`, cell 21 | Electron/ion gradients change together; eight complete H/D/T letter groups. |
| Starter model definitions and grouping | `task1/02_nonlinear_exploration.ipynb`, cell 23 | Ten physical inputs; hold out each letter group; inner five-fold grouped tuning over ridge penalties 0.1, 1, 10, 100. Fixed forest with 100 trees, depth 4, minimum leaf size 3. |
| Table 1: starter MAEs | `task1/02_nonlinear_exploration.ipynb`, cell 24, output 23 | Letter-balanced MAE, not case-level RMSE. Model families compared using the same outer scores. |
| Negative quadratic heat predictions | `task1/02_nonlinear_exploration.ipynb`, cell 25, output 4 | `b_T`: prediction -1.3369, TGLF 4.8956; `z_D`: prediction -20.7826, TGLF 6.4899. Rounded in the paper. |
| Larger data split and exclusions | `task2/01_prepare_data.ipynb`, cells 3, 5, 7-8 | 5,000 parsed; seed 42; original split 3,500/750/750. Four negative-ion-heat rows held for review, all in training; 3,496/750/750 remain. Raw rows and assignments retained. |
| Features and target transformations | `task2/01_prepare_data.ipynb`, cells 9-10, 15-17 | Eleven features, logarithmic collision frequency, derived shear; log10 heat targets and signed particle flux. Training-only scaling in the model notebooks. |
| Five-member model and baselines | `task2/02_nn_ensemble.ipynb`, cells 2-7 | Layers 11-64-64-8-3; GELU/ReLU; seeds 42-46; AdamW settings and checkpoint rule. Mean baseline in raw units; unregularized linear baseline on transformed targets. |
| Table 2: random-test performance | `task2/03_evaluation.ipynb`, cell 3, outputs 0-1 | Five-network RMSE 15.5294, 11.2706, 3.2885; R2 0.9585, 0.9646, 0.9750. RMSE reductions vs linear: 59.6%, 61.8%, 77.2%. |
| Table 3: activation comparison | `task2/04_activation_comparison.ipynb`, cell 6, output 0; cell 16, output 0 | Validation scores, not test scores, select GELU/ReLU. Mean normalized RMSE improves 4.3% over GELU/tanh; electron-heat RMSE worsens. All-ReLU has 11.5% higher aggregate error than control. |
| Matching and tuning limits | `task2/04_activation_comparison.ipynb`, cells 2, 7, 9, 16 | Matched seeds/initial weights/minibatch order. Learning rate inherited from control; alternatives not separately tuned. Five member seeds are not five independent ensembles. |
| Figure 1 and learning-curve findings | `task2/03_evaluation.ipynb`, cells 7-9; image from cell 8, output 0 | Saved PNG extracted without modification. One nested sequence, seed 2026; 250/500/1,000/2,000/3,496 cases. Mean normalized test RMSE drops 50.2% overall and 11.1% from 2,000 to 3,496. Full-training target SDs normalize all curve points. |
| Timing | `task2/03_evaluation.ipynb`, cells 5-6 | Training 24.0119 s; single-case latency 814.3493 microseconds; 750-case batch 4.5100 microseconds/case. Four threads for training, one for prediction; warm medians over seven repeats. No measured TGLF speedup ratio. |
| Starter-set network performance and failures | `task2/03_evaluation.ipynb`, cells 9-10, 13, 15, 17 | R2 0.9702/0.9973/0.9876. Directions: 25/27 gradient comparisons, 45/48 isotope comparisons. Two particle-flux sign errors among 43 cases. Starter cases were explored earlier but did not select activation. |
| Test-set reuse | `task2/03_evaluation.ipynb`, cell 0; `task2/04_activation_comparison.ipynb`, cells 3, 9 | The test set was inspected previously. Later comparisons remain exploratory even though activation selection used validation. |
| Separate 15-member uncertainty experiment | `task3/01_ensemble_uncertainty.ipynb`, cells 0, 2 | Adds ten members to an earlier five-member model; calibration uses reused validation data. Saved outputs do not display the activation identity. Do not describe it as the selected five-member model plus ten verified GELU/ReLU members, or attribute performance differences to ensemble size. |
| Interval definition | `task3/01_ensemble_uncertainty.ipynb`, cell 2 | Member sample SD with a 1% training-target SD floor; multiplier is the 676th ordered normalized residual from 750 validation cases. This is empirical exploratory coverage, not a formal guarantee. |
| Ordinary test coverage | `task3/01_ensemble_uncertainty.ipynb`, cell 5, output 2 | 91.2%, 90.1%, 91.9% on all 750 cases. The notebook plot displays only 50 cases. |
| Outside-range failure | `task3/01_ensemble_uncertainty.ipynb`, cell 7, output 2 | Collisionality group: ion-heat RMSE ratio 1.324 and member-SD ratio 0.343 relative to ordinary test. Each outside group contains 30 cases. |
| Matched DIII-D particle-flux bias | `task3/01_ensemble_uncertainty.ipynb`, cell 10, outputs 1-2; cell 13 | Same 300 inputs: predicted mean 0.995 vs TGLF 0.483. All predictions above reference; 100% interval coverage, average width 3.85. Do not confuse with the separate 10,000-draw forward-propagation result. Cause not established. |

The two model-comparison tables use different datasets, input sets, splits,
metrics, and tuning procedures. They do not establish a forest-versus-ensemble
ranking. Similarly, the separate 15-network results do not isolate the effect
of ensemble size or establish the activation configuration used in that run.

## Literature checks

All five entries in `reference.bib` were checked against the supplied PDFs.
The full surname `van de Plassche` is protected in BibTeX because the supplied
course bibliography style otherwise omits name particles in text citations.
The style files themselves are unchanged.

| Source | Verified material used in the draft |
| --- | --- |
| Meneghini et al. (2017), Sections 2-3 | Neural network TGLF/EPED1 surrogates and averaging independently initialized networks. Section 3.1 and Figure 3 (printed page 4) describe 32 TGLF networks, each with one hidden layer of 60 neurons. Their architecture and data differ from ours. |
| van de Plassche et al. (2020), Section IV | QLKNN is a neural network surrogate of QuaLiKiz. Local transport behavior matters beyond global RMSE. Our models do not implement their full training design. |
| Dudding et al. (2022), abstract and Sections 3-6 | CGYRO basis for SAT3 and isotope dependence. The course Task 1 handout establishes the use of SAT2 for the added TGLF runs. |
| Cao et al. (2026), Sections 4.2-4.3 and 5.1 | Feature engineering, transformed targets, wavenumber supervision, and component comparisons. Our work does not implement TGLF-WINN or its spectral loss. |
| Zanisi et al. (2024), Sections 3.2-3.4 | Neural network ensembles and active learning for QuaLiKiz. Our fixed-data ensemble calibration is not an ADEPT replication. |

The Meneghini paper has a count inconsistency: Figure 4's caption on printed
page 5 refers to 30 TGLF networks, while Section 3.1 and Figure 3 state 32.
The draft attributes 32 specifically to the architecture description. Its
separate EPED1 model also uses 30 networks, with two hidden layers of 10 units.
The paper does not establish that 32 is optimal for our data.

## Proposed model investigations

These directions come from Ryan's clarification of the group's interests.
They fit within the broader surrogate work but are not all prescribed by
Emily's slide. They are plans, not additional completed results.

- Extend activation comparisons beyond four combinations, one architecture,
  one split, and the inherited training schedule. The current aggregate gain
  is promising, but electron-heat performance worsened relative to the control.
- Vary ensemble member count under otherwise matched conditions. Include a
  single-network reference and report the cost of the full ensemble. Existing
  five- and fifteen-network outputs are not a controlled member-count study.
- Test alternative input transformations and representations. Distinguish
  these from transformations of the heat-flux targets. Fit learned transforms
  only on training data and select them using validation data.
- Explore different network architectures and compare shared three-output
  networks with separate ensembles for each target. Account for the total
  training and prediction cost across all three ensembles.

## Mentor's November plan

Source: `../emily_slides/110226_project.pptx`, the single slide titled
“Project: Building a DIII-D TGLF surrogate with error bars.” The case-count
interpretation is clarified by `../tasks/Task4_DIIID_database.docx`.

| Milestone | Specified work |
| --- | --- |
| Week of November 2 | Verify TGLF results at 13 DIII-D radii against supplied answers. Generate additional cases by varying five inputs around those profiles. Clean the data, including negative heat-flux cases. |
| Week of November 9 | Train a baseline and neural network for the three fluxes. Add uncertainty estimates, with network ensembles suggested as an approach. Evaluate new cases supplied as `input.tglf` files. |
| Week of November 16 | Train on 500, 1,000, 2,000, and all cases. Examine test error, uncertainty-interval width, and regions with sparse training coverage. |
| Week of November 23 | Compare the `scan_db` surrogate and the DIII-D surrogate on the same DIII-D test cases, checking accuracy and the reliability of uncertainty estimates. |
| November 30 | Final presentations. |

Task 4 specifies **500-1,000 cases per team member**, spread across the
13 radii, with at least 2,000 cases for the team. The slide's “each” does not
mean 500-1,000 cases at every radius. The five varied inputs are the separate
electron and ion temperature gradients, a shared density gradient, the
temperature ratio, and collision frequency. The geometry is taken from the
reference input at each radius. Negative particle flux is permitted; it is
not the negative heat-flux cleaning criterion.

The optional extensions are (1) a comparison against the original nonlinear
CGYRO cases, distinguishing surrogate approximation error from TGLF-versus-CGYRO
discrepancy, and (2) examining low- and high-wavenumber contributions to electron
heat flux using `out.tglf.sum_flux_spectrum`, with `k_y <= 1` as the low-k
cutoff. Neither is presented as completed in this draft.

The slide also emphasizes correlations among real plasma inputs, removal of
questionable data, uncertainty estimates for every prediction, and explicit
flags for inputs outside the training data. The method comparisons should
feed into this concrete application schedule without redefining the entire
report as a DIII-D-only or uncertainty-only project.

## Additional figure priorities

For this checkpoint, two further figures would add more than several plots
that repeat the existing tables:

1. Prediction-versus-TGLF or residual plots for all three targets, showing
   error concentration and systematic bias. Comparisons between models must
   use the same cases and comparable axes.
2. A controlled temperature-gradient or isotope comparison showing the TGLF
   reference and selected surrogate, including a documented trend failure.

A completed input-transform comparison could replace the second figure if it
isolates the transformation's effect. Ensemble-size and new architecture
figures should wait for completed experiments. Identify the model version,
dataset and split, units, and underlying experiment for every added figure.
The figures Ryan mentioned have not yet been supplied, so no new figure or
figure-dependent claim has been added to the paper in this revision.

## Work still needed

- Incorporate teammates' notebooks and results, especially gradient boosted
  trees and the additional individual-network comparisons. Confirm model
  settings, splits, metrics, and computational conditions before combining them.
- Verify the neural-network training loss against the version-matched training
  implementation before specifying it. The supplied notebook text documents
  transforms, optimizer settings, and validation selection but does not display
  the loss implementation. The draft does not assume that it was MSE.
- Confirm the final scope and measurable evaluation criteria with the mentors.
- Confirm Task 4 execution status. The planned DIII-D training database is not
  established as completed by the supplied notebooks. The 300 supplied DIII-D
  evaluation cases are a different dataset.
- Preserve new assessment cases from model selection, and record all prior
  validation/test use. An additional calibrated evaluation needs a clearly
  documented allocation of data.
- Confirm author order and mentor attribution. Sweekrit Bhatnagar's spelling
  matches the root README correction in GitHub commit `bd23a00` and the
  Week 5 presentation.
- Check the report against the group's conclusions before submission. Update
  the abstract when the combined comparisons are available. The checkpoint
  label and remaining-work subsections are appropriate to this draft; revise
  them for the final paper.
- Complete code reproducibility separately: notebook links refer to helpers,
  model files, and environments outside the shared project. Reading saved
  outputs for this draft does not verify execution from a fresh clone.
- Follow the TA's checkpoint requirements: agreed scope, substantial progress
  on both code and report, and a repository push from each team member.

## Figure provenance

`figures/learning_curves.png` is the existing PNG from
`task2/03_evaluation.ipynb`, cell 8, output 0. It was decoded from the notebook's
stored base64 output, with no rerun, replotting, or image edits.

SHA-256: `7b4a2aa5871b93aaff7a83196b590a020e85bf5a7fc4ea899d13b3b2f60f44ba`.
