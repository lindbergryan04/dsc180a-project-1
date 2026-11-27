# dsc180a-project-1

UCSD DSC 180 group capstone (mentors at General Atomics and SDSC). We build
machine-learning surrogates for TGLF, a quasilinear plasma turbulence code.
The surrogates predict three fluxes: ion heat `Q_i`, electron heat `Q_e`, and
electron particle flux `Gamma_e`.

- **Quarter 1 (done):** compare surrogates (MLPs, ensembles, XGBoost, random
  forests) on accuracy, cost, and uncertainty, with DIII-D as a case study.
  The task specs are in `tasks/` and the report is in `quarter_1_report/`.
- **Quarter 2 (next):** a neural-network correction that moves TGLF
  predictions toward about 300 nonlinear CGYRO cases. See
  `quarter_2_proposal/`.

`README.md` covers layout and how to reproduce each person's results.
`DATA.md` covers datasets and paths. Read those before you guess.

## Layout and ownership
- `notebooks/<name>/taskN/` holds each member's work: `ryan`, `sbhatnagar`
  (Sweekrit), `jscalderon` (Joey), and `aryan`. **Edit only the folder of the
  person you are working with.** Read teammates' folders but don't change
  them. Ask if you aren't sure whose session this is.
- `src/` holds shared helpers (`tglf_parser.py`, `d3d_profile.py`). Changes
  there affect everyone, so keep them backward compatible.
- The docs (`README.md`, `DATA.md`, `REFERENCES.md`, the report `.tex`) are
  shared. Edit them when asked. Update the README when the commands or
  outputs it describes change.

## Environment
- Python 3.12 in a `.venv` at the repo root:
  `python -m pip install -r requirements.txt`. Torch is CPU only.
- Versions are pinned. Don't add or upgrade a dependency without asking.
  If one is agreed, pin it in `requirements.txt`.
- TGLF and CGYRO runs need a GACODE install (we use DSMLP). They don't work
  locally.
- Reports build with `latexmk` from their own folder. Build files go to a
  gitignored `build/`.

## Data rules
- Datasets live in a gitignored `data/` folder (see `DATA.md`). Never commit
  raw data, archives, or model weights (`*.pt`, `*.joblib`, and so on). Small
  parsed CSV tables in the notebook folders *are* committed.
- Never modify raw downloaded files. Write derived tables to a new file.
- Despite its name, the `SAT3_dataset` TGLF runs use the **SAT2** saturation
  rule.

## DS/ML practices
- **Splits:** fit any scaler, imputer, or feature selection on the training
  split only. Tune on validation and touch test only for the final numbers.
  Keep out-of-range data (`scan_db_outside`) and DIII-D data as separate
  evaluation sets, not training data.
- **Reproducibility:** fix seeds (the team uses `42`, and `42` to `46` for
  ensembles). Prefer deterministic torch and single-threaded XGBoost when
  timing or comparing. Notebooks must run top to bottom on a fresh kernel.
- **Baselines:** report a simple baseline next to every model. Report each
  flux separately, since `Q_i`, `Q_e`, and `Gamma_e` have very different
  scales.
- **Uncertainty:** when you report intervals, check their coverage on held-out
  data (for example, whether 90% intervals cover about 90%).
- **Physics sanity:** fluxes can be negative or heavy-tailed. Flag anything
  that looks odd instead of silently clipping or dropping it.
- **Numbers:** never retype metrics by hand into the report or README. Copy
  them from the saved outputs or tables. Never claim a result without running
  the code that produces it.
- **Paths:** keep paths relative to the task folder or the repo root, with an
  environment-variable override for large data (for example `SAT3_DIR`). No
  absolute paths to someone's machine.
- **Code:** put reusable logic in `.py` modules and keep notebooks for
  running and plotting. Don't duplicate helpers that already exist in `src/`.

## AI use
`AI_USE.md` is our course AI-use statement. The team chooses the models and
analyses and interprets the results. Keep work consistent with it, and tell
the user when a request would change what it says.

## Git
- Work on `main`, pull before you start, and keep commits scoped to one
  person's folder or one shared change. Commit only when asked.
