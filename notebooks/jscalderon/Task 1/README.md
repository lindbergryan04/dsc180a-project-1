# Task 1 — SAT3 patterns

Final presentation: `slides/SAT3_dataset_patterns_two_slides.pptx` (two slides). The final chat revision removed the isotope slide and retained the requested temperature-gradient and safety-factor-gradient wording. Corresponding final slide PNGs are in `graphics/`.

`support/parsed.json` and `raw_cases.json` retain the analysis values. `support/` also preserves historical build scripts and final validation receipt; original absolute build paths may require adjustment. `references/` contains the groupmate example and mentor background deck. Participation Assignment 3 covers this task; Assignments 1–2 are earlier course/project background.

Source chat: **Create Sat3 dataset slides**, `01a151e1-eb46-7ea3-a711-3bc0ba174360`. Participation context: **Create two participation assignments**, `01a151fb-944e-76c3-8128-6e64f8f7bee9`.

## Reproduce the analysis

`support/analyze_sat3.py` recovers the parsing and matched-case checks originally run inline in the source chat. It needs only Python's standard library. From this task folder, run:

```sh
python3 support/analyze_sat3.py
```

This reads the locally retained `SAT3_dataset.tar`, writes `support/raw_cases.json` and `support/parsed.json`, and checks the 43 cases, both gradient scans, and eight isotope triplets. The existing presentation build/revision scripts are also retained in `support/`.
