# Report draft

`report.tex` is the working checkpoint paper and `report.pdf` is its compiled
preview. Both use the supplied course template. The report loads its formatting
files from `style/`. The template source is in `DSC_Capstone_LaTeX_Template/`.

The draft includes an abstract, introduction and literature review, relevant
data, methods, preliminary results, and a discussion with remaining work.
It describes completed experiments in research-paper prose. Section 4 separates
the group's proposed model comparisons from the mentor's planned DIII-D study.
Drafting instructions are kept out of the paper.

The report centers on exploring, developing, and comparing surrogate methods.
Uncertainty is part of evaluation and DIII-D is a case study. It retains the
title Ryan accepted. The current results cover the experiments documented in
Ryan's notebooks; they do not establish a final ranking of all group methods.

## Files

| File | Purpose |
| --- | --- |
| `report.tex` | Editable report source. |
| `report.pdf` | Compiled checkpoint preview. |
| `reference.bib` | Five supplied research papers. |
| `figures/learning_curves.png` | Existing notebook figure, extracted unchanged. |
| `style/` | Unchanged course formatting files. |

## Build

From the repository root, with XeLaTeX, BibTeX, and latexmk installed:

```sh
cd report
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
```

The output is `report.pdf`. The course style uses XCharter and Arial, with
Fira Code for code text. The draft falls back to Latin Modern Mono if Fira Code
is unavailable. A minimal TeX installation may need additional packages used
by the course style. For Overleaf, upload `report.tex`, `reference.bib`,
`figures/`, and `style/`, and select XeLaTeX as the compiler.

## Evidence and review

The report uses saved notebook outputs and
records qualifications about model selection and test reuse.

Before submission, consolidate the other members' experiments, confirm author
names and the mentor-approved scope, and review all claims as a group. Code
reproducibility still needs separate work on the shared dependencies, helpers,
and model artifacts. A compiled report does not establish that the code can
be reproduced from this repository.
