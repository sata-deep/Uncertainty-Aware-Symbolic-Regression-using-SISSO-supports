Uncertainty-Aware SISSO: Bayesian Support Selection
This repository contains the Heusler-alloy data, deterministic SISSO outputs,

and Python code for conditional Bayesian analysis of screened symbolic supports.

The analysis provides support probabilities, feature inclusion probabilities,

and Bayesian-model-averaged predictions.

Repository layout
.
├── README.md
├── Thumbnail.png
├── requirements.txt
├── reproduce.py
├── SISSO-2D/
│   ├── train.dat
│   ├── SISSO.in
│   ├── SISSO.out
│   ├── SIS_subspaces/
│   └── Models/
├── SISSO-3D/
│   ├── train.dat
│   ├── SISSO.in
│   ├── SISSO.out
│   ├── SIS_subspaces/
│   └── Models/
├── bayes_so2/
│   ├── io.py
│   ├── models.py
│   └── utils.py
└── revision_nested_cv/
    ├── support_inference.py
    ├── symbolic.py
    ├── nested_cv.py
    ├── full_data_diagnostics.py
    ├── synthetic_test.py
    ├── run_nested_cv.sh
    ├── full_data_posterior/
    ├── runs/
    └── synthetic_test/

Dependencies
Use Python 3.9 or newer. Install the packages required by the analysis:
python -m pip install -r requirements.txt

A compatible SISSO 3.5 executable is needed only to regenerate the symbolic

feature spaces. Bayesian analysis of the supplied screened spaces does not

require a SISSO installation.
Dataset
The processed Heusler-alloy dataset is included in SISSO-2D/train.dat and

SISSO-3D/train.dat. Each file contains compound identifiers, magnetic moments,

and elemental primary features. The source materials database is cited in the

paper.
Each SISSO directory includes its input settings, deterministic output, screened

feature values, and symbolic expressions. Feature identifiers are local to each

run; use the expression strings when comparing different runs.
Run the Bayesian analysis
Run from the repository root:
python reproduce.py

This recomputes the full-data posteriors and fold-level predictions from the

archived SISSO spaces and writes the output to reproduced/.
To include the prescribed-candidate synthetic tests:
python reproduce.py --synthetic

Run fold-local SISSO validation
To regenerate the symbolic spaces and repeat the Bayesian analysis, provide

the absolute path to a compatible SISSO executable:
bash revision_nested_cv/run_nested_cv.sh \
  --sisso-executable /absolute/path/to/SISSO \
  --output-root reproduced/end_to_end

The supplied fold records are under revision_nested_cv/runs/.

Source selection and archive notes are documented in PROVENANCE.md.
Citation
If you use this code or data, please cite:
Uncertainty-Aware Symbolic Regression through Bayesian Support Selection
@misc{bhattacharjee2026uncertainty,
  title={Uncertainty-Aware Symbolic Regression through Bayesian Support Selection},
  author={Bhattacharjee, Satadeep},
  year={2026},
  eprint={2606.04042},
  archivePrefix={arXiv},
  primaryClass={cond-mat.mtrl-sci},
  url={https://arxiv.org/abs/2606.04042}
}

