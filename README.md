# Multimodal MRI Classification of Alzheimer's Disease: A TRIBE-Inspired Deviation Framework

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Independent research pipeline for binary classification of Alzheimer's Disease (AD) vs. Healthy Controls (HC)
> using resting-state fMRI functional connectivity and FreeSurfer hippocampal volumes from the OASIS-3 dataset.

---

## Overview

This repository contains the complete preprocessing, feature extraction, and classification code for a study investigating whether a **TRIBE-inspired deviation encoding framework** adds predictive value to structural MRI biomarkers for Alzheimer's disease classification.

**Key Finding:** Hippocampal volume alone achieves **AUC = 0.901 ± 0.079**, while functional connectivity and deviation features show **no additive value at this sample size** — a null result framed honestly and published as legitimate science.

| Feature Set | AUC | Accuracy |
|---|---|---|
| Structural (Hippocampal Volume) | **0.901 ± 0.079** | **0.827 ± 0.037** |
| FC only | 0.404 ± 0.273 | 0.425 ± 0.165 |
| FC + Structural | 0.496 ± 0.260 | — |
| Deviation + Structural | 0.496 ± 0.260 | — |
| Combined (All) | 0.451 ± 0.254 | — |

*5-fold stratified cross-validation, LDA with shrinkage='auto', random_state=42*

---

## Dataset

- **Source:** [OASIS-3](https://www.oasis-brains.org/) (Open Access Series of Imaging Studies)
- **Access:** [NITRC Image Repository](https://nitrc.org/ir) (requires signed Data Use Agreement)
- **Subjects:** 52 (29 HC, 23 AD) after FreeSurfer matching and quality control
- **Scans:** Resting-state fMRI (BOLD, TR = 2.2s) + T1w + FreeSurfer 5.3 outputs
- **Parcellation:** Schaefer-100 (71 common parcels after subject alignment)

---

## Repository Structure

```
oasis-scripts/
├── scripts/
│   ├── 01_preprocess.py               # nilearn preprocessing, Schaefer-100, TR=2.2s
│   ├── 02_feature_extraction_v3.py    # FC matrices + common parcel alignment (71 parcels)
│   ├── 03_classify_structural_f.py    # ★ Final model: LDA, FC + deviation + hippocampal
│   ├── 04_final_abletion.py           # Full ablation (5 feature sets × 4 classifiers)
│   ├── hippocampal-only-baseline.py   # Structural-only validation
│   └── Tests/
│       ├── diagnostic_leak.py         # Data-leakage diagnostic utility
│       └── merging.py                 # Feature merging utility
├── models/
│   ├── 06_baseline_classify.py        # FC-only & structural-only baselines
│   ├── 07_deviation_classify.py       # Deviation + structural multimodal
│   └── 08_ablation_study.py           # Full ablation with all feature sets
├── results/
│   ├── ablation_results.csv           # Quantitative results table
│   ├── fig_ablation_roc.pdf           # ROC curves by feature set
│   └── fig_ablation_bar.pdf           # AUC/Accuracy bar chart
├── paper/                             # Manuscript (draft stored locally, not pushed)
├── data/
│   └── download_scripts/              # Shell scripts for NITRC-IR data download
│       ├── download_oasis_scans_bids.sh
│       ├── download_oasis_freesurfer.sh
│       ├── filter_clinical.py
│       └── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── OASIS_CHANGELOG.md
└── README.md

# NEVER committed (covered by .gitignore):
# freesurfer_output/      ← FreeSurfer outputs (DUA prohibits redistribution)
# processed_timeseries/   ← Preprocessed fMRI time series
# scripts/nilearn_cache/  ← nilearn cache
# session_matchup/        ← Contains session cookies
# bold_60.csv             ← Subject IDs
# pilot_90_sessions.csv   ← Subject data
# pilot_90_subjects.csv   ← Subject data
# features/*.npy          ← Derived data linked to subject IDs
# Any .nii.gz imaging file
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/oasis-scripts.git
cd oasis-scripts

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Data Access & Download

1. Register at [nitrc.org/ir](https://nitrc.org/ir) and join the **OASIS3** project
2. Sign the Data Use Agreement (DUA) — allow 24–48 h for access
3. Use scripts in `data/download_scripts/` (see `data/download_scripts/README.md`)

```bash
# Fix Windows line endings first
tr -d '\r' < sessions.csv > sessions_unix.csv

bash data/download_scripts/download_oasis_scans_bids.sh \
    sessions_unix.csv downloaded_bold/ YOUR_NITRC_USERNAME bold

bash data/download_scripts/download_oasis_freesurfer.sh \
    freesurfer_ids.csv freesurfer_output/ YOUR_NITRC_USERNAME
```

---

## Pipeline

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `scripts/01_preprocess.py` | Schaefer-100 parcel extraction, bandpass 0.01–0.1 Hz, QC |
| 2 | `scripts/02_feature_extraction_v3.py` | FC matrices, 71-parcel alignment, HC template |
| 3 | `scripts/03_classify_structural_f.py` | **Final model**: FC + deviation + hippocampal, 4 classifiers |
| 4 | `scripts/04_final_abletion.py` | Ablation: 5 feature sets × LDA |
| 5 | `scripts/hippocampal-only-baseline.py` | Structural-only sanity check |

Run in order from the repo root:

```bash
python scripts/01_preprocess.py
# → processed_timeseries/*_timeseries.npy  (gitignored)

python scripts/02_feature_extraction_v3.py
# → features/X_baseline_raw.npy, hippocampal_features.csv, subject_index.csv  (gitignored)

python scripts/03_classify_structural_f.py
# → results/binary_results_structural.csv, fig1_*, fig2_*, fig4_*

python scripts/04_final_abletion.py
# → results/ablation_results.csv, fig_ablation_roc.*, fig_ablation_bar.*

python scripts/hippocampal-only-baseline.py
# → prints hippocampal AUC = 0.901 ± 0.079 to stdout
```

### Step-by-step details

**Step 1 — Preprocessing (`01_preprocess.py`)**
- Fetches Schaefer-100 atlas (7 Yeo networks, 2 mm resolution)
- `NiftiLabelsMasker` extracts parcel-level BOLD time series
- Bandpass filter applied manually per run (avoids `padlen` error on short scans)
- QC: rejects if < 100 timepoints, NaN values, or flat signal (std < 0.01)
- Saves `.npy` time series + `qc_log.csv`

**Step 2 — Feature Extraction (`02_feature_extraction_v3.py`)**
- Pearson correlation FC matrices → 4,950 upper-triangle features
- Intersects parcels across subjects → 71 common parcels retained
- Builds HC template (mean + std FC) for deviation computation
- Outputs: `X_baseline_raw.npy`, `X_deviation_raw.npy`, `hippocampal_features.csv`

**Step 3 — Classification (`03_classify_structural_f.py`)**
- Loads FC time series + hippocampal volume (left, right, total normalized by eTIV)
- Three feature modes: Baseline FC | Deviation FC | Combined FC+Dev
- All augmented with structural features
- 5-fold stratified CV, classifiers: LDA, SVM (linear), LR, Random Forest
- Per-fold HC template + StandardScaler (no data leakage)
- Outputs: results table CSV + confusion matrix + ROC curves + bar chart

**Step 4 — Ablation Study (`04_final_abletion.py`)**

| Condition | Features |
|-----------|----------|
| A. Structural only | Hippocampal volume only |
| B. FC only | Functional connectivity only |
| C. FC + Structural | Multimodal baseline |
| D. Deviation + Structural | TRIBE-inspired method |
| E. Combined + Structural | FC + deviation + hippocampal *(Ours)* |

**Step 5 — Structural Baseline (`hippocampal-only-baseline.py`)**
- LDA + Ledoit-Wolf shrinkage on hippocampal volume alone
- Confirms AUC = 0.901 ± 0.079

---

## Results

### Hippocampal Volume

| Group | Mean Normalized Volume | Std |
|-------|------------------------|-----|
| HC | 0.00535 | 0.00082 |
| AD | 0.00390 | 0.00071 |
| **Reduction** | **27.1%** | — |

### Ablation Study (LDA, 5-fold CV)

```
Condition                       AUC              Accuracy
────────────────────────────────────────────────────────────
A. Structural only              0.901 ± 0.079    0.827 ± 0.037
B. FC only                      0.404 ± 0.273    0.425 ± 0.165
C. FC + Structural              0.496 ± 0.260    —
D. Deviation + Structural       0.496 ± 0.260    —
E. Combined + Structural (Ours) 0.451 ± 0.254    —
```

**Interpretation:** Hippocampal volume is a strong standalone biomarker. Functional connectivity — raw or deviation-encoded — does not improve classification at N=52. This is a **null result for the deviation framework at small scale**, not a failure of methodology.

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Schaefer-100 atlas | Balance between spatial resolution and sample size (N=52) |
| 71-parcel intersection | 21 subjects had inhomogeneous parcel counts; fixed by retaining only parcels present in all subjects |
| Per-fold HC template | Prevents data leakage in deviation computation |
| LDA with `shrinkage='auto'` | PCA(20) overfits on 44 training subjects; Ledoit-Wolf shrinkage is statistically honest |
| Single StandardScaler | Double-scaling destroyed deviation z-score properties; scaler applied once in `get_features()` only |
| eTIV normalization | Left + right hippocampus normalized by estimated total intracranial volume |

---

## Troubleshooting

### `padlen` error in bandpass filtering
**Cause:** `NiftiLabelsMasker` called filter on runs with too few timepoints.  
**Fix:** Moved bandpass filter outside the masker; added per-run minimum timepoint check (< 50 TPs → skip run).

### "Illegal character" errors from download scripts
**Cause:** Windows line endings (`\r\n`) in CSV files.  
**Fix:**
```bash
tr -d '\r' < myfile.csv > myfile_unix.csv
```

### Inhomogeneous parcel counts across subjects
**Cause:** Some subjects had partial brain coverage → fewer than 100 Schaefer parcels.  
**Fix:** Compute intersection of parcels across all subjects; trim to 71 common parcels.

### Identical results across Baseline / Deviation / Combined
**Cause:** `StandardScaler` applied twice — once in `get_features()` and again in `run_cv()`.  
**Fix:** Remove scaler from `run_cv()`; scale only inside `get_features()`.

### LDA AUC worse than chance (~0.37)
**Cause:** PCA(20) before LDA compressed the feature space too aggressively on N=44 training subjects.  
**Fix:** Remove PCA; use `LDA(solver='lsqr', shrinkage='auto')` directly.

---

## Known Limitations

1. **Small sample size** (N=52) — limits generalizability and statistical power
2. **Single parcellation** — Schaefer-400 may reveal finer-grained deviation patterns
3. **Resting-state only** — task-fMRI may show stronger deviation signals
4. **Cross-sectional** — longitudinal deviation trajectories not modeled
5. **LDA linearity** — RF peak AUC=0.922 may reflect overfitting; LDA chosen for honesty

---

## Future Work

- [ ] Re-run with Schaefer-400 on full OASIS-3 sample (850 HC, 294 AD available)
- [ ] Test deviation framework on task-fMRI data
- [ ] Longitudinal modeling: deviation trajectories over multiple sessions
- [ ] External validation on ADNI or HCP-Aging
- [ ] Deep learning extension (graph neural networks on FC matrices)

---

## Paper

The manuscript is in preparation:

1. Introduction
2. Materials and Methods
3. Results
4. Discussion
5. Conclusion
6. References

**Target journal:** Frontiers in Neuroscience (Methods section)  
**Planned submission:** November 2026  
**arXiv preprint:** Simultaneous with journal submission

### Citation

```bibtex
@article{saumya2026tribe,
  title   = {Multimodal MRI Classification of Alzheimer's Disease:
             A TRIBE-Inspired Deviation Framework},
  author  = {Saumya},
  journal = {Frontiers in Neuroscience},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Data were provided by OASIS-3: Longitudinal Multimodal Neuroimaging.  
Principal Investigators: T. Benzinger, D. Marcus, J. Morris;  
NIH P30AG066444, P50AG00561, P30NS09857781, P01AG026276, P01AG003991, R01AG043434, UL1TR000448, R01EB009352.

- **NITRC-IR:** Neuroimaging Informatics Tools and Resources Clearinghouse
- **oasis-scripts:** NRG at Washington University (https://github.com/NrgXnat/oasis-scripts)
- **Schaefer Atlas:** Schaefer et al., 2018, *Cerebral Cortex*
- **TRIBE framework:** [cite original TRIBE paper]

---

## Contributors

The contributors shown on GitHub include authors of [NrgXnat/oasis-scripts](https://github.com/NrgXnat/oasis-scripts),
which this repository uses for OASIS-3 data download. The research pipeline itself was developed independently.

---

> *"Negative/null results with good methodology are accepted at Frontiers."* — Research advisor, July 2026
