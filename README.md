# GLMC: What Interpolators Store and What They Express

Code and supplementary material for

> Nikolaos Nikolaou. *What Interpolators Store and What They Express: An Information-Theoretic Framework for Benign Overfitting and Label-Free Model Selection.* 2026. ([arXiv link to be added.](https://arxiv.org/abs/2611.23630))

The supplementary material (appendices A–K of the paper, with their reference list) is in [`supplementary/glmc_supplementary_material.pdf`](supplementary/glmc_supplementary_material.pdf). Section, theorem, table and figure numbers below refer to the paper; numbers with the prefix S refer to the appendices.

This repository reproduces every figure, table and number in the paper and its appendices. Everything runs on a CPU; no GPU is needed.

## Contents

```
code/       all experiment, analysis and figure scripts (flat; scripts import each other by module name)
results/    stored results: JSON summaries and the intermediate prediction files (*.npz)
figures/    reference copies of the figures as they appear in the paper
supplementary/  the supplementary material (appendices) as a PDF
data/       fetch_pmlb.py downloads the ten PMLB datasets used in Protocol B
run_all.sh  driver script (see below)
```

## Installation

```
pip install -r requirements.txt
```

The five small datasets (`wdbc`, `digits`, `biopsy`, `OJ`, `Caravan`) load offline from scikit-learn, ISLP and pydataset. The ten PMLB datasets are only needed to rerun Protocol B from scratch:

```
python data/fetch_pmlb.py     # writes data/pmlb/<name>/<name>.tsv.gz
```

`fetch_pmlb.py` uses the repository's LFS media URL. `raw.githubusercontent.com` serves LFS pointer files, not data. Behind a TLS-intercepting proxy, set `REQUESTS_CA_BUNDLE`. To keep the data elsewhere, set `GLMC_PMLB=/path/to/pmlb`.

## Running

```
bash run_all.sh figures    # redraw all figures from stored results              (~1 minute)
bash run_all.sh analysis   # recompute all tables from stored predictions + figures (~1-2 minutes)
bash run_all.sh all        # rerun all experiments from scratch, then the above  (about 10 hours on a 2-core machine; Table S2)
```

All scripts run with `results/` as the working directory, and figures are written to `figs/`. The `analysis` mode was smoke-tested: it regenerates the stored analysis JSONs byte for byte.

## Map from paper to scripts

| Paper item | Experiment script → output | Analysis / figure script |
|---|---|---|
| Table 1 (running example) | `pilot.py` → `results.json` | |
| §7.1, Fig. 1 (synthetic validation; 120 configs) | `e1.py` → `e1_results.json` | `make_fig_e1_band.py` (left: two-sided band of Thm. 9), `make_figs.py` (right) |
| §7.2, Fig. 2 (double descent), Table S3 (width selection) | `e2_dd.py` → `e2_results.json`; `r23_real.py r3` → `r3_results.json` | `width_select.py`, `make_figs.py`, `make_figs_real.py` |
| §7.3, Fig. 3 (boosting), Fig. S3 | `e5_boost.py` → `e5_results.json`; `r23_real.py r2` → `r2_results.json` | `make_figs.py`, `make_figs_real.py` |
| §7.4, Fig. 4 (real-data ranking) | `r1_real.py` → `r1_results.json` | `make_figs_real.py` |
| Table 4 Panel A; Table S4 (Protocol A) | `r4_select.py`, `r5_select.py` (seeds 0–5) → `r4_*.npz`, `r5_*.npz` | `r4_analyse.py`, `r5_analyse.py`, `r5_consensus_check.py`, `r5_remedy.py` |
| Table 4 Panel B; Fig. 5 (Protocol B, overparameterized set; 5 seeds) | `r6_cost.py` (seed 0), `r6_seeds.py 1 2 3 4` → `r6_*_s{0..4}.npz` | `r6_overparam.py` → `r6_overparam.json`; `r6_multiseed.py` → `r6_multiseed.json`; `make_fig_overparam.py` |
| Fig. 6; Appendix §I.5, Table S5 (Protocol B, all 19 configs; 5 seeds) | `r6_cost.py`, `r6_seeds.py` | `r6_analyse.py` → `r6_analysis.json`; `r6_multiseed.py`; `make_fig_cost.py` |
| §7.6, Table 5 (min-V vs selection by e-CMI / f-CMI at matched cost); §7.7 and Appendix §I.6, Table S6 (two-stage selection: screen by training fit, then min-V) | `r8_screen_cmi.py 0 1 2 3 4` → `r8_*_s{0..4}.npz` (refits the Protocol B 2-fold models exactly, plus two further random half-splits of a fixed supersample) | `r8_analyse.py` → `r8_analysis.json`, `r8_cmi_main_table.tex` (Table 5), `r8_table.tex` (Table S6); `r8_protA.py` → `r8_protA.json` (Protocol A) |
| §4.6 / §7.1, Table 3 (disagreement vs risk) | `t_disagreement.py` → `t_disagreement.json` | (printed) |
| Fig. S1 (estimator bias) | `e3_estimator.py` → `e3_results.json` | `make_figs.py` |
| Appendix §J, Tables S7–S8, Fig. S5 (regression: storage, min-variance selection, consensus criterion) | `r9_regression.py 0 1 2 3 4` and `--batch-b` → `r9_*.npz`, `r9b_*.npz` (needs `data/pmlb_reg`, fetched by `data/fetch_pmlb.py`); `r9_exact.py` (and `--batch-b`) → `r9_exact*.json` | `r9_analyse.py` → `r9_analysis.json`; `r9_tables.py` → `r9_table_main.tex`, `r9_table_exact.tex`, `figs/r9_exact.pdf` |
| Fig. S4, §I.7 (stress tests) | `s_consensus.py` → `sc_*.npz` | `s_analyse.py`, `s_adaptive.py`, `make_fig_consensus.py` |
| Appendix §I.2 (mechanism checks) | `t_knn_check.py`, `t_ens_check.py`, `t_theory2.py` | (printed) |

Shared helpers: `pilot.py` (the running example and learners), `realdata.py` (dataset loaders), `r1_real.py` (the 19 learner configurations), `r6_analyse.py` (criteria: plurality consensus, estimation of T, asymmetry, V with Miller–Madow correction, consensus criterion).

## Key definitions in code

- **Expressed information V** (`r6_analyse.vstats`): for each unlabelled input, the entropy of the R copies' predictions (bits), plus the Miller–Madow term (k−1)/(2R ln 2), averaged over inputs. `pibar` is the mean probability of disagreeing with the mode.
- **Consensus criterion** (`r6_analyse.labelfree`): D + π̄·(1 − D·K/(K−1)), where D is the disagreement of the candidate's modal prediction with the plurality of all candidates' modal predictions.
- **Noise-corrected CV** (`r6_analyse.est_T`, `safe_inv`): T̂ is the cross-tabulation of the out-of-fold consensus against the noisy training labels, shrunk by 0.05 towards I. The corrected loss is 1 − (T̂⁻¹)[ỹ, ŷ].
- **Two-stage rule** (`r8_analyse.py`): keep candidates whose 2-fold models' training error (on their own halves, noisy labels) passes a screen (≤1%, legacy score ≥0.9, below the median, or ≤ ρ̂+0.02 with ρ̂ the disagreement of the 19-configuration out-of-fold plurality with the noisy labels), then take the least V₂ (or the consensus criterion).
- **e-CMI / f-CMI** (`r8_analyse.pair_mi`): per supersample pair, the plug-in mutual information (nats) between the pair's loss pattern (e-CMI) or prediction pattern (f-CMI) and the selection bit, over the models of all draws; averaged over pairs. Bounds: training error + √(2·e-CMI) and training error + mean over pairs of √(2·I_i). CMI conditions on the labels of both members of each pair, so it is not label-free.
- **Min-variance (regression)** (`r9_analyse.py`): mean over unlabelled inputs of ½(f₁(x) − f₂(x))² for two half-sample fits; the regression consensus criterion adds (m̂ − c)², with m̂ the mean of the two fits and c the pointwise median of all candidates' m̂, and halves the variance term.
- **Switch rule** (`r6_analyse.asym`): use noise-corrected CV if the largest deviation of T̂ from the closest symmetric noise matrix exceeds 0.1. Otherwise use the consensus criterion.

All thresholds (clustering 0.05, trigger 1/3, switch 0.1) were frozen before the Protocol B experiments.

## Notes

- The regression prediction files `r9_*.npz` and `r9b_*.npz` (74 MB) are not in the repository, to keep it small; they are distributed as `glmc_r9_regression_predictions.zip`, to be attached to a release of this repository (unzip its contents into `results/`). `r9_analysis.json` and `r9_exact*.json` are included, so `bash run_all.sh analysis` still reproduces Tables S7–S8 and Fig. S5 (it skips `r9_analyse.py` when the prediction files are absent). Alternatively, run `python code/r9_regression.py 0 1 2 3 4` and `--batch-b` from `results/` to regenerate them (about an hour).

- Seeds are fixed in every script. Results can differ in the last digit across library versions; the versions used are listed in `requirements.txt`.
- `r6_cost.py` records the CPU time of every fit and every prediction. The compute ratios in the paper are medians of these times.
