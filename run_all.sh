#!/usr/bin/env bash
# Reproduce the results of "What Interpolators Store and What They Express".
# Usage: bash run_all.sh [figures|analysis|all]
#   figures  : redraw every figure from the stored results (minutes)
#   analysis : recompute every table/number from the stored predictions (*.npz) and JSONs, then redraw figures
#   all      : rerun every experiment from scratch (~10 h wall-clock on 2 cores), then analysis and figures
set -euo pipefail
MODE="${1:-figures}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/results"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
ln -sfn "$ROOT" paper            # scripts write figures to paper/figs/ -> ROOT/figs
mkdir -p "$ROOT/figs"
py() { echo ">> $*"; python "$ROOT/code/$@"; }

if [[ "$MODE" == "all" ]]; then
  py pilot.py 20                 # Table 1 (results.json)
  py e1.py 30                    # Sec. 7.1 (e1_results.json)
  py e2_dd.py 40                 # Sec. 7.2 synthetic double descent
  py e3_estimator.py             # Supp. I.1 estimator bias
  py e5_boost.py 30              # Sec. 7.3 boosting
  py r1_real.py 12               # Sec. 7.4 real-data ranking
  py r23_real.py both            # real boosting (r2) and real double descent (r3)
  for s in 0 1 2 3 4 5; do py r4_select.py 12 $s; py r5_select.py 12 $s; done   # Protocol A
  py s_consensus.py              # Supp. I.6 stress tests
  py r6_cost.py                  # Protocol B, seed 0 (needs data/pmlb; see data/fetch_pmlb.py)
  py r6_seeds.py 1 2 3 4         # Protocol B, seeds 1-4
  py t_disagreement.py           # Sec. 7.1 disagreement table (Table 3)
  py t_knn_check.py; py t_ens_check.py; py t_theory2.py                        # Supp. I.2 checks
  py r9_regression.py 0 1 2 3 4; py r9_regression.py 0 1 2 3 4 --batch-b; py r9_exact.py; py r9_exact.py --batch-b   # Supp. J regression (~80 min)
  py r8_screen_cmi.py 0 1 2 3 4  # Sec. 7.6 (CMI) and Supp. I.6 (two-stage) (~2.2 h on 2 cores; reuses the Protocol B data)
fi
if [[ "$MODE" == "all" || "$MODE" == "analysis" ]]; then
  py width_select.py             # Sec. 7.2 width selection / Supp. Table S2
  py r4_analyse.py; py r5_analyse.py; py r5_consensus_check.py; py r5_remedy.py  # Protocol A tables
  py s_analyse.py; py s_adaptive.py                                             # stress tests
  py r6_analyse.py; py r6_overparam.py; py r6_multiseed.py                      # Protocol B, 5 seeds (Table 4B, Supp. Table S5, Figs. 5-6)
  if ls r9_*_s0.npz >/dev/null 2>&1; then py r9_analyse.py; fi; py r9_tables.py                                            # Supp. Tables S7-S8, Fig. S5 (regression)
  py r8_analyse.py; py r8_protA.py                                              # Table 5 (CMI) and Supp. Table S6 (two-stage)
fi
py make_figs.py; py make_fig_e1_band.py; py make_figs_real.py; py make_fig_consensus.py; py make_fig_cost.py; py make_fig_overparam.py
echo "Figures written to $ROOT/figs"
