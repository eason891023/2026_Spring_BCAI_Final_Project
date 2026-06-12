#!/bin/bash
# Phase 0 baseline grid: {split, permuted} x {baseline, scms, ncms, icms} x optimizers
# CMS-only ablation first: optimizer is held fixed at SGD unless overridden,
# so architecture effects are not confounded with optimizer effects.
# Usage: bash scripts/run_baseline_grid.sh "42 123 7" ["SGD Adam M3"] ["baseline scms"]
set -u
cd "$(dirname "$0")/.."

SEEDS=${1:-42}
OPTS=${2:-SGD}
MODELS=${3:-"baseline scms ncms icms"}
LOG_DIR="data/results/raw_log"
mkdir -p "$LOG_DIR"

for seed in $SEEDS; do
  for dataset in split permuted; do
    extra=""
    [ "$dataset" = "permuted" ] && extra="--num_tasks 10"
    for model in $MODELS; do
      for opt in $OPTS; do
        tag="${dataset}_${model}_${opt}_sd${seed}"
        echo "[$(date +%H:%M:%S)] RUN $tag"
        python3 main.py --dataset "$dataset" $extra \
          --model "$model" --optimizer "$opt" \
          --epochs 5 --seed "$seed" --save_ckpt 1 \
          > "$LOG_DIR/${tag}.log" 2>&1 \
          || echo "[$(date +%H:%M:%S)] FAILED $tag (see $LOG_DIR/${tag}.log)"
      done
    done
  done
done
echo "[$(date +%H:%M:%S)] Grid complete."
