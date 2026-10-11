#!/usr/bin/env bash
set -euo pipefail

LATENT_DIR="${LATENT_DIR:?set LATENT_DIR to a LightningDiT latent shard folder}"
RESULTS_DIR="${RESULTS_DIR:-results}"
NPROC="${NPROC:-8}"

torchrun --nproc_per_node="${NPROC}" train.py \
    --data-path "${LATENT_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --model SiT-XL/1 \
    --latent-size 8 \
    --level-channels 8,32,128 \
    --gap 0.25 \
    --p-l0 0.9
