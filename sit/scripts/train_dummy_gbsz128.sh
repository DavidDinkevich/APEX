#!/usr/bin/env bash
set -euo pipefail

LATENT_DIR="${LATENT_DIR:-/common/david-temp/apex-sit-dummy-latents}"
RESULTS_DIR="${RESULTS_DIR:-/common/david-temp/apex-sit-gbsz128-dummy}"
VAE_CONFIG="${VAE_CONFIG:-../vae/configs/apex-f64c8-32-128_phase1.yaml}"
VAE_CKPT="${VAE_CKPT:-/common/david-temp/apex-vae-im256-gbsz16/checkpoints/last.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-run-apex-sit-gbsz128-dummy}"

python scripts/write_dummy_latents.py --output "${LATENT_DIR}"

CKPT_DIR="${RESULTS_DIR}/${EXPERIMENT_NAME}/checkpoints"
EXTRA=()
if compgen -G "${CKPT_DIR}/*.pt" > /dev/null; then
  LATEST="$(ls -1 "${CKPT_DIR}"/*.pt | sort | tail -1)"
  echo "RESUME ${LATEST}"
  EXTRA=(--ckpt "${LATEST}")
else
  echo "FRESH ${RESULTS_DIR}/${EXPERIMENT_NAME}"
fi

torchrun --nproc_per_node=4 train.py \
  --data-path "${LATENT_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --model SiT-XL/1 \
  --latent-size 8 \
  --level-channels 8,32,128 \
  --gap 0.25 \
  --p-l0 0.9 \
  --global-batch-size 128 \
  --epochs 1400 \
  --ckpt-every 500 \
  --wandb \
  --vae-config "${VAE_CONFIG}" \
  --vae-ckpt "${VAE_CKPT}" \
  "${EXTRA[@]}"
