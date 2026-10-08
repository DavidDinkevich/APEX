#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:?set CKPT to a SiT training checkpoint}"
OUT="${OUT:-sample_latents.pt}"

python sample.py \
    --ckpt "${CKPT}" \
    --output "${OUT}" \
    --model SiT-XL/1 \
    --latent-size 8 \
    --level-channels 8,32,128 \
    --gap 0.25 \
    --num-sampling-steps 250 \
    --cfg-scale 4.0
