#!/usr/bin/env bash
set -euo pipefail

primo-train-scvi \
  --dataset sciplex \
  --config configs/sciplex.yaml \
  --fold "${FOLD:-1}" \
  --test_size "${TEST_SIZE:-0.2}" \
  --n_latent "${N_LATENT:-10}" \
  --n_hidden "${N_HIDDEN:-128}" \
  --n_layers "${N_LAYERS:-1}" \
  --epochs "${EPOCHS:-400}"
