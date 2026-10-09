#!/usr/bin/env bash
set -euo pipefail

primo-train-scvi \
  --dataset nips \
  --config configs/nips.yaml \
  --fold "${FOLD:-1}" \
  --test_size "${TEST_SIZE:-0.2}" \
  --n_latent "${N_LATENT:-20}" \
  --n_hidden "${N_HIDDEN:-256}" \
  --n_layers "${N_LAYERS:-2}" \
  --epochs "${EPOCHS:-500}" \
  --n_epochs_kl_warmup "${KL_WARMUP:-300}" \
  --early_stopping \
  --es_save_best_state
