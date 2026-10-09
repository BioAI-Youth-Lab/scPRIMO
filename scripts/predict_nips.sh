#!/usr/bin/env bash
set -euo pipefail

primo-predict-nips \
  --config configs/nips.yaml \
  --fold "${FOLD:-1}" \
  --test_size "${TEST_SIZE:-0.2}" \
  --model_path "${MODEL_PATH:-outputs/nips/cinn/fold${FOLD:-1}_${TEST_SIZE:-0.2}_grca}" \
  --output_file "${OUTPUT_FILE:-outputs/nips/predicted.h5ad}" \
  --add_predicted_alias
