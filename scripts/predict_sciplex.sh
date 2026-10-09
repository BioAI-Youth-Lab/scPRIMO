#!/usr/bin/env bash
set -euo pipefail

primo-predict-sciplex \
  --fold "${FOLD:-1}" \
  --test_size "${TEST_SIZE:-0.2}" \
  --model_path "${MODEL_PATH:-outputs/sciplex/cinn/fold${FOLD:-1}_${TEST_SIZE:-0.2}_grca}" \
  --output_file "${OUTPUT_FILE:-outputs/sciplex/predicted.h5ad}"
