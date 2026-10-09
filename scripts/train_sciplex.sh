#!/usr/bin/env bash
set -euo pipefail

primo-train-sciplex \
  --fold "${FOLD:-1}" \
  --test_size "${TEST_SIZE:-0.2}" \
  --conditional_vector_mode grca
