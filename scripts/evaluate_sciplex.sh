#!/usr/bin/env bash
set -euo pipefail

primo-evaluate-sciplex-direction \
  --input_h5ad_path "${INPUT_H5AD:-outputs/sciplex/predicted.h5ad}" \
  --reference_adata_path "${REFERENCE_H5AD:-data/sciplex/sciplex_processed.h5ad}" \
  --output_dir "${OUTPUT_DIR:-outputs/sciplex/direction_eval}" \
  --fold "${FOLD:-1}"
