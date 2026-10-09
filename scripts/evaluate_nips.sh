#!/usr/bin/env bash
set -euo pipefail

primo-evaluate-nips-direction \
  --input_type h5ad \
  --input_path "${INPUT_H5AD:-outputs/nips/predicted.h5ad}" \
  --reference_adata_path "${REFERENCE_H5AD:-outputs/nips/nips_ref_global_hvg5000_globaldeg1000.h5ad}" \
  --output_dir "${OUTPUT_DIR:-outputs/nips/direction_eval}"
