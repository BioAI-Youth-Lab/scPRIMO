# scPRIMO

PRIMO is a paper-release codebase for LLM-guided perturbation response prediction. The release path is centered on the final old-flow GRCA implementation used in the paper experiments:

- control-derived cell context;
- control-centered residual effect statistics;
- LLM-based retrieval of functionally similar known drugs for unseen drugs;
- GRCA conditioning for cINN generation;
- scVI decoder export to predicted expression profiles.

This repository is intentionally narrower than the original experimental workspace. Baseline training scripts, notebooks, scratch artifacts, and hard-coded local experiment paths are not part of the main release.

## Layout

```text
src/primo/
  conditioning/   control PCA and residual-stat covariates
  data/           dataset/split loading and NIPS reference construction
  llm/            LLM drug-similar selector
  models/         old-flow cINN with GRCA conditioning
  predict/        NIPS/SciPlex h5ad export
  train/          scVI decoder and GRCA training entrypoints
  evaluation/     direction metrics
  utils/          standardization, split, embedding, and compatibility helpers
  vendor/         minimal PerturbNet/net2net/scVI compatibility code
configs/          example NIPS/SciPlex configs
scripts/          shell wrappers for common workflows
assets/prompts/   LLM prompt template
```

## Installation

```bash
conda env create -f environment.yml
conda activate primo
pip install -e .
```

The release has been smoke-tested in the `torch_pbnet` environment with
Python 3.10, PyTorch 1.13.1+cu117, and `scvi-tools==0.7.1`. GPU training
requires a PyTorch/CUDA build compatible with your system.

## Data and Checkpoints

Large files are not stored in Git. Place them under:

```text
data/
  nips/
  sciplex/
checkpoints/
  chemicalVAE/
  nips/scvi/
  sciplex/scvi/
artifacts/
  nips/
  sciplex/
outputs/
```

Expected ChemicalVAE files:

```text
checkpoints/chemicalVAE/model_params_525.pt
checkpoints/chemicalVAE/onehot_zinc.npy
checkpoints/chemicalVAE/mu.npy
checkpoints/chemicalVAE/std.npy
```

## Basic Workflow

Build the NIPS reference:

```bash
primo-build-nips-reference \
  --adata_path data/nips/nips_processed_min50_split82.h5ad \
  --output_h5ad outputs/nips/nips_ref_global_hvg5000_globaldeg1000.h5ad
```

Train scVI decoders:

```bash
scripts/train_scvi_nips.sh
scripts/train_scvi_sciplex.sh
```

These commands write checkpoints to `checkpoints/nips/scvi/fold{fold}_{test_size}/`
and `checkpoints/sciplex/scvi/fold{fold}_{test_size}/`, which are the default
locations consumed by the GRCA training and prediction scripts.

Train GRCA cINN:

```bash
scripts/train_nips.sh
scripts/train_sciplex.sh
```

Export predictions:

```bash
scripts/predict_nips.sh
scripts/predict_sciplex.sh
```

Evaluate direction metrics:

```bash
scripts/evaluate_nips.sh
scripts/evaluate_sciplex.sh
```

## LLM Similar Drugs

Set one of the supported API keys before generating similar-drug maps:

```bash
export OPENAI_API_KEY=...
# or ANTHROPIC_API_KEY / GEMINI_API_KEY / LLM_API_KEY
```

The prompt template is stored at `assets/prompts/llm_drug_select_prompt.txt`. Generated maps should be placed under `artifacts/nips/<test_size>/` or `artifacts/sciplex/<test_size>/`.

## Baselines

This release focuses on reproducing PRIMO/ours. External baseline predictions can be evaluated if converted to the formats accepted by `primo.evaluation`, but baseline training stacks are intentionally excluded.

## Release Notes

The files in `src/primo/models`, `src/primo/train`, and `src/primo/predict` were migrated from the final paper GRCA scripts and renamed into a clean package layout. Utility code originally inherited from `perturbnet/util.py` has been split into focused modules; `src/primo/utils/perturbnet_legacy.py` is kept only as a compatibility re-export layer. The original experimental `refactor` training pipeline is not the source of truth for this release.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
