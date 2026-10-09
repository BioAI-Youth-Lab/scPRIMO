#!/usr/bin/env python3
"""
Generate SciPlex prediction h5ad for the PRIMO/GRCA cINN model.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np
import scvi
import torch

from primo.vendor.perturbnet.chemicalvae.chemicalVAE import ChemicalVAE
from primo.vendor.perturbnet.pytorch_scvi.scvi_generate_z import scvi_predictive_z
from primo.conditioning.enhanced_covariates import build_unseen_enhanced_cov_with_similars
from primo.utils.perturbation_embeddings import prepare_embeddings_cinn
from primo.utils.splits import load_kfold_split_testsize
from primo.utils.standardize import StandardizeLoad

from primo.predict.generator import SCVIZ_CheckNet2Net
from primo.models.sciplex_flow import (
    ConditionalFlatCouplingFlow,
    Net2NetFlow_TFVAE_Covariate_Flow,
)
from primo.conditioning.sciplex import (
    infer_sciplex_covariate_layout,
    load_cached_train_artifacts,
)
from primo.paths import ARTIFACT_DIR, CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SciPlex predicted h5ad from PRIMO/GRCA checkpoints."
    )
    p.add_argument(
        "--adata_path",
        type=str,
        default=str(DATA_DIR / "sciplex" / "sciplex_processed.h5ad"),
    )
    p.add_argument(
        "--split_dir",
        type=str,
        default=str(DATA_DIR / "sciplex" / "splits_82"),
    )
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--good_smile_key", type=str, default="good_smile_chemvae")
    p.add_argument("--good_smile_value", type=int, default=1)

    p.add_argument("--cell_type_col", type=str, default="cell_type")
    p.add_argument("--treatment_col", type=str, default="treatment")
    p.add_argument("--dose_col", type=str, default="dose")
    p.add_argument("--control_col", type=str, default="product_name_clean")
    p.add_argument("--control_label", type=str, default="Vehicle")
    p.add_argument("--perturbation_col", type=str, default="product_name_clean")
    p.add_argument("--perturbation_key", type=str, default="perturb_string_processed")
    p.add_argument("--trt_key", type=str, default="ordered_all_smile_chemvae")
    p.add_argument("--embed_key", type=str, default="ordered_all_onehot_chemvae")

    p.add_argument(
        "--scvi_model_dir_template",
        type=str,
        default=str(CHECKPOINT_DIR / "sciplex" / "scvi" / "fold{fold}_{test_size}"),
    )
    p.add_argument("--scvi_model_dir", type=str, default="")
    p.add_argument(
        "--chemvae_model_path",
        type=str,
        default=str(CHECKPOINT_DIR / "chemicalVAE" / "model_params_525.pt"),
    )
    p.add_argument(
        "--chem_onehot_path",
        type=str,
        default=str(CHECKPOINT_DIR / "chemicalVAE" / "onehot_zinc.npy"),
    )
    p.add_argument(
        "--std_param_dir",
        type=str,
        default=str(CHECKPOINT_DIR / "chemicalVAE"),
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Checkpoint dir. Empty means resolved from run_config or default template.",
    )
    p.add_argument(
        "--model_path_template",
        type=str,
        default=str(OUTPUT_DIR / "sciplex" / "cinn" / "fold{fold}_{test_size}_grca"),
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Directory containing train_fit/enhanced_cov cache files.",
    )
    p.add_argument(
        "--similars_map_path",
        type=str,
        default="",
        help="LLM similars map JSON path. Empty means artifacts/sciplex/{test_size}/similars_map_fold{fold}.json.",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default=str(OUTPUT_DIR / "sciplex" / "predicted.h5ad"),
    )
    p.add_argument(
        "--output_layout",
        type=str,
        default="lite",
        choices=["lite", "full"],
        help=(
            "lite: only keep obs columns required by direction eval plus var_names "
            "and prediction layers; full: copy full adata_test before attaching predictions."
        ),
    )
    p.add_argument(
        "--prediction_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )
    p.add_argument(
        "--max_cells_per_chunk",
        type=int,
        default=512,
        help="Maximum number of cells to decode per chunk within a treatment group.",
    )
    p.add_argument(
        "--similars_weight_mode",
        type=str,
        default="linear",
        choices=["linear", "threshold_uniform_kept", "threshold_ceil_sqrt_ceil"],
        help="How candidate scores are transformed into weights for selected_drugs entries.",
    )
    p.add_argument(
        "--similars_score_threshold",
        type=float,
        default=0.0,
        help="Minimum raw score retained by threshold-based similars weight modes.",
    )
    p.add_argument(
        "--add_predicted_alias",
        action="store_true",
        help="Write legacy layer 'predicted' in addition to 'prediction'.",
    )
    p.add_argument(
        "--no_add_predicted_alias",
        action="store_false",
        dest="add_predicted_alias",
    )
    p.set_defaults(add_predicted_alias=True)

    p.add_argument(
        "--library_source",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    p.add_argument(
        "--deterministic_decode",
        action="store_true",
        help="Use px_rate instead of sampling from the decoder.",
    )
    p.add_argument(
        "--strict_cond_alignment",
        action="store_true",
        help="Fail-fast when conditional builder falls back unexpectedly.",
    )
    p.add_argument(
        "--conditional_vector_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "full", "simple", "adapter", "grca"],
    )
    p.add_argument("--prediction_seed", type=int, default=2025)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA unavailable, falling back to CPU.")
        return "cpu"
    return device_arg


def _load_model_run_config(model_path: str) -> Dict[str, Any]:
    run_cfg_path = Path(model_path) / "run_config.json"
    if not run_cfg_path.exists():
        return {}
    try:
        return json.loads(run_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    return args.model_path_template.format(fold=args.fold, test_size=args.test_size)


def _resolve_scvi_model_dir(args: argparse.Namespace, run_cfg: Dict[str, Any]) -> str:
    if args.scvi_model_dir:
        return args.scvi_model_dir
    if run_cfg.get("scvi_model_dir"):
        return str(run_cfg["scvi_model_dir"])
    return args.scvi_model_dir_template.format(fold=args.fold, test_size=args.test_size)


def _resolve_similars_map_path(args: argparse.Namespace) -> str:
    if args.similars_map_path:
        return args.similars_map_path
    candidates = [
        str(ARTIFACT_DIR / "sciplex" / f"{args.test_size}" / f"similars_map_fold{args.fold}_per_target.json"),
        str(ARTIFACT_DIR / "sciplex" / f"{args.test_size}" / f"similars_map_fold{args.fold}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _resolve_conditional_vector_mode(
    mode_arg: str,
    run_cfg: Dict[str, Any],
) -> str:
    mode = mode_arg
    if mode == "auto":
        cfg_mode = run_cfg.get("conditional_vector_mode", "grca")
        if isinstance(cfg_mode, bool):
            mode = "full" if cfg_mode else "none"
        else:
            mode = str(cfg_mode)

    alias = {
        "none": "none",
        "concat": "none",
        "raw": "none",
        "false": "none",
        "full": "full",
        "residual": "full",
        "true": "full",
        "simple": "simple",
        "adapter": "adapter",
        "grca": "grca",
        "gated_residual_cell_adapter": "grca",
    }
    return alias.get(str(mode).strip().lower(), "none")


def _resolve_drug_proj_from_run_cfg(
    run_cfg: Dict[str, Any],
    chemvae_dim: int,
) -> tuple[str, int]:
    mode = str(run_cfg.get("drug_proj_mode", "none")).strip().lower()
    alias = {
        "none": "none",
        "off": "none",
        "false": "none",
        "linear": "linear",
    }
    mode = alias.get(mode, "none")
    if mode == "linear":
        out_dim = int(
            run_cfg.get(
                "drug_proj_output_dim",
                run_cfg.get("drug_proj_dim_requested", chemvae_dim),
            )
        )
        if out_dim <= 0:
            raise ValueError("run_config 中的 drug_proj_output_dim 非法。")
        return mode, out_dim
    return "none", int(chemvae_dim)


def _resolve_cache_dir(args: argparse.Namespace, run_cfg: Dict[str, Any]) -> str:
    if args.cache_dir:
        return args.cache_dir
    if run_cfg.get("cached_cov_dir"):
        return str(run_cfg["cached_cov_dir"])
    raise ValueError("cache_dir is required when run_config.json does not contain cached_cov_dir.")


def _load_std_model(std_param_dir: str, device: str) -> StandardizeLoad:
    mu_std = np.load(os.path.join(std_param_dir, "mu.npy"))
    std_std = np.load(os.path.join(std_param_dir, "std.npy"))
    return StandardizeLoad(mu_std, std_std, device)


def _load_scvi_var_names(scvi_model_dir: str) -> list[str] | None:
    var_path = Path(scvi_model_dir) / "var_names.csv"
    if not var_path.exists():
        return None
    var_names = [line.strip() for line in var_path.read_text().splitlines() if line.strip()]
    return var_names if var_names else None


def _infer_chemvae_latent_dim(model_chemvae: ChemicalVAE) -> int:
    if hasattr(model_chemvae, "linear_1"):
        return int(model_chemvae.linear_1.out_features)
    if hasattr(model_chemvae, "z_dim"):
        return int(model_chemvae.z_dim)
    raise AttributeError("Unable to infer chemVAE latent dim from model.")


def _add_indices_if_missing(adata_obj: ad.AnnData) -> ad.AnnData:
    if "indices" not in adata_obj.obs.columns:
        adata_obj.obs["indices"] = np.arange(adata_obj.n_obs)
    return adata_obj


def _build_prediction_output_adata(
    adata_test: ad.AnnData,
    pred_full: np.ndarray,
    args: argparse.Namespace,
) -> ad.AnnData:
    if str(args.output_layout) == "full":
        adata_out = adata_test.copy()
    else:
        keep_cols: list[str] = []
        for col in ("indices", str(args.cell_type_col), str(args.treatment_col)):
            if col not in adata_test.obs.columns:
                raise KeyError(f"Missing required obs column for lite output: {col}")
            if col not in keep_cols:
                keep_cols.append(col)

        obs_out = adata_test.obs.loc[:, keep_cols].copy()
        var_out = adata_test.var.iloc[:, 0:0].copy()
        adata_out = ad.AnnData(obs=obs_out, var=var_out)

    adata_out.layers["prediction"] = pred_full
    if args.add_predicted_alias:
        adata_out.layers["predicted"] = pred_full.copy()
    return adata_out


def _load_and_split_sciplex(args: argparse.Namespace):
    adata_raw = ad.read_h5ad(args.adata_path)
    adata_raw = _add_indices_if_missing(adata_raw)
    adata = adata_raw
    if args.good_smile_key in adata.obs.columns:
        mask = np.asarray(adata.obs[args.good_smile_key] == args.good_smile_value)
        adata = adata[mask, :].copy()

    split = load_kfold_split_testsize(
        fold_num=args.fold,
        split_dir=args.split_dir,
        k=args.k,
        test_size=args.test_size,
    )
    adata_train = adata[split["train_idx"], :].copy()
    adata_test = adata[split["test_idx"], :].copy()
    return adata_train, adata_test


def _align_to_scvi_vars(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    scvi_model_dir: str,
) -> tuple[ad.AnnData, ad.AnnData]:
    scvi_var_names = _load_scvi_var_names(scvi_model_dir)
    if scvi_var_names is None:
        return adata_train, adata_test

    missing_vars = [g for g in scvi_var_names if g not in adata_train.var_names]
    if missing_vars:
        raise ValueError(
            f"adata_train missing {len(missing_vars)} scVI vars, first few: {missing_vars[:10]}"
        )

    if not np.array_equal(adata_train.var_names.to_numpy(), np.asarray(scvi_var_names)):
        adata_train = adata_train[:, scvi_var_names].copy()
        adata_test = adata_test[:, scvi_var_names].copy()
    return adata_train, adata_test


def _load_train_artifacts(cache_dir: str, fold: int, test_size: float) -> dict[str, Any]:
    cached = load_cached_train_artifacts(
        cache_dir=cache_dir,
        fold=fold,
        test_size=test_size,
    )
    if cached is None:
        raise ValueError(
            f"Missing train artifacts in cache_dir={cache_dir} for fold={fold}, test_size={test_size}"
        )
    return cached


def _build_model(
    *,
    args: argparse.Namespace,
    run_cfg: Dict[str, Any],
    chemvae_model,
    std_model,
    scvi_model,
    adata_train,
    enhanced_cov_train,
    train_fit,
    model_path: str,
    device: str,
):
    cond_stage_data, embeddings, perturb_to_embed = prepare_embeddings_cinn(
        adata_train,
        perturbation_key=args.perturbation_key,
        trt_key=args.trt_key,
        embed_key=args.embed_key,
    )

    chemvae_dim = _infer_chemvae_latent_dim(chemvae_model)
    drug_proj_mode, drug_proj_output_dim = _resolve_drug_proj_from_run_cfg(
        run_cfg,
        chemvae_dim,
    )
    conditioning_dim = int(
        run_cfg.get(
            "conditioning_dim_resolved",
            drug_proj_output_dim + int(enhanced_cov_train.shape[1]),
        )
    )
    flow_model = ConditionalFlatCouplingFlow(
        conditioning_dim=conditioning_dim,
        embedding_dim=int(run_cfg.get("embedding_dim", 10)),
        conditioning_depth=int(run_cfg.get("conditioning_depth", 2)),
        n_flows=int(run_cfg.get("n_flows", 20)),
        in_channels=int(run_cfg.get("in_channels", 10)),
        hidden_dim=int(run_cfg.get("hidden_dim", 1024)),
        hidden_depth=int(run_cfg.get("hidden_depth", 2)),
        activation="none",
    )

    cov_layout = infer_sciplex_covariate_layout(enhanced_cov_train, train_fit)
    cell_type_dim = int(run_cfg.get("cell_type_dim_resolved", cov_layout["cell_context_dim"]))

    model_cinn = Net2NetFlow_TFVAE_Covariate_Flow(
        configured_flow=flow_model,
        first_stage_data=np.zeros((adata_train.n_obs, 1), dtype=np.float32),
        cond_stage_data=cond_stage_data,
        perturbToOnehotLib=perturb_to_embed,
        oneHotData=embeddings,
        model_con=chemvae_model,
        std_model=std_model,
        covariates=enhanced_cov_train,
        scvi_model=scvi_model,
        n_epochs=int(run_cfg.get("epochs", 50)),
        cell_type_dim=cell_type_dim,
        drug_proj_mode=drug_proj_mode,
        drug_proj_dim=(drug_proj_output_dim if drug_proj_mode == "linear" else None),
    )
    model_cinn.to(device=device)
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    model_cinn.load(
        dir_path=str(model_path),
        use_cuda=use_cuda,
        save_use_cuda=use_cuda,
    )
    model_cinn.to(device=device)
    model_cinn.eval()
    return model_cinn


def _build_name_to_smiles(adata_train: ad.AnnData, adata_test: ad.AnnData, perturbation_col: str, perturbation_key: str) -> Dict[str, str]:
    name_to_smiles: Dict[str, str] = {}
    for adata_obj in (adata_train, adata_test):
        if perturbation_col not in adata_obj.obs.columns or perturbation_key not in adata_obj.obs.columns:
            continue
        pairs = adata_obj.obs[[perturbation_col, perturbation_key]].drop_duplicates()
        for _, row in pairs.iterrows():
            name_to_smiles[str(row[perturbation_col])] = str(row[perturbation_key])
    return name_to_smiles


def _maybe_batch_data(adata_subset: ad.AnnData) -> Optional[np.ndarray]:
    for col in ("_scvi_batch", "batch", "batch_index"):
        if col in adata_subset.obs.columns:
            arr = adata_subset.obs[col].to_numpy()
            return np.asarray(arr, dtype=np.int64).reshape(-1, 1)
    return None


def main() -> None:
    args = parse_args()
    random.seed(args.prediction_seed)
    np.random.seed(args.prediction_seed)
    torch.manual_seed(args.prediction_seed)

    device = _resolve_device(args.device)
    model_path = _resolve_model_path(args)
    run_cfg = _load_model_run_config(model_path)
    scvi_model_dir = _resolve_scvi_model_dir(args, run_cfg)
    similars_map_path = _resolve_similars_map_path(args)
    conditional_vector_mode = _resolve_conditional_vector_mode(
        args.conditional_vector_mode,
        run_cfg,
    )
    cache_dir = _resolve_cache_dir(args, run_cfg)

    print(f"[INFO] device={device}")
    print(f"[INFO] model_path={model_path}")
    print(f"[INFO] scvi_model_dir={scvi_model_dir}")
    print(f"[INFO] cache_dir={cache_dir}")
    print(f"[INFO] conditional_vector_mode={conditional_vector_mode}")

    adata_train, adata_test = _load_and_split_sciplex(args)
    adata_train, adata_test = _align_to_scvi_vars(adata_train, adata_test, scvi_model_dir)

    scvi.data.setup_anndata(adata_train, layer="counts")
    scvi_model = scvi.model.SCVI.load(scvi_model_dir, adata_train, use_cuda=False)
    scvi_model_de = scvi_predictive_z(scvi_model)

    chem_onehot = np.load(args.chem_onehot_path)
    chemvae_model = ChemicalVAE(
        n_char=chem_onehot.shape[2],
        max_len=chem_onehot.shape[1],
    ).to(device)
    chemvae_model.load_state_dict(torch.load(args.chemvae_model_path, map_location=device))
    chemvae_model.eval()
    std_model = _load_std_model(args.std_param_dir, device)

    cached = _load_train_artifacts(cache_dir, args.fold, args.test_size)
    enhanced_cov_train = cached["enhanced_cov_train"]
    train_fit = cached["train_fit"]

    model_cinn = _build_model(
        args=args,
        run_cfg=run_cfg,
        chemvae_model=chemvae_model,
        std_model=std_model,
        scvi_model=scvi_model,
        adata_train=adata_train,
        enhanced_cov_train=enhanced_cov_train,
        train_fit=train_fit,
        model_path=model_path,
        device=device,
    )
    predictor = SCVIZ_CheckNet2Net(model_cinn, device, scvi_model_de)

    with open(similars_map_path, "r", encoding="utf-8") as f:
        similars_map = json.load(f)
    name_to_smiles = _build_name_to_smiles(
        adata_train,
        adata_test,
        args.perturbation_col,
        args.perturbation_key,
    )

    perturb_with_onehot_removed = np.asarray(adata_test.obs[args.perturbation_key]).astype(str)
    unique_trts = np.unique(perturb_with_onehot_removed)
    library_source_adata = adata_train if args.library_source == "train" else adata_test
    library_source = scvi_model.get_latent_library_size(
        adata=library_source_adata,
        give_mean=False,
    )

    pred_dtype = np.float32 if args.prediction_dtype == "float32" else np.float64
    pred_full = np.zeros((adata_test.n_obs, adata_test.n_vars), dtype=pred_dtype)
    skipped: list[dict[str, Any]] = []

    for trt in unique_trts:
        idx_trt = np.where(perturb_with_onehot_removed == trt)[0]
        if idx_trt.size == 0:
            continue

        pert_values = adata_test.obs.iloc[idx_trt][args.perturbation_col].astype(str).unique().tolist()
        if len(pert_values) != 1:
            skipped.append(
                {
                    "perturbation_key": str(trt),
                    "reason": f"non_unique_perturbation_col:{pert_values[:3]}",
                }
            )
            continue
        pert_name = str(pert_values[0])

        onehot_idx = np.where(adata_test.uns[args.trt_key] == trt)[0]
        if onehot_idx.size == 0:
            skipped.append({"perturbation_key": str(trt), "reason": "missing_trt_key_lookup"})
            continue
        onehot_template = adata_test.uns[args.embed_key][[int(onehot_idx[0])]]
        chunk_size = max(1, int(args.max_cells_per_chunk))
        n_chunks = int(np.ceil(idx_trt.size / chunk_size))
        if n_chunks > 1:
            print(
                f"[INFO] chunking treatment={pert_name} ({trt}) "
                f"cells={idx_trt.size} chunk_size={chunk_size} n_chunks={n_chunks}"
            )

        for start in range(0, idx_trt.size, chunk_size):
            idx_chunk = idx_trt[start : start + chunk_size]
            onehot_indice_trt = np.tile(onehot_template, (idx_chunk.size, 1, 1))

            with torch.no_grad():
                _, _, _, embdata_torch = chemvae_model(
                    torch.tensor(onehot_indice_trt).float().to(device)
                )
                embdata_np = std_model.standardize_z(embdata_torch.cpu().detach().numpy())

            lib_idx = np.random.choice(library_source.shape[0], idx_chunk.size, replace=True)
            library_trt = library_source[lib_idx]

            enhanced_cov_unseen = build_unseen_enhanced_cov_with_similars(
                adata_meta_like=adata_test[idx_chunk],
                unseen_names=[pert_name] * idx_chunk.size,
                similars_map=similars_map,
                train_fit=train_fit,
                cell_type_col=args.cell_type_col,
                dose_col=args.dose_col,
                name_to_smiles=name_to_smiles,
                similars_weight_mode=args.similars_weight_mode,
                similars_score_threshold=args.similars_score_threshold,
            )

            batch_data = _maybe_batch_data(adata_test[idx_chunk])

            with torch.no_grad():
                _, fake_data = predictor.sample_data(
                    z_drug_std=embdata_np,
                    covariates=enhanced_cov_unseen,
                    library_data=library_trt,
                    deterministic=args.deterministic_decode,
                    batch_data=batch_data,
                    strict_cond_alignment=args.strict_cond_alignment,
                    conditional_vector=conditional_vector_mode,
                )

            pred_full[idx_chunk] = np.asarray(fake_data, dtype=pred_dtype)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    adata_out = _build_prediction_output_adata(
        adata_test=adata_test,
        pred_full=pred_full,
        args=args,
    )
    adata_out.uns["prediction_run_config"] = {
        "model_path": str(model_path),
        "scvi_model_dir": str(scvi_model_dir),
        "cache_dir": str(cache_dir),
        "similars_map_path": str(similars_map_path),
        "conditional_vector_mode": str(conditional_vector_mode),
        "similars_weight_mode": str(args.similars_weight_mode),
        "similars_score_threshold": float(args.similars_score_threshold),
        "prediction_seed": int(args.prediction_seed),
        "library_source": str(args.library_source),
        "deterministic_decode": bool(args.deterministic_decode),
        "output_layout": str(args.output_layout),
        "n_skipped_groups": int(len(skipped)),
    }
    if skipped:
        adata_out.uns["prediction_skipped_groups"] = skipped

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_out.write_h5ad(output_path)
    print(f"[DONE] wrote predicted h5ad to: {output_path}")
    print(f"[INFO] prediction layer shape={adata_out.layers['prediction'].shape}")
    if skipped:
        print(f"[WARN] skipped groups={len(skipped)}")


if __name__ == "__main__":
    main()
