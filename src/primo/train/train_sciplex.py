#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scvi
import torch

from primo.vendor.perturbnet.chemicalvae.chemicalVAE import ChemicalVAE
from primo.utils.perturbation_embeddings import prepare_embeddings_cinn
from primo.utils.splits import load_kfold_split_testsize
from primo.utils.standardize import StandardizeLoad

from primo.models.sciplex_flow import (
    ConditionalFlatCouplingFlow,
    Net2NetFlow_TFVAE_Covariate_Flow,
)
from primo.conditioning.sciplex import (
    DEFAULT_SCIPLEX_CELL_CONTEXT_DIM,
    DEFAULT_SCIPLEX_CELL_CONTEXT_MODE,
    DEFAULT_SCIPLEX_CONTROL_COL,
    DEFAULT_SCIPLEX_CONTROL_LABEL,
    DEFAULT_SCIPLEX_RESIDUAL_SHRINKAGE,
    DEFAULT_SCIPLEX_STATS_PCA_DIM,
    build_or_load_sciplex_train_artifacts,
    default_sciplex_cache_dir,
    infer_sciplex_covariate_layout,
    requested_sciplex_covariate_config,
    summarize_train_fit,
)
from primo.paths import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SciPlex PRIMO/GRCA cINN with control-anchored covariates."
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
    p.add_argument("--dose_col", type=str, default="dose")
    p.add_argument("--control_col", type=str, default=DEFAULT_SCIPLEX_CONTROL_COL)
    p.add_argument("--control_label", type=str, default=DEFAULT_SCIPLEX_CONTROL_LABEL)
    p.add_argument(
        "--perturbation_key",
        type=str,
        default="perturb_string_processed",
        help="Primary perturbation key used to look up chemVAE onehot rows.",
    )
    p.add_argument(
        "--trt_key",
        type=str,
        default="ordered_all_smile_chemvae",
    )
    p.add_argument(
        "--embed_key",
        type=str,
        default="ordered_all_onehot_chemvae",
    )

    p.add_argument(
        "--scvi_model_dir_template",
        type=str,
        default=str(CHECKPOINT_DIR / "sciplex" / "scvi" / "fold{fold}_{test_size}"),
    )
    p.add_argument(
        "--scvi_model_dir",
        type=str,
        default="",
        help="If set, override --scvi_model_dir_template.",
    )
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
        "--cached_cov_dir",
        type=str,
        default="",
        help="Exact cache dir for train_fit/enhanced_cov. If empty, use versioned default.",
    )
    p.add_argument(
        "--cell_context_mode",
        type=str,
        default=DEFAULT_SCIPLEX_CELL_CONTEXT_MODE,
        choices=["onehot", "control_pca"],
    )
    p.add_argument("--cell_context_dim", type=int, default=DEFAULT_SCIPLEX_CELL_CONTEXT_DIM)
    p.add_argument("--stats_pca_dim", type=int, default=DEFAULT_SCIPLEX_STATS_PCA_DIM)
    p.add_argument(
        "--residualize_to_control",
        dest="residualize_to_control",
        action="store_true",
    )
    p.add_argument(
        "--no_residualize_to_control",
        dest="residualize_to_control",
        action="store_false",
    )
    p.set_defaults(residualize_to_control=True)
    p.add_argument(
        "--residual_shrinkage",
        type=float,
        default=DEFAULT_SCIPLEX_RESIDUAL_SHRINKAGE,
    )
    p.add_argument(
        "--rebuild_cached_cov",
        action="store_true",
        help="Force rebuilding train_fit/enhanced_cov even if cache already exists.",
    )

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1.2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
    )
    p.add_argument("--centroid_lambda", type=float, default=0.20)
    p.add_argument("--decoded_effect_lambda", type=float, default=0.0)
    p.add_argument("--decoded_effect_target_size", type=float, default=1e4)
    p.add_argument(
        "--conditional_vector_mode",
        type=str,
        default="grca",
        choices=["full", "simple", "adapter", "grca", "none"],
    )

    p.add_argument(
        "--drug_proj_mode",
        type=str,
        default="none",
        choices=["none", "linear"],
    )
    p.add_argument("--drug_proj_dim", type=int, default=0)

    p.add_argument("--conditioning_dim", type=int, default=267)
    p.add_argument("--embedding_dim", type=int, default=10)
    p.add_argument("--conditioning_depth", type=int, default=2)
    p.add_argument("--n_flows", type=int, default=20)
    p.add_argument("--in_channels", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--hidden_depth", type=int, default=2)
    p.add_argument(
        "--cell_type_dim",
        type=int,
        default=DEFAULT_SCIPLEX_CELL_CONTEXT_DIM,
        help="Legacy hint only. Effective value is inferred from train_fit/cached cov.",
    )

    p.add_argument(
        "--scvi_give_mean",
        dest="scvi_give_mean",
        action="store_true",
        help="Use deterministic scVI latent mean during cINN training.",
    )
    p.add_argument(
        "--scvi_sample_latent",
        dest="scvi_give_mean",
        action="store_false",
        help="Sample scVI latent instead of using latent mean.",
    )
    p.set_defaults(scvi_give_mean=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--auto_save_every", type=int, default=0)
    p.add_argument(
        "--save_best_checkpoint",
        dest="save_best_checkpoint",
        action="store_true",
    )
    p.add_argument(
        "--no_save_best_checkpoint",
        dest="save_best_checkpoint",
        action="store_false",
    )
    p.set_defaults(save_best_checkpoint=True)
    p.add_argument("--best_checkpoint_name", type=str, default="model_params_best.pt")
    p.add_argument("--promote_best_checkpoint", action="store_true")
    p.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Explicit output dir. If empty, use --save_dir_template.",
    )
    p.add_argument(
        "--save_dir_template",
        type=str,
        default=str(OUTPUT_DIR / "sciplex" / "cinn" / "fold{fold}_{test_size}_grca"),
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_text: str) -> torch.device:
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def resolve_scvi_model_dir(args: argparse.Namespace) -> str:
    if args.scvi_model_dir:
        return args.scvi_model_dir
    return args.scvi_model_dir_template.format(fold=args.fold, test_size=args.test_size)


def resolve_cached_cov_dir(args: argparse.Namespace) -> str:
    if args.cached_cov_dir:
        return args.cached_cov_dir
    return default_sciplex_cache_dir(
        test_size=args.test_size,
        cell_context_mode=args.cell_context_mode,
        cell_context_dim=args.cell_context_dim,
        stats_pca_dim=args.stats_pca_dim,
        control_label=args.control_label,
        residualize_to_control=args.residualize_to_control,
        residual_shrinkage=args.residual_shrinkage,
        perturbation_key=args.perturbation_key,
    )


def resolve_save_dir(args: argparse.Namespace) -> str:
    if args.save_dir:
        return args.save_dir
    return args.save_dir_template.format(fold=args.fold, test_size=args.test_size)


def resolve_effective_auto_save_every(requested_auto_save_every: int) -> int:
    requested = max(0, int(requested_auto_save_every))
    if requested > 0:
        print(
            "[INFO] requested auto_save_every=%d, but preserving only "
            "model_params_best.pt and the final model_params.pt; "
            "disabling intermediate epoch checkpoints."
            % requested
        )
    return 0


def cleanup_extra_checkpoints(save_dir: Path, best_checkpoint_name: str) -> None:
    keep_names = {"model_params.pt", str(best_checkpoint_name)}
    removed_names: list[str] = []
    for checkpoint_path in sorted(save_dir.glob("model_params*.pt")):
        if checkpoint_path.name in keep_names:
            continue
        checkpoint_path.unlink(missing_ok=True)
        removed_names.append(checkpoint_path.name)
    if removed_names:
        print(
            "[INFO] removed extra checkpoints: %s"
            % ", ".join(removed_names)
        )


def load_scvi_var_names(scvi_model_dir: str) -> list[str] | None:
    var_path = Path(scvi_model_dir) / "var_names.csv"
    if not var_path.exists():
        return None
    var_names = [line.strip() for line in var_path.read_text().splitlines() if line.strip()]
    return var_names if var_names else None


def infer_chemvae_latent_dim(model_chemvae: ChemicalVAE) -> int:
    if hasattr(model_chemvae, "linear_1"):
        return int(model_chemvae.linear_1.out_features)
    if hasattr(model_chemvae, "z_dim"):
        return int(model_chemvae.z_dim)
    raise AttributeError("Unable to infer chemVAE latent dim from model.")


def resolve_drug_proj_output_dim(
    drug_proj_mode: str,
    drug_proj_dim: int,
    chemvae_latent_dim: int,
) -> int:
    mode = str(drug_proj_mode).strip().lower()
    if mode == "none":
        return int(chemvae_latent_dim)
    if mode != "linear":
        raise ValueError(f"Unsupported drug_proj_mode: {drug_proj_mode}")
    if int(drug_proj_dim) <= 0:
        raise ValueError("开启线性药物投影时，--drug_proj_dim 必须是正整数。")
    return int(drug_proj_dim)


def build_placeholder_first_stage(n_rows: int) -> np.ndarray:
    return np.zeros((int(n_rows), 1), dtype=np.float32)


def load_and_split_sciplex(args: argparse.Namespace):
    adata_raw = ad.read_h5ad(args.adata_path)
    adata_control_ref = None
    if args.control_col in adata_raw.obs.columns:
        control_mask = adata_raw.obs[args.control_col].astype(str) == str(args.control_label)
        if np.asarray(control_mask).any():
            adata_control_ref = adata_raw[control_mask, :].copy()

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
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    adata_train = adata[train_idx, :].copy()
    adata_test = adata[test_idx, :].copy()
    return adata_train, adata_test, adata_control_ref, split


def align_to_scvi_vars(
    adata_train,
    adata_test,
    adata_control_ref,
    scvi_model_dir: str,
):
    scvi_var_names = load_scvi_var_names(scvi_model_dir)
    if scvi_var_names is None:
        return adata_train, adata_test, adata_control_ref

    missing_vars = [g for g in scvi_var_names if g not in adata_train.var_names]
    if missing_vars:
        raise ValueError(
            f"adata_train missing {len(missing_vars)} scVI vars, first few: {missing_vars[:10]}"
        )

    if not np.array_equal(adata_train.var_names.to_numpy(), np.asarray(scvi_var_names)):
        print(
            "[INFO] aligning adata vars to scVI var_names: "
            f"{adata_train.n_vars} -> {len(scvi_var_names)}"
        )
        adata_train = adata_train[:, scvi_var_names].copy()
        adata_test = adata_test[:, scvi_var_names].copy()
        if adata_control_ref is not None:
            keep = [g for g in scvi_var_names if g in adata_control_ref.var_names]
            adata_control_ref = adata_control_ref[:, keep].copy()

    return adata_train, adata_test, adata_control_ref


def save_run_config(
    path: Path,
    args: argparse.Namespace,
    scvi_model_dir: str,
    cached_cov_dir: str,
    train_fit_summary: dict,
    enhanced_cov_shape: tuple[int, ...],
    conditioning_dim_resolved: int,
    cell_type_dim_resolved: int,
    drug_proj_output_dim: int,
    effective_auto_save_every: int,
    promote_best_checkpoint_applied: bool,
) -> None:
    cfg = {
        "flow_api": "old",
        "dataset": "sciplex",
        "fold": args.fold,
        "test_size": args.test_size,
        "adata_path": args.adata_path,
        "split_dir": args.split_dir,
        "good_smile_key": args.good_smile_key,
        "good_smile_value": int(args.good_smile_value),
        "cell_type_col": args.cell_type_col,
        "dose_col": args.dose_col,
        "control_col": args.control_col,
        "control_label": args.control_label,
        "perturbation_key": args.perturbation_key,
        "trt_key": args.trt_key,
        "embed_key": args.embed_key,
        "scvi_model_dir": scvi_model_dir,
        "cached_cov_dir": cached_cov_dir,
        "cell_context_mode": args.cell_context_mode,
        "cell_context_dim_requested": int(args.cell_context_dim),
        "stats_pca_dim_requested": int(args.stats_pca_dim),
        "residualize_to_control": bool(args.residualize_to_control),
        "residual_shrinkage": float(args.residual_shrinkage),
        "train_fit_summary": train_fit_summary,
        "enhanced_cov_shape": list(enhanced_cov_shape),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "conditioning_dim_requested": int(args.conditioning_dim),
        "conditioning_dim_resolved": int(conditioning_dim_resolved),
        "drug_proj_mode": str(args.drug_proj_mode),
        "drug_proj_dim_requested": int(args.drug_proj_dim),
        "drug_proj_output_dim": int(drug_proj_output_dim),
        "embedding_dim": int(args.embedding_dim),
        "conditioning_depth": int(args.conditioning_depth),
        "n_flows": int(args.n_flows),
        "in_channels": int(args.in_channels),
        "hidden_dim": int(args.hidden_dim),
        "hidden_depth": int(args.hidden_depth),
        "cell_type_dim_requested": int(args.cell_type_dim),
        "cell_type_dim_resolved": int(cell_type_dim_resolved),
        "scheduler_type": args.scheduler_type,
        "weight_decay": args.weight_decay,
        "centroid_lambda": args.centroid_lambda,
        "decoded_effect_lambda": float(args.decoded_effect_lambda),
        "decoded_effect_target_size": float(args.decoded_effect_target_size),
        "conditional_vector_mode": args.conditional_vector_mode,
        "scvi_give_mean": bool(args.scvi_give_mean),
        "save_best_checkpoint": bool(args.save_best_checkpoint),
        "best_checkpoint_name": args.best_checkpoint_name,
        "auto_save_every": int(effective_auto_save_every),
        "requested_auto_save_every": int(args.auto_save_every),
        "seed": args.seed,
        "device": str(args.device),
        "promote_best_checkpoint_requested": bool(args.promote_best_checkpoint),
        "promote_best_checkpoint_applied": bool(promote_best_checkpoint_applied),
        "release_pipeline": "primo_old_flow_grca",
    }
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"[INFO] device={device}")
    print(f"[INFO] fold={args.fold}, test_size={args.test_size}, epochs={args.epochs}")
    print(
        "[INFO] SciPlex old-flow training | perturbation_key=%s | cond_mode=%s"
        % (args.perturbation_key, args.conditional_vector_mode)
    )

    adata_train, _, adata_control_ref, _ = load_and_split_sciplex(args)
    scvi_model_dir = resolve_scvi_model_dir(args)
    print(f"[INFO] scvi_model_dir={scvi_model_dir}")

    adata_train, _, adata_control_ref = align_to_scvi_vars(
        adata_train,
        adata_train[:0].copy(),
        adata_control_ref,
        scvi_model_dir,
    )

    scvi.data.setup_anndata(adata_train, layer="counts")
    scvi_model_train = scvi.model.SCVI.load(scvi_model_dir, adata_train, use_cuda=False)

    data_chem_onehot = np.load(args.chem_onehot_path)
    model_chemvae = ChemicalVAE(
        n_char=data_chem_onehot.shape[2],
        max_len=data_chem_onehot.shape[1],
    ).to(device)
    model_chemvae.load_state_dict(torch.load(args.chemvae_model_path, map_location=device))
    model_chemvae.eval()

    mu_std_model = np.load(os.path.join(args.std_param_dir, "mu.npy"))
    std_std_model = np.load(os.path.join(args.std_param_dir, "std.npy"))
    std_model = StandardizeLoad(mu_std_model, std_std_model, device)

    cond_stage_data, embeddings, perturb_to_embed = prepare_embeddings_cinn(
        adata_train,
        perturbation_key=args.perturbation_key,
        trt_key=args.trt_key,
        embed_key=args.embed_key,
    )

    cached_cov_dir = resolve_cached_cov_dir(args)
    requested_cov_config = requested_sciplex_covariate_config(
        cell_context_mode=args.cell_context_mode,
        cell_context_dim=args.cell_context_dim,
        stats_pca_dim=args.stats_pca_dim,
        control_col=args.control_col,
        control_label=args.control_label,
        residualize_to_control=args.residualize_to_control,
        residual_shrinkage=args.residual_shrinkage,
    )
    cached_data = build_or_load_sciplex_train_artifacts(
        adata_train=adata_train,
        adata_control_ref=adata_control_ref,
        cond_stage_data=cond_stage_data,
        fold=args.fold,
        test_size=args.test_size,
        cache_dir=cached_cov_dir,
        requested_config=requested_cov_config,
        force_rebuild=args.rebuild_cached_cov,
    )
    enhanced_cov_train = cached_data["enhanced_cov_train"]
    train_fit = cached_data["train_fit"]
    cov_layout = infer_sciplex_covariate_layout(enhanced_cov_train, train_fit)
    print(
        "[INFO] cache_action=%s cache_dir=%s cov_dim=%d cell_context_dim=%d stats_pca_dim=%d"
        % (
            cached_data["cache_action"],
            cached_cov_dir,
            cov_layout["cov_dim"],
            cov_layout["cell_context_dim"],
            cov_layout["stats_pca_dim"],
        )
    )
    if cached_data.get("cache_mismatch_reason"):
        print(f"[INFO] rebuilt cached cov because: {cached_data['cache_mismatch_reason']}")

    chemvae_latent_dim = infer_chemvae_latent_dim(model_chemvae)
    drug_proj_output_dim = resolve_drug_proj_output_dim(
        args.drug_proj_mode,
        args.drug_proj_dim,
        chemvae_latent_dim,
    )
    conditioning_dim_resolved = int(drug_proj_output_dim + enhanced_cov_train.shape[1])
    cell_type_dim_resolved = int(cov_layout["cell_context_dim"])
    if int(args.conditioning_dim) != conditioning_dim_resolved:
        print(
            f"[INFO] conditioning_dim requested={args.conditioning_dim} -> resolved={conditioning_dim_resolved}"
        )
    if int(args.cell_type_dim) != cell_type_dim_resolved:
        print(
            f"[INFO] cell_type_dim requested={args.cell_type_dim} -> resolved={cell_type_dim_resolved}"
        )
    print(f"[INFO] train_fit summary={json.dumps(summarize_train_fit(train_fit), sort_keys=True)}")

    flow_model = ConditionalFlatCouplingFlow(
        conditioning_dim=conditioning_dim_resolved,
        embedding_dim=args.embedding_dim,
        conditioning_depth=args.conditioning_depth,
        n_flows=args.n_flows,
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        activation="none",
    )

    model_c = Net2NetFlow_TFVAE_Covariate_Flow(
        configured_flow=flow_model,
        first_stage_data=build_placeholder_first_stage(adata_train.n_obs),
        cond_stage_data=cond_stage_data,
        perturbToOnehotLib=perturb_to_embed,
        oneHotData=embeddings,
        model_con=model_chemvae,
        std_model=std_model,
        covariates=enhanced_cov_train,
        scvi_model=scvi_model_train,
        n_epochs=args.epochs,
        cell_type_dim=cell_type_dim_resolved,
        drug_proj_mode=args.drug_proj_mode,
        drug_proj_dim=(
            drug_proj_output_dim if str(args.drug_proj_mode).lower() == "linear" else None
        ),
    )

    save_dir = Path(resolve_save_dir(args))
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    effective_auto_save_every = resolve_effective_auto_save_every(args.auto_save_every)
    promote_best_checkpoint_applied = False
    if args.promote_best_checkpoint:
        print(
            "[INFO] ignoring --promote_best_checkpoint so model_params.pt "
            "stays as the final-epoch checkpoint and model_params_best.pt "
            "stays as the best checkpoint."
        )

    model_c.to(device=device)
    model_c.train_cinn(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        auto_save=effective_auto_save_every,
        auto_save_path=str(save_dir),
        use_scheduler=True,
        scheduler_type=args.scheduler_type,
        centroid_lambda=args.centroid_lambda,
        decoded_effect_lambda=args.decoded_effect_lambda,
        decoded_effect_target_size=args.decoded_effect_target_size,
        weight_decay=args.weight_decay,
        conditional_vector=args.conditional_vector_mode,
        scvi_give_mean=args.scvi_give_mean,
        save_best_checkpoint=args.save_best_checkpoint,
        best_checkpoint_name=args.best_checkpoint_name,
        device=device,
    )

    model_c.save(str(save_dir), overwrite=True)
    cleanup_extra_checkpoints(save_dir, args.best_checkpoint_name)
    save_run_config(
        path=save_dir / "run_config.json",
        args=args,
        scvi_model_dir=scvi_model_dir,
        cached_cov_dir=cached_cov_dir,
        train_fit_summary=summarize_train_fit(train_fit),
        enhanced_cov_shape=tuple(enhanced_cov_train.shape),
        conditioning_dim_resolved=conditioning_dim_resolved,
        cell_type_dim_resolved=cell_type_dim_resolved,
        drug_proj_output_dim=drug_proj_output_dim,
        effective_auto_save_every=effective_auto_save_every,
        promote_best_checkpoint_applied=promote_best_checkpoint_applied,
    )

    print(f"[DONE] SciPlex PRIMO/GRCA model saved to: {save_dir}")


if __name__ == "__main__":
    main()
