#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scvi
import torch
from scipy import sparse

from primo.vendor.perturbnet.chemicalvae.chemicalVAE import ChemicalVAE
from primo.models.grca_flow import Net2NetFlow_TFVAE_Covariate_Flow
from primo.vendor.perturbnet.net2net.modules.flow.flatflow import ConditionalFlatCouplingFlow
from primo.utils.perturbation_embeddings import prepare_embeddings_cinn
from primo.utils.splits import load_kfold_split_testsize
from primo.utils.standardize import StandardizeLoad
from primo.conditioning.nips import (
    DEFAULT_NIPS_CELL_CONTEXT_DIM,
    DEFAULT_NIPS_CELL_CONTEXT_MODE,
    DEFAULT_NIPS_CONTROL_COL,
    DEFAULT_NIPS_CONTROL_LABEL,
    DEFAULT_NIPS_RESIDUAL_SHRINKAGE,
    DEFAULT_NIPS_STATS_PCA_DIM,
    build_or_load_nips_train_artifacts,
    default_nips_cache_dir,
    infer_nips_covariate_layout,
    requested_nips_covariate_config,
    summarize_train_fit,
)
from primo.paths import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train cINN with old flow API (simplified for NIPS)."
    )
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument(
        "--adata_path",
        type=str,
        default=str(DATA_DIR / "nips" / "nips_processed_min50_split82.h5ad"),
    )
    p.add_argument("--split_dir", type=str, default=str(DATA_DIR / "nips" / "splits_82"))

    p.add_argument(
        "--scvi_model_dir_template",
        type=str,
        default=str(CHECKPOINT_DIR / "nips" / "scvi" / "fold{fold}_{test_size}"),
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
        "--drug_proj_mode",
        type=str,
        default="none",
        choices=["none", "linear"],
    )
    p.add_argument("--drug_proj_dim", type=int, default=0)
    p.add_argument(
        "--cached_cov_dir",
        type=str,
        default="",
        help="Exact cache dir for train_fit/enhanced_cov. Overrides --cached_cov_dir_template.",
    )
    p.add_argument(
        "--cached_cov_dir_template",
        type=str,
        default="",
        help="Legacy template. If empty, use an auto-versioned cache dir for the requested NIPS covariate setup.",
    )
    p.add_argument(
        "--cell_context_mode",
        type=str,
        default=DEFAULT_NIPS_CELL_CONTEXT_MODE,
        choices=["onehot", "control_pca"],
    )
    p.add_argument("--cell_context_dim", type=int, default=DEFAULT_NIPS_CELL_CONTEXT_DIM)
    p.add_argument("--stats_pca_dim", type=int, default=DEFAULT_NIPS_STATS_PCA_DIM)
    p.add_argument("--control_col", type=str, default=DEFAULT_NIPS_CONTROL_COL)
    p.add_argument("--control_label", type=str, default=DEFAULT_NIPS_CONTROL_LABEL)
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
        default=DEFAULT_NIPS_RESIDUAL_SHRINKAGE,
    )
    p.add_argument(
        "--rebuild_cached_cov",
        action="store_true",
        help="Force rebuilding train_fit/enhanced_cov even if cache already exists.",
    )

    p.add_argument(
        "--cinn_perturbation_key",
        type=str,
        default="condition",
        choices=["condition", "SMILES"],
        help="Use condition (recommended) or SMILES (legacy).",
    )
    p.add_argument(
        "--no_strict_perturbation_key_check",
        dest="strict_perturbation_key_check",
        action="store_false",
        help="Disable strict one-to-one check between condition and SMILES.",
    )
    p.set_defaults(strict_perturbation_key_check=True)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=9e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
    )

    p.add_argument("--centroid_lambda", type=float, default=0.2)
    p.add_argument(
        "--decoded_effect_lambda",
        type=float,
        default=0.0,
        help="Auxiliary decoded effect magnitude loss weight. Keep small, e.g. 0.01-0.05.",
    )
    p.add_argument(
        "--decoded_effect_target_size",
        type=float,
        default=1e6,
        help="Normalization target size used inside decoded effect magnitude loss.",
    )
    p.add_argument(
        "--conditional_vector_mode",
        type=str,
        default="simple",
        choices=["full", "simple", "adapter", "grca", "none"],
        help="Conditioning builder mode used in old flow training.",
    )
    p.add_argument(
        "--conditioner_norm",
        type=str,
        default="bn",
        choices=["bn", "ln", "none", "actnorm"],
    )

    p.add_argument(
        "--scvi_give_mean",
        dest="scvi_give_mean",
        action="store_true",
        help="Use deterministic scVI latent mean (old API default).",
    )
    p.add_argument(
        "--scvi_sample_latent",
        dest="scvi_give_mean",
        action="store_false",
        help="Sample scVI latent instead of using latent mean.",
    )
    p.set_defaults(scvi_give_mean=True)

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
        "--conditioning_dim",
        type=int,
        default=267,
        help="Legacy hint only. Effective value is inferred from chemVAE z dim + cached cov dim.",
    )
    p.add_argument("--embedding_dim", type=int, default=10)
    p.add_argument("--conditioning_depth", type=int, default=2)
    p.add_argument("--n_flows", type=int, default=20)
    p.add_argument("--in_channels", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--hidden_depth", type=int, default=2)
    p.add_argument(
        "--cell_type_dim",
        type=int,
        default=6,
        help="Legacy hint only. Effective value is inferred from train_fit/cached cov.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--auto_save_every",
        type=int,
        default=0,
        help="Save periodic model_params{epoch}.pt when > 0.",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=str(OUTPUT_DIR / "nips" / "cinn" / "fold{fold}_{test_size}_grca"),
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


def resolve_save_dir(args: argparse.Namespace) -> str:
    return str(args.save_dir).format(fold=args.fold, test_size=args.test_size)


def resolve_cached_cov_dir(args: argparse.Namespace) -> str:
    if args.cached_cov_dir:
        return args.cached_cov_dir
    if args.cached_cov_dir_template:
        return args.cached_cov_dir_template.format(fold=args.fold, test_size=args.test_size)
    return default_nips_cache_dir(
        test_size=args.test_size,
        cell_context_mode=args.cell_context_mode,
        cell_context_dim=args.cell_context_dim,
        stats_pca_dim=args.stats_pca_dim,
        control_label=args.control_label,
        residualize_to_control=args.residualize_to_control,
        residual_shrinkage=args.residual_shrinkage,
        perturbation_key=args.cinn_perturbation_key,
    )


def load_scvi_var_names(scvi_model_dir: str) -> list[str] | None:
    var_path = Path(scvi_model_dir) / "var_names.csv"
    if not var_path.exists():
        return None
    var_names = [line.strip() for line in var_path.read_text().splitlines() if line.strip()]
    return var_names if var_names else None


def check_condition_smiles_bijection(adata_train, strict: bool = True, max_examples: int = 6) -> None:
    required_cols = {"condition", "SMILES"}
    missing = required_cols.difference(set(adata_train.obs.columns))
    if missing:
        msg = f"missing required obs columns for consistency check: {sorted(missing)}"
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")
        return

    obs = adata_train.obs[["condition", "SMILES"]].copy()
    cond_to_smiles = obs.groupby("condition", observed=True)["SMILES"].nunique()
    smiles_to_cond = obs.groupby("SMILES", observed=True)["condition"].nunique()

    bad_conds = cond_to_smiles[cond_to_smiles > 1].index.tolist()
    bad_smiles = smiles_to_cond[smiles_to_cond > 1].index.tolist()
    if not bad_conds and not bad_smiles:
        return

    details = []
    if bad_conds:
        details.append("conditions with multiple SMILES:")
        for cond in bad_conds[:max_examples]:
            smiles_values = (
                obs.loc[obs["condition"] == cond, "SMILES"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            details.append(f"  - condition={cond}: smiles={smiles_values}")
    if bad_smiles:
        details.append("SMILES mapped to multiple conditions:")
        for smile in bad_smiles[:max_examples]:
            cond_values = (
                obs.loc[obs["SMILES"] == smile, "condition"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            details.append(f"  - smiles={smile}: conditions={cond_values}")

    msg = "condition/SMILES mapping is not one-to-one in adata_train.\n" + "\n".join(details)
    if strict:
        raise ValueError(msg)
    print(f"[WARN] {msg}")


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


def build_first_stage_data(adata_train, use_scvi_latent: bool) -> np.ndarray:
    if use_scvi_latent:
        print("[INFO] 使用 scVI latent 训练，跳过 adata_train.X 的稠密化。")
        return np.zeros((adata_train.n_obs, 1), dtype=np.float32)

    X = adata_train.X
    if sparse.issparse(X):
        est_gb = X.shape[0] * X.shape[1] * 4 / 1024 / 1024 / 1024
        print(f"[WARN] 正在稠密化 adata_train.X，预计至少占用 {est_gb:.3f} GB 内存。")
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def save_run_config(
    path: Path,
    args: argparse.Namespace,
    scvi_model_dir: str,
    trt_lookup_key: str,
    cached_cov_dir: str,
    train_fit_summary: dict,
    enhanced_cov_shape: tuple[int, ...],
    conditioning_dim_resolved: int,
    cell_type_dim_resolved: int,
    drug_proj_output_dim: int,
) -> None:
    cfg = {
        "flow_api": "old",
        "fold": args.fold,
        "test_size": args.test_size,
        "adata_path": args.adata_path,
        "split_dir": args.split_dir,
        "scvi_model_dir": scvi_model_dir,
        "cinn_perturbation_key": args.cinn_perturbation_key,
        "strict_perturbation_key_check": bool(args.strict_perturbation_key_check),
        "trt_lookup_key": trt_lookup_key,
        "cached_cov_dir": cached_cov_dir,
        "cell_context_mode": args.cell_context_mode,
        "cell_context_dim_requested": int(args.cell_context_dim),
        "stats_pca_dim_requested": int(args.stats_pca_dim),
        "control_col": args.control_col,
        "control_label": args.control_label,
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
        "conditioner_norm": args.conditioner_norm,
        "scvi_give_mean": bool(args.scvi_give_mean),
        "save_best_checkpoint": bool(args.save_best_checkpoint),
        "best_checkpoint_name": args.best_checkpoint_name,
        "auto_save_every": args.auto_save_every,
        "seed": args.seed,
        "device": str(args.device),
    }
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"[INFO] device={device}")
    print(f"[INFO] fold={args.fold}, test_size={args.test_size}, epochs={args.epochs}")
    print(
        "[INFO] old-flow training | perturbation_key=%s | strict_check=%s | cond_mode=%s"
        % (
            args.cinn_perturbation_key,
            args.strict_perturbation_key_check,
            args.conditional_vector_mode,
        )
    )

    adata = ad.read_h5ad(args.adata_path)
    adata.obs["indices"] = np.arange(adata.n_obs)

    split = load_kfold_split_testsize(
        fold_num=args.fold,
        split_dir=args.split_dir,
        k=3,
        test_size=args.test_size,
    )
    train_idx = split["train_idx"]
    adata_train = adata[train_idx, :].copy()
    del adata

    # 中文说明：
    # 当 cINN 用 condition 作为主键时，这里强制检查 condition 与 SMILES 是否一一对应，
    # 避免“同一个 condition 对应多个 SMILES”导致 one-hot 查表错位。
    if args.cinn_perturbation_key == "condition":
        check_condition_smiles_bijection(
            adata_train,
            strict=args.strict_perturbation_key_check,
        )

    scvi_model_dir = resolve_scvi_model_dir(args)
    print(f"[INFO] scvi_model_dir={scvi_model_dir}")

    # 中文说明：
    # 旧 flow 训练也必须和 scVI 的 var_names 完全对齐，否则 latent 解码和评估会错位。
    scvi_var_names = load_scvi_var_names(scvi_model_dir)
    if scvi_var_names is not None:
        missing_vars = [g for g in scvi_var_names if g not in adata_train.var_names]
        if missing_vars:
            raise ValueError(
                f"adata_train missing {len(missing_vars)} scVI vars, first few: {missing_vars[:10]}"
            )
        current_vars = adata_train.var_names.to_numpy()
        target_vars = np.asarray(scvi_var_names, dtype=current_vars.dtype)
        if current_vars.shape[0] != target_vars.shape[0] or not np.array_equal(
            current_vars, target_vars
        ):
            print(
                "[INFO] aligning adata_train vars to scVI var_names: "
                f"{adata_train.n_vars} -> {len(scvi_var_names)}"
            )
            adata_train = adata_train[:, scvi_var_names].copy()

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

    trt_lookup_key = (
        "ordered_all_condition_chemvae"
        if args.cinn_perturbation_key == "condition"
        else "ordered_all_smile_chemvae"
    )
    print(f"[INFO] embedding lookup key={trt_lookup_key}")

    cond_stage_data, embeddings, perturb_to_embed = prepare_embeddings_cinn(
        adata_train,
        perturbation_key=args.cinn_perturbation_key,
        trt_key=trt_lookup_key,
        embed_key="ordered_all_onehot_chemvae",
    )

    cached_cov_dir = resolve_cached_cov_dir(args)
    requested_cov_config = requested_nips_covariate_config(
        cell_context_mode=args.cell_context_mode,
        cell_context_dim=args.cell_context_dim,
        stats_pca_dim=args.stats_pca_dim,
        control_col=args.control_col,
        control_label=args.control_label,
        residualize_to_control=args.residualize_to_control,
        residual_shrinkage=args.residual_shrinkage,
    )
    cached_data = build_or_load_nips_train_artifacts(
        adata_train=adata_train,
        cond_stage_data=cond_stage_data,
        fold=args.fold,
        test_size=args.test_size,
        cache_dir=cached_cov_dir,
        requested_config=requested_cov_config,
        force_rebuild=args.rebuild_cached_cov,
    )
    enhanced_cov_train = cached_data["enhanced_cov_train"]
    train_fit = cached_data["train_fit"]
    cov_layout = infer_nips_covariate_layout(enhanced_cov_train, train_fit)
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
    print(
        f"[INFO] drug_proj_mode={args.drug_proj_mode} drug_proj_output_dim={drug_proj_output_dim}"
    )
    if int(args.cell_type_dim) != cell_type_dim_resolved:
        print(
            f"[INFO] cell_type_dim requested={args.cell_type_dim} -> resolved={cell_type_dim_resolved}"
        )
    print(f"[INFO] train_fit summary={json.dumps(summarize_train_fit(train_fit), sort_keys=True)}")

    adata_train_x = build_first_stage_data(
        adata_train=adata_train,
        use_scvi_latent=True,
    )

    conditioner_use_bn = args.conditioner_norm == "bn"
    conditioner_use_ln = args.conditioner_norm == "ln"
    conditioner_use_an = args.conditioner_norm == "actnorm"

    flow_model = ConditionalFlatCouplingFlow(
        conditioning_dim=conditioning_dim_resolved,
        embedding_dim=args.embedding_dim,
        conditioning_depth=args.conditioning_depth,
        n_flows=args.n_flows,
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        activation="none",
        # conditioner_use_bn=conditioner_use_bn,
        # conditioner_use_ln=conditioner_use_ln,
        # conditioner_use_an=conditioner_use_an,
    )

    model_c = Net2NetFlow_TFVAE_Covariate_Flow(
        configured_flow=flow_model,
        first_stage_data=adata_train_x,
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

    # 中文说明：
    # 这里严格只传 old train_cinn 支持的参数，避免新 API 参数误传导致报错。
    model_c.to(device=device)
    model_c.train_cinn(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        auto_save=max(0, int(args.auto_save_every)),
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
    save_run_config(
        path=save_dir / "run_config.json",
        args=args,
        scvi_model_dir=scvi_model_dir,
        trt_lookup_key=trt_lookup_key,
        cached_cov_dir=cached_cov_dir,
        train_fit_summary=summarize_train_fit(train_fit),
        enhanced_cov_shape=tuple(enhanced_cov_train.shape),
        conditioning_dim_resolved=conditioning_dim_resolved,
        cell_type_dim_resolved=cell_type_dim_resolved,
        drug_proj_output_dim=drug_proj_output_dim,
    )

    if args.promote_best_checkpoint:
        best_path = save_dir / args.best_checkpoint_name
        final_path = save_dir / "model_params.pt"
        if best_path.exists():
            shutil.copy2(best_path, final_path)
            print(f"[INFO] promoted best checkpoint: {best_path} -> {final_path}")
        else:
            print(f"[WARN] best checkpoint not found, skip promote: {best_path}")

    print(f"[DONE] NIPS PRIMO/GRCA model saved to: {save_dir}")


if __name__ == "__main__":
    main()
