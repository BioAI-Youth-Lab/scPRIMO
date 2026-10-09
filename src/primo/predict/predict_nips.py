#!/usr/bin/env python3
"""
Generate NIPS prediction h5ad for the ours (PerturbNet+LLM) model.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import scvi
from scipy import sparse
from tqdm import tqdm

from primo.vendor.perturbnet.chemicalvae.chemicalVAE import ChemicalVAE
from primo.vendor.perturbnet.cinn.flow_generate_update import SCVIZ_CheckNet2Net
from primo.vendor.perturbnet.net2net.modules.flow.flatflow import ConditionalFlatCouplingFlow
from primo.models.grca_flow import (
    Net2NetFlow_TFVAE_Covariate_Flow as Net2NetFlow_TFVAE_Covariate_Flow_Old,
)
from primo.vendor.perturbnet.pytorch_scvi.scvi_generate_z import scvi_predictive_z
from primo.conditioning.enhanced_covariates import build_unseen_enhanced_cov_with_similars
from primo.utils.perturbation_embeddings import prepare_embeddings_cinn
from primo.config import load_full_nips_config
from primo.data.nips_dataset import build_nips_dataset
from primo.evaluation.prediction_utils import load_std_model
from primo.evaluation.deg_utils import get_deg_indices_from_precomputed
from primo.paths import ARTIFACT_DIR, CONFIG_DIR, OUTPUT_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PRIMO/GRCA NIPS predicted h5ad."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR / "nips.yaml"),
        help="Path to NIPS PRIMO config.",
    )
    parser.add_argument("--fold", type=int, default=3, help="Fold index.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device.")
    parser.add_argument(
        "--adata_path",
        type=str,
        default="",
        help="Override dataset h5ad path.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="",
        help="Override split directory.",
    )
    parser.add_argument(
        "--scvi_model_dir",
        type=str,
        default="",
        help="Override scVI model dir/path. If empty, auto-resolve from model run_config then config.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Override cache dir containing train_fit/enhanced_cov.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Model directory. Empty means resolved from config paths.model_save_dir.",
    )
    parser.add_argument(
        "--similars_map_path",
        type=str,
        default="",
        help=(
            "LLM similars map JSON path. Empty means "
            "artifacts/nips/{test_size}/similars_map_fold{fold}.json"
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(OUTPUT_DIR / "nips" / "predicted.h5ad"),
        help="Output h5ad path.",
    )
    parser.add_argument(
        "--add_predicted_alias",
        action="store_true",
        help="Create legacy layer 'predicted' alias to prediction.",
    )
    parser.add_argument(
        "--no_add_predicted_alias",
        action="store_false",
        dest="add_predicted_alias",
        help="Do not create legacy layer 'predicted'.",
    )
    parser.add_argument(
        "--prediction_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Dtype used when writing prediction layers.",
    )
    parser.add_argument(
        "--notebook_style_full_output",
        action="store_true",
        help=(
            "Write full adata_test shape and fill predicted values by obs index "
            "for compatibility with downstream evaluation."
        ),
    )
    parser.add_argument(
        "--no_notebook_style_full_output",
        action="store_false",
        dest="notebook_style_full_output",
        help="Write only predicted subsets instead of full adata_test.",
    )
    parser.add_argument(
        "--min_cells_per_group",
        type=int,
        default=10,
        help="Skip (drug, cell_type) groups with <= this many cells.",
    )
    parser.add_argument(
        "--max_cells_per_group",
        type=int,
        default=2000,
        help="Cap sampled cells per (drug, cell_type) group.",
    )
    parser.add_argument(
        "--prediction_seed",
        type=int,
        default=2025,
        help="Random seed for group subsampling and library sampling.",
    )
    parser.add_argument(
        "--library_source",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Source adata used to sample scVI library sizes.",
    )
    parser.add_argument(
        "--deterministic_decode",
        action="store_true",
        help="Use deterministic scVI decoder output (px_rate).",
    )
    parser.add_argument(
        "--strict_cond_alignment",
        action="store_true",
        help="Fail-fast if model conditional builder fails instead of silent concat fallback.",
    )
    parser.add_argument(
        "--allow_cond_fallback",
        action="store_false",
        dest="strict_cond_alignment",
        help="Allow legacy condition concat fallback if builder fails.",
    )
    parser.add_argument(
        "--conditional_vector_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "full", "simple", "adapter", "grca"],
        help="Condition vector build mode passed to sample_data. `auto` reads cinn run_config.json.",
    )
    parser.set_defaults(
        add_predicted_alias=True,
        notebook_style_full_output=True,
        strict_cond_alignment=False,
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA unavailable, falling back to CPU.")
        return "cpu"
    return device_arg


def _resolve_model_path(paths: Dict[str, Any], fold: int, test_size: float, model_path: str) -> str:
    if model_path:
        return model_path
    test_size_str = str(test_size)
    template = paths.get("model_save_dir")
    if not template:
        raise ValueError("paths.model_save_dir missing in config.")
    return str(template).format(fold=fold, test_size=test_size_str)


def _resolve_similars_map_path(
    fold: int,
    test_size: float,
    similars_map_path_arg: str,
) -> str:
    if similars_map_path_arg:
        return similars_map_path_arg
    return str(ARTIFACT_DIR / "nips" / f"{test_size}" / f"similars_map_fold{fold}.json")


def _load_model_run_config(model_path: str) -> Dict[str, Any]:
    run_cfg_path = Path(model_path) / "run_config.json"
    if not run_cfg_path.exists():
        return {}
    try:
        return json.loads(run_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_conditional_vector_mode(
    mode_arg: str,
    run_cfg: Dict[str, Any],
) -> str:
    mode = mode_arg
    if mode == "auto":
        cfg_mode = run_cfg.get("conditional_vector_mode", "none")
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


def _resolve_scvi_model_path(
    paths: Dict[str, Any],
    fold: int,
    test_size: float,
    run_cfg: Dict[str, Any],
    scvi_model_dir_arg: str,
) -> str:
    if scvi_model_dir_arg:
        return scvi_model_dir_arg
    if run_cfg.get("scvi_model_dir"):
        return str(run_cfg["scvi_model_dir"])
    return os.path.join(paths["scvi_model_dir"], f"fold{fold}_{test_size}")


def _resolve_cache_dir(
    cache_dir_arg: str,
    config_cache_dir: Optional[str],
    run_cfg: Optional[Dict[str, Any]],
    fold: int,
    test_size: float,
) -> str:
    cache_key = f"{test_size}_{fold}"
    if cache_dir_arg:
        return cache_dir_arg

    if run_cfg and run_cfg.get("cached_cov_dir"):
        cand = str(run_cfg["cached_cov_dir"])
        tf = os.path.join(cand, f"train_fit_{cache_key}.pkl")
        ec = os.path.join(cand, f"enhanced_cov_train_{cache_key}.npy")
        if os.path.exists(tf) and os.path.exists(ec):
            return cand

    candidates = [
        config_cache_dir,
        str(ARTIFACT_DIR / "nips" / f"cached_data_{test_size}"),
        str(ARTIFACT_DIR / "nips" / "cached_data"),
    ]
    for cand in candidates:
        if not cand:
            continue
        tf = os.path.join(cand, f"train_fit_{cache_key}.pkl")
        ec = os.path.join(cand, f"enhanced_cov_train_{cache_key}.npy")
        if os.path.exists(tf) and os.path.exists(ec):
            return cand
    if run_cfg and run_cfg.get("cached_cov_dir"):
        return str(run_cfg["cached_cov_dir"])
    return config_cache_dir or ""


def _load_train_fit(cache_dir: Optional[str], fold: int, test_size: float) -> Optional[Dict[str, Any]]:
    if not cache_dir:
        return None
    cache_key = f"{test_size}_{fold}"
    train_fit_path = os.path.join(cache_dir, f"train_fit_{cache_key}.pkl")
    if not os.path.exists(train_fit_path):
        return None
    with open(train_fit_path, "rb") as f:
        return pickle.load(f)


def _to_numpy_dtype(x: object, dtype: np.dtype) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    return arr.astype(dtype, copy=False)


def _to_dense_np(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _placeholder_first_stage(n_rows: int) -> np.ndarray:
    return np.zeros((int(n_rows), 1), dtype=np.float32)


def _load_enhanced_cov_train(cache_dir: str, fold: int, test_size: float) -> np.ndarray:
    cache_key = f"{test_size}_{fold}"
    path = os.path.join(cache_dir, f"enhanced_cov_train_{cache_key}.npy")
    if not os.path.exists(path):
        raise ValueError(f"Missing enhanced_cov cache: {path}")
    return np.load(path)


def _build_oldflow_cinn_model(
    model_cfg,
    bundle,
    dataset_cfg,
    run_cfg,
    chemvae_model,
    std_model,
    scvi_model,
    device: str,
    cache_dir: str,
    fold: int,
    test_size: float,
    conditioner_norm: str,
    model_path: str,
):
    cond_stage_data, embeddings, perturb_to_embed = prepare_embeddings_cinn(
        bundle.adata_train,
        perturbation_key=dataset_cfg.perturbation_key,
        trt_key=dataset_cfg.trt_key,
        embed_key=dataset_cfg.embed_key,
    )
    enhanced_cov_train = _load_enhanced_cov_train(cache_dir, fold, test_size)

    chemvae_dim = (
        int(chemvae_model.linear_1.out_features)
        if hasattr(chemvae_model, "linear_1")
        else int(model_cfg.conditioning_dim)
    )
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

    conditioner_norm = (conditioner_norm or "bn").lower()
    flow_kwargs = dict(
        conditioning_dim=conditioning_dim,
        embedding_dim=int(run_cfg.get("embedding_dim", model_cfg.embedding_dim)),
        conditioning_depth=int(
            run_cfg.get("conditioning_depth", model_cfg.conditioning_depth)
        ),
        n_flows=int(run_cfg.get("n_flows", model_cfg.n_flows)),
        in_channels=int(run_cfg.get("in_channels", model_cfg.in_channels)),
        hidden_dim=int(run_cfg.get("hidden_dim", model_cfg.hidden_dim)),
        hidden_depth=int(run_cfg.get("hidden_depth", model_cfg.hidden_depth)),
        activation=model_cfg.activation,
        conditioner_use_bn=conditioner_norm == "bn",
        conditioner_use_ln=conditioner_norm == "ln",
        conditioner_use_an=conditioner_norm == "actnorm",
    )
    # Match blocker-time reconstruction for old-flow checkpoints.
    flow_kwargs.pop("conditioner_use_bn", None)
    flow_kwargs.pop("conditioner_use_ln", None)
    flow_kwargs.pop("conditioner_use_an", None)
    supported = set(inspect.signature(ConditionalFlatCouplingFlow.__init__).parameters.keys())
    flow_kwargs = {k: v for k, v in flow_kwargs.items() if k in supported}
    flow_model = ConditionalFlatCouplingFlow(**flow_kwargs)

    cov_dim = int(enhanced_cov_train.shape[1])
    cell_type_dim = cov_dim - 1 - 64 if cov_dim > 70 else max(1, cov_dim - 1)
    cell_type_dim = int(run_cfg.get("cell_type_dim_resolved", cell_type_dim))

    model_cinn = Net2NetFlow_TFVAE_Covariate_Flow_Old(
        configured_flow=flow_model,
        first_stage_data=_placeholder_first_stage(bundle.adata_train.n_obs),
        cond_stage_data=cond_stage_data,
        perturbToOnehotLib=perturb_to_embed,
        oneHotData=embeddings,
        model_con=chemvae_model,
        std_model=std_model,
        covariates=enhanced_cov_train,
        scvi_model=scvi_model,
        n_epochs=model_cfg.n_epochs,
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


def _align_bundle_to_scvi_vars(bundle, scvi_model_path: str):
    var_path = Path(scvi_model_path) / "var_names.csv"
    if not var_path.exists():
        return bundle

    try:
        var_names = np.loadtxt(var_path, delimiter=",", dtype=str)
    except Exception:
        return bundle

    train_var = bundle.adata_train.var_names.astype(str)
    var_to_idx = {g: i for i, g in enumerate(train_var)}
    keep = [var_to_idx[g] for g in var_names if g in var_to_idx]
    if not keep:
        return bundle

    bundle.adata_train = bundle.adata_train[:, keep].copy()
    bundle.adata_test = bundle.adata_test[:, keep].copy()
    print(
        f"Aligned adata vars to scVI var_names: n_vars={bundle.adata_train.n_vars}"
    )
    return bundle


def _build_notebook_style_output(
    adata_test,
    adata_pred,
    pred_layer: str,
    out_dtype: np.dtype,
):
    adata_out = adata_test.copy()
    pred_full = np.zeros((adata_out.n_obs, adata_out.n_vars), dtype=out_dtype)
    pred_part = _to_numpy_dtype(adata_pred.layers[pred_layer], out_dtype)
    out_idx = adata_out.obs_names.get_indexer(adata_pred.obs_names)
    valid = out_idx >= 0
    if pred_part.shape[0] != valid.sum():
        raise ValueError(
            f"Prediction row mismatch: pred_rows={pred_part.shape[0]} vs matched_rows={valid.sum()}"
        )
    pred_full[out_idx[valid]] = pred_part
    adata_out.layers["prediction"] = pred_full
    for k in ("deg_indices", "control_means"):
        if k in adata_pred.uns:
            adata_out.uns[k] = adata_pred.uns[k]
    return adata_out


def _build_name_to_smiles(
    adata_test,
    perturbation_col: str,
    perturbation_key: str,
    adata_train=None,
) -> Dict[str, str]:
    name_to_smiles: Dict[str, str] = {}

    def _update_map(adata_obj):
        if perturbation_col not in adata_obj.obs or perturbation_key not in adata_obj.obs:
            return
        pairs = adata_obj.obs[[perturbation_col, perturbation_key]].drop_duplicates()
        for _, row in pairs.iterrows():
            name_to_smiles[str(row[perturbation_col])] = str(row[perturbation_key])

    _update_map(adata_test)
    if adata_train is not None:
        _update_map(adata_train)
    return name_to_smiles


def _compute_control_means(
    adata_train,
    cell_type_col: str,
    control_label: str,
    perturbation_col: str,
) -> Dict[str, np.ndarray]:
    control_means: Dict[str, np.ndarray] = {}
    for cell_type in adata_train.obs[cell_type_col].unique():
        mask = (
            (adata_train.obs[perturbation_col].astype(str) == str(control_label))
            & (adata_train.obs[cell_type_col].astype(str) == str(cell_type))
        )
        if int(mask.sum()) <= 0:
            continue
        y_ctrl = adata_train[mask].layers.get("counts", adata_train[mask].X)
        if sparse.issparse(y_ctrl):
            y_ctrl = y_ctrl.toarray()
        else:
            y_ctrl = np.asarray(y_ctrl)
        control_means[str(cell_type)] = y_ctrl.mean(axis=0)
    return control_means


def _compute_deg_indices_nips(
    adata_test,
    perturbations,
    cell_types,
    perturbation_col: str,
    n_deg_genes: int = 50,
) -> Dict[str, np.ndarray]:
    deg_dict: Dict[str, np.ndarray] = {}
    for pert in perturbations:
        for ct in cell_types:
            key = f"{ct}_{pert}"
            try:
                deg_dict[key] = get_deg_indices_from_precomputed(
                    adata_test,
                    str(pert),
                    n_deg_genes,
                    "nips",
                    str(ct),
                )
            except (ValueError, KeyError):
                continue
    return deg_dict


def _resolve_batch_data_for_indices(
    adata_test,
    idx_trt_type: np.ndarray,
):
    if "_scvi_batch" not in adata_test.obs.columns:
        return None
    return (
        adata_test.obs["_scvi_batch"]
        .to_numpy()
        .astype(np.int64)[idx_trt_type]
        .reshape(-1, 1)
    )


def _generate_predictions_nips_notebook_aligned(
    model_cinn,
    adata_test,
    adata_train,
    perturbation_col: str,
    cell_type_col: str,
    perturbation_key: str,
    trt_key: str,
    embed_key: str,
    dose_col: str,
    std_model,
    scvi_model,
    train_fit: Dict[str, Any],
    similars_map: Dict[str, Any],
    device: str,
    control_label: str,
    out_dtype: np.dtype,
    min_cells_per_group: int,
    max_cells_per_group: int,
    prediction_seed: int,
    library_source: str,
    notebook_style_full_output: bool,
    deterministic_decode: bool,
    strict_cond_alignment: bool,
    conditional_vector_mode: str,
):
    model_cinn.eval()
    model_g = model_cinn.model_con
    model_g.eval()

    rng = np.random.RandomState(prediction_seed)

    if trt_key not in adata_test.uns:
        raise KeyError(f"{trt_key} not found in adata_test.uns")
    if embed_key not in adata_test.uns:
        raise KeyError(f"{embed_key} not found in adata_test.uns")

    perturbnet_model = SCVIZ_CheckNet2Net(
        model_cinn, device, scvi_predictive_z(scvi_model)
    )

    n_batch = int(getattr(scvi_model.model, "n_batch", 1))
    if "_scvi_batch" not in adata_test.obs.columns:
        if n_batch > 1:
            raise ValueError(
                "adata_test is missing '_scvi_batch' while scVI has n_batch>1. "
                "Please ensure batch indices are prepared consistently with training setup."
            )
        # Notebook-compatible fallback for single-batch models.
        scvi.data.setup_anndata(adata_test, layer="counts")

    trt_lookup = np.asarray(adata_test.uns[trt_key]).astype(str)
    trt_to_idx = {k: i for i, k in enumerate(trt_lookup)}
    onehot_lookup = np.asarray(adata_test.uns[embed_key])

    perturb_with_onehot_removed = np.asarray(adata_test.obs[perturbation_key]).astype(str)
    trt_unseen_list = np.unique(perturb_with_onehot_removed)
    unique_cell_types = np.asarray(adata_test.obs[cell_type_col]).astype(str)
    cell_types = np.unique(unique_cell_types)

    name_to_smiles = _build_name_to_smiles(
        adata_test=adata_test,
        perturbation_col=perturbation_col,
        perturbation_key=perturbation_key,
        adata_train=adata_train,
    )

    if library_source == "train":
        lib_pool_adata = adata_train
    else:
        lib_pool_adata = adata_test
    lsample_obs = scvi_model.get_latent_library_size(
        adata=lib_pool_adata,
        give_mean=False,
    )

    skipped_small = 0
    skipped_control = 0
    predicted_groups = 0
    total_groups = 0
    result_adatas = []

    if notebook_style_full_output:
        adata_result = adata_test.copy()
        pred_full = np.full(
            (adata_result.n_obs, adata_result.n_vars),
            np.nan,
            dtype=out_dtype,
        )

    for trt_type in tqdm(trt_unseen_list, desc="Predicting (full-layout)"):
        if trt_type not in trt_to_idx:
            continue
        idx = trt_to_idx[trt_type]

        mask_trt = perturb_with_onehot_removed == trt_type
        pert_vals = adata_test.obs.loc[mask_trt, perturbation_col].astype(str).unique()
        if len(pert_vals) == 0:
            continue
        pert = str(pert_vals[0])
        if pert == str(control_label):
            skipped_control += 1
            continue

        onehot_indice_trt_single = onehot_lookup[[idx]]
        with torch.no_grad():
            _, _, _, embdata_torch_single = model_g(
                torch.tensor(onehot_indice_trt_single).float().to(device)
            )
            embdata_np_single = std_model.standardize_z(
                embdata_torch_single.cpu().detach().numpy()
            )

        idx_trt_all = np.where(mask_trt)[0]
        for c_type in cell_types:
            total_groups += 1
            idx_ct = np.where(unique_cell_types == str(c_type))[0]
            idx_trt_type = np.intersect1d(idx_trt_all, idx_ct)
            if idx_trt_type.shape[0] > int(max_cells_per_group):
                idx_trt_type = rng.choice(
                    idx_trt_type, int(max_cells_per_group), replace=False
                )
            if idx_trt_type.shape[0] <= int(min_cells_per_group):
                skipped_small += 1
                continue

            n_samples = len(idx_trt_type)
            embdata_np = np.tile(embdata_np_single, (n_samples, 1))
            lsample_idx = rng.choice(lsample_obs.shape[0], n_samples, replace=True)
            library_trt_latent = lsample_obs[lsample_idx]

            adata_subset = adata_test[idx_trt_type]
            unseen_names = [pert] * n_samples
            enhanced_cov_unseen = build_unseen_enhanced_cov_with_similars(
                adata_meta_like=adata_subset,
                unseen_names=unseen_names,
                similars_map=similars_map,
                train_fit=train_fit,
                cell_type_col=cell_type_col,
                dose_col=dose_col,
                name_to_smiles=name_to_smiles,
            )

            batch_trt = _resolve_batch_data_for_indices(adata_test, idx_trt_type)
            with torch.no_grad():
                _, predict_data = perturbnet_model.sample_data(
                    z_drug_std=embdata_np,
                    covariates=enhanced_cov_unseen,
                    library_data=library_trt_latent,
                    cell_type_dim=6,
                    batch_data=batch_trt,
                    deterministic=bool(deterministic_decode),
                    strict_cond_alignment=bool(strict_cond_alignment),
                    conditional_vector=conditional_vector_mode,
                )

            predict_data = np.asarray(predict_data, dtype=out_dtype)
            predicted_groups += 1

            if notebook_style_full_output:
                pred_full[idx_trt_type, :] = predict_data
            else:
                subset = adata_subset.copy()
                subset.layers["prediction"] = predict_data
                result_adatas.append(subset)

    if notebook_style_full_output:
        adata_result.layers["prediction"] = pred_full
    else:
        if not result_adatas:
            raise ValueError("No predicted groups produced; check split/model/similars_map.")
        import anndata as ad  # lazy import for concat path
        adata_result = ad.concat(result_adatas)

    adata_result.uns["control_means"] = _compute_control_means(
        adata_train=adata_train,
        cell_type_col=cell_type_col,
        control_label=control_label,
        perturbation_col=perturbation_col,
    )
    adata_result.uns["deg_indices"] = _compute_deg_indices_nips(
        adata_test=adata_test,
        perturbations=np.unique(adata_test.obs[perturbation_col].astype(str)),
        cell_types=np.unique(adata_test.obs[cell_type_col].astype(str)),
        perturbation_col=perturbation_col,
        n_deg_genes=50,
    )

    print(
        "Notebook-aligned generation summary: "
        f"predicted_groups={predicted_groups}, total_groups={total_groups}, "
        f"skipped_small={skipped_small}, skipped_control={skipped_control}"
    )
    return adata_result


def main() -> None:
    args = _parse_args()
    dataset_cfg, model_cfg, training_cfg, paths = load_full_nips_config(args.config)

    # Backfill legacy NIPS fields used by the historical export path.
    if not hasattr(dataset_cfg, "nips_perturbation_col"):
        dataset_cfg.nips_perturbation_col = "condition"
    if not hasattr(dataset_cfg, "perturbation_key"):
        dataset_cfg.perturbation_key = "condition"
    if not hasattr(dataset_cfg, "trt_key"):
        dataset_cfg.trt_key = "ordered_all_condition_chemvae"
    if not hasattr(dataset_cfg, "embed_key"):
        dataset_cfg.embed_key = "ordered_all_onehot_chemvae"

    dataset_cfg.fold = args.fold
    dataset_cfg.test_size = args.test_size
    if args.adata_path:
        dataset_cfg.h5ad_path = args.adata_path
    if args.split_dir:
        dataset_cfg.split_dir = args.split_dir

    training_cfg.device = _resolve_device(args.device)
    model_cfg.use_enhanced_cov = True
    model_cfg.use_llm_similars = True
    model_path = _resolve_model_path(paths, dataset_cfg.fold, dataset_cfg.test_size, args.model_path)
    run_cfg = _load_model_run_config(model_path)
    if run_cfg:
        if (not args.adata_path) and run_cfg.get("adata_path"):
            dataset_cfg.h5ad_path = str(run_cfg["adata_path"])
        if (not args.split_dir) and run_cfg.get("split_dir"):
            dataset_cfg.split_dir = str(run_cfg["split_dir"])

    flow_api = str(run_cfg.get("flow_api", "new")).lower()
    if flow_api != "old":
        raise ValueError(
            f"PRIMO release supports the paper old-flow GRCA checkpoints only; got flow_api={flow_api!r}."
        )
    conditioner_norm = str(run_cfg.get("conditioner_norm", "bn")).lower()
    conditional_vector_mode = _resolve_conditional_vector_mode(
        mode_arg=args.conditional_vector_mode,
        run_cfg=run_cfg,
    )

    cinn_perturbation_key = str(
        run_cfg.get("cinn_perturbation_key", dataset_cfg.perturbation_key)
    )
    if cinn_perturbation_key in {"SMILES", "condition"}:
        dataset_cfg.perturbation_key = cinn_perturbation_key
    if run_cfg.get("trt_lookup_key"):
        dataset_cfg.trt_key = str(run_cfg["trt_lookup_key"])
    elif dataset_cfg.perturbation_key == "condition":
        dataset_cfg.trt_key = "ordered_all_condition_chemvae"
    else:
        dataset_cfg.trt_key = "ordered_all_smile_chemvae"

    model_cfg.similars_map_path = _resolve_similars_map_path(
        dataset_cfg.fold, dataset_cfg.test_size, args.similars_map_path
    )
    cache_dir = _resolve_cache_dir(
        args.cache_dir,
        paths.get("cache_dir"),
        run_cfg,
        dataset_cfg.fold,
        dataset_cfg.test_size,
    )
    scvi_model_path = _resolve_scvi_model_path(
        paths,
        dataset_cfg.fold,
        dataset_cfg.test_size,
        run_cfg,
        args.scvi_model_dir,
    )

    print(
        f"Resolved flow_api={flow_api}, perturbation_key={dataset_cfg.perturbation_key}, "
        f"trt_key={dataset_cfg.trt_key}, embed_key={dataset_cfg.embed_key}"
    )
    print(f"Resolved conditional_vector_mode={conditional_vector_mode}")
    print(f"Resolved scVI path: {scvi_model_path}")
    print(f"Resolved cache_dir: {cache_dir}")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print("Loading dataset...")
    bundle = build_nips_dataset(dataset_cfg)
    bundle = _align_bundle_to_scvi_vars(bundle, scvi_model_path)
    print(f"Test cells: {bundle.adata_test.n_obs}, genes: {bundle.adata_test.n_vars}")

    print("Loading prerequisite models...")
    chem_onehot = np.load(dataset_cfg.chem_onehot_path)
    chemvae_model = ChemicalVAE(
        n_char=chem_onehot.shape[2],
        max_len=chem_onehot.shape[1],
    ).to(training_cfg.device)
    chemvae_model.load_state_dict(torch.load(paths["chemvae_model"], map_location=training_cfg.device))
    chemvae_model.eval()

    scvi.data.setup_anndata(bundle.adata_train, layer="counts")
    scvi_model = scvi.model.SCVI.load(
        scvi_model_path,
        bundle.adata_train,
        use_cuda=(str(training_cfg.device).startswith("cuda") and torch.cuda.is_available()),
    )

    std_model = load_std_model(paths["std_param_dir"], training_cfg.device)

    print("Building CINN architecture...")
    model_cinn = _build_oldflow_cinn_model(
        model_cfg=model_cfg,
        bundle=bundle,
        dataset_cfg=dataset_cfg,
        run_cfg=run_cfg,
        chemvae_model=chemvae_model,
        std_model=std_model,
        scvi_model=scvi_model,
        device=training_cfg.device,
        cache_dir=cache_dir,
        fold=dataset_cfg.fold,
        test_size=dataset_cfg.test_size,
        conditioner_norm=conditioner_norm,
        model_path=model_path,
    )

    train_fit = _load_train_fit(cache_dir, dataset_cfg.fold, dataset_cfg.test_size)
    if train_fit is None:
        raise ValueError(
            "Missing train_fit cache for enhanced covariates. "
            "Expected under cache_dir/train_fit_{test_size}_{fold}.pkl"
        )

    similars_map = None
    if os.path.exists(model_cfg.similars_map_path):
        with open(model_cfg.similars_map_path, "r", encoding="utf-8") as f:
            similars_map = json.load(f)
        print(f"Loaded similars_map with {len(similars_map)} entries.")
    else:
        raise ValueError(f"similars_map not found: {model_cfg.similars_map_path}")

    print("Generating predictions...")
    out_dtype = np.dtype(args.prediction_dtype)
    adata_result = _generate_predictions_nips_notebook_aligned(
        model_cinn=model_cinn,
        adata_test=bundle.adata_test,
        adata_train=bundle.adata_train,
        perturbation_col=dataset_cfg.nips_perturbation_col,
        cell_type_col=dataset_cfg.cell_type_col,
        perturbation_key=dataset_cfg.perturbation_key,
        trt_key=dataset_cfg.trt_key,
        embed_key=dataset_cfg.embed_key,
        dose_col=dataset_cfg.dose_col,
        std_model=std_model,
        scvi_model=scvi_model,
        train_fit=train_fit,
        similars_map=similars_map,
        device=training_cfg.device,
        control_label="DMSO",
        out_dtype=out_dtype,
        min_cells_per_group=args.min_cells_per_group,
        max_cells_per_group=args.max_cells_per_group,
        prediction_seed=args.prediction_seed,
        library_source=args.library_source,
        notebook_style_full_output=args.notebook_style_full_output,
        deterministic_decode=args.deterministic_decode,
        strict_cond_alignment=args.strict_cond_alignment,
        conditional_vector_mode=conditional_vector_mode,
    )

    if "prediction" not in adata_result.layers:
        raise ValueError("Generated adata is missing layer 'prediction'.")

    pred_arr = _to_numpy_dtype(adata_result.layers["prediction"], out_dtype)
    adata_result.layers["prediction"] = pred_arr
    adata_result.layers["predicted_counts"] = pred_arr.copy()
    if args.add_predicted_alias:
        adata_result.layers["predicted"] = pred_arr.copy()

    nnz_ratio = float(np.mean(pred_arr != 0))
    print(
        f"prediction dtype={pred_arr.dtype}, shape={pred_arr.shape}, "
        f"nnz_ratio={nnz_ratio:.6f}"
    )

    print(f"Writing h5ad -> {args.output_file}")
    adata_result.write_h5ad(args.output_file)
    print("Done.")


if __name__ == "__main__":
    main()
