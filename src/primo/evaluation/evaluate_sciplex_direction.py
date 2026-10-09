#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from primo.paths import DATA_DIR, OUTPUT_DIR

DEFAULT_REFERENCE_ADATA_PATH = str(DATA_DIR / "sciplex" / "sciplex_processed.h5ad")
DEFAULT_INPUT_H5AD_PATH = str(OUTPUT_DIR / "sciplex" / "predicted.h5ad")


def _to_dense_2d(x: object) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray(arr, dtype=np.float32)


def _counts_to_x(data_counts: np.ndarray, target_sum: float = 10_000.0) -> np.ndarray:
    arr = np.asarray(data_counts, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    lib = arr.sum(axis=1, keepdims=True)
    lib[~np.isfinite(lib) | (lib == 0)] = 1.0
    return np.log1p(arr / lib * float(target_sum))


def _raw_matrix_from_indices(
    adata_obj: ad.AnnData,
    row_idx: np.ndarray,
    layer_name: Optional[str],
) -> np.ndarray:
    if row_idx.size == 0:
        return np.zeros((0, adata_obj.n_vars), dtype=np.float32)
    if layer_name is None:
        mat = adata_obj.X[row_idx]
    else:
        mat = adata_obj.layers[layer_name][row_idx]
    return _to_dense_2d(mat)


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if x.size < 2 or y.size < 2 or x.shape != y.shape:
        return np.nan
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        if kind == "pearson":
            v, _ = pearsonr(x, y)
        elif kind == "spearman":
            v, _ = spearmanr(x, y)
        else:
            raise ValueError(f"Unsupported corr kind: {kind}")
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _safe_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2 or x.shape != y.shape:
        return np.nan
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    if np.std(x) == 0:
        return np.nan
    try:
        v = r2_score(x, y)
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _profile_metrics(real_mat: np.ndarray, pred_mat: np.ndarray) -> Tuple[float, float]:
    real_arr = np.asarray(real_mat, dtype=np.float32)
    pred_arr = np.asarray(pred_mat, dtype=np.float32)
    if real_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan, np.nan

    real_arr = real_arr[real_arr.sum(axis=1) != 0]
    pred_arr = pred_arr[pred_arr.sum(axis=1) != 0]
    if real_arr.shape[0] == 0 or pred_arr.shape[0] == 0:
        return np.nan, np.nan

    true_vec = real_arr.mean(axis=0)
    pred_vec = pred_arr.mean(axis=0)
    return _safe_r2(true_vec, pred_vec), _safe_corr(true_vec, pred_vec, "pearson")


def _profile_metrics_norm_by_all(
    real_raw: np.ndarray,
    pred_raw: np.ndarray,
    target_sum: float,
    gene_idx: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    real_arr = np.asarray(real_raw, dtype=np.float32)
    pred_arr = np.asarray(pred_raw, dtype=np.float32)
    if real_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan, np.nan

    real_lib = real_arr.sum(axis=1)
    pred_lib = pred_arr.sum(axis=1)

    if gene_idx is not None:
        real_arr = real_arr[:, gene_idx]
        pred_arr = pred_arr[:, gene_idx]

    real_keep = real_lib != 0
    pred_keep = pred_lib != 0
    real_arr = real_arr[real_keep]
    pred_arr = pred_arr[pred_keep]
    real_lib = real_lib[real_keep]
    pred_lib = pred_lib[pred_keep]
    if real_arr.shape[0] == 0 or pred_arr.shape[0] == 0:
        return np.nan, np.nan

    real_norm = np.log1p(real_arr / real_lib[:, None] * float(target_sum))
    pred_norm = np.log1p(pred_arr / pred_lib[:, None] * float(target_sum))
    true_vec = real_norm.mean(axis=0)
    pred_vec = pred_norm.mean(axis=0)
    return _safe_r2(true_vec, pred_vec), _safe_corr(true_vec, pred_vec, "pearson")


def _direction_acc(true_delta: np.ndarray, pred_delta: np.ndarray) -> float:
    if true_delta.size == 0 or pred_delta.size == 0 or true_delta.shape != pred_delta.shape:
        return np.nan
    if np.isnan(true_delta).any() or np.isnan(pred_delta).any():
        return np.nan
    return float((np.sign(true_delta) == np.sign(pred_delta)).mean())


def _series_stat(df: pd.DataFrame, column: str, fn: str) -> float:
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return np.nan
    if fn == "mean":
        return float(series.mean())
    if fn == "median":
        return float(series.median())
    raise ValueError(f"Unsupported stat function: {fn}")


def _resolve_prediction_layer(adata_pred: ad.AnnData, requested: str) -> str:
    if requested != "auto":
        if requested not in adata_pred.layers:
            raise KeyError(f"Requested prediction layer '{requested}' not found. layers={list(adata_pred.layers.keys())}")
        return requested
    for key in ("predicted", "prediction"):
        if key in adata_pred.layers:
            return key
    raise KeyError(f"No prediction layer found. layers={list(adata_pred.layers.keys())}")


def _resolve_split_key(
    adata_ref: ad.AnnData,
    pred_indices: Sequence[int],
    fold: Optional[int],
    split_key: Optional[str],
    prediction_split: str,
    good_smile_col: str,
) -> Tuple[str, Dict[str, object]]:
    pred_index_set = set(int(x) for x in pred_indices)

    if split_key:
        resolved = str(split_key)
    elif fold is not None:
        resolved = f"split_82_{int(fold)}_cpa"
    else:
        candidates: List[Tuple[str, str, int, int, int]] = []
        obs = adata_ref.obs
        good_mask = pd.to_numeric(obs[good_smile_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        for col in obs.columns:
            if not (str(col).startswith("split_82_") and str(col).endswith("_cpa")):
                continue
            eval_mask = good_mask & (obs[col].astype(str).to_numpy() == str(prediction_split))
            eval_indices = set(int(x) for x in obs.loc[eval_mask, "indices"].to_numpy())
            inter = len(pred_index_set & eval_indices)
            pred_only = len(pred_index_set - eval_indices)
            split_only = len(eval_indices - pred_index_set)
            if pred_index_set == eval_indices:
                candidates.append((str(col), "exact", inter, pred_only, split_only))
            elif pred_index_set.issubset(eval_indices):
                candidates.append((str(col), "subset", inter, pred_only, split_only))
        if not candidates:
            raise ValueError("Could not auto-resolve split_key from predicted indices.")
        candidates.sort(key=lambda x: (0 if x[1] == "exact" else 1, x[3], x[4], x[0]))
        resolved = candidates[0][0]

    obs = adata_ref.obs
    good_mask = pd.to_numeric(obs[good_smile_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    eval_mask = good_mask & (obs[resolved].astype(str).to_numpy() == str(prediction_split))
    eval_indices = set(int(x) for x in obs.loc[eval_mask, "indices"].to_numpy())
    match_report = {
        "split_key": resolved,
        "prediction_split": str(prediction_split),
        "n_pred_indices": int(len(pred_index_set)),
        "n_split_indices": int(len(eval_indices)),
        "n_intersection": int(len(pred_index_set & eval_indices)),
        "n_pred_only": int(len(pred_index_set - eval_indices)),
        "n_split_only": int(len(eval_indices - pred_index_set)),
        "match_mode": (
            "exact"
            if pred_index_set == eval_indices
            else "subset"
            if pred_index_set.issubset(eval_indices)
            else "partial"
        ),
    }
    return resolved, match_report


def _matrix_from_indices(
    adata_obj: ad.AnnData,
    row_idx: np.ndarray,
    layer_name: Optional[str],
    target_sum: float,
    source_mode: str,
) -> np.ndarray:
    if row_idx.size == 0:
        return np.zeros((0, adata_obj.n_vars), dtype=np.float32)
    if source_mode == "counts_to_x":
        if layer_name is None:
            raise KeyError("counts_to_x mode requires a source layer.")
        mat = adata_obj.layers[layer_name][row_idx]
        return _counts_to_x(_to_dense_2d(mat), target_sum=target_sum)
    if layer_name is None:
        mat = adata_obj.X[row_idx]
    else:
        mat = adata_obj.layers[layer_name][row_idx]
    return _to_dense_2d(mat)


def _build_deg_map(adata_ref: ad.AnnData, degs_key: str, topk: int) -> Dict[str, np.ndarray]:
    if degs_key not in adata_ref.uns:
        raise KeyError(f"Missing uns key: {degs_key}")
    rg = adata_ref.uns[degs_key]
    if "names" not in rg:
        raise KeyError(f"Missing '{degs_key}[\"names\"]'")
    names = rg["names"]
    if not getattr(names, "dtype", None) or names.dtype.names is None:
        raise ValueError(f"{degs_key}['names'] must be a structured array.")

    gene_to_idx = {str(g): i for i, g in enumerate(adata_ref.var_names.astype(str))}
    out: Dict[str, np.ndarray] = {}
    for treatment in names.dtype.names:
        genes = [str(x) for x in np.asarray(names[treatment]).tolist() if str(x) in gene_to_idx]
        out[str(treatment)] = np.asarray([gene_to_idx[g] for g in genes[:topk]], dtype=int)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SciPlex ours predicted h5ad on cell_type+treatment DE-direction metrics."
    )
    p.add_argument("--input_h5ad_path", type=str, default=DEFAULT_INPUT_H5AD_PATH)
    p.add_argument("--reference_adata_path", type=str, default=DEFAULT_REFERENCE_ADATA_PATH)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--output_prefix", type=str, default="ours_direction")

    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--split_key", type=str, default=None)
    p.add_argument("--prediction_split", type=str, default="ood")
    p.add_argument("--good_smile_col", type=str, default="good_smile_chemvae")

    p.add_argument("--cell_type_col", type=str, default="cell_type")
    p.add_argument("--treatment_col", type=str, default="treatment")
    p.add_argument("--control_label", type=str, default="Vehicle")
    p.add_argument("--control_col", type=str, default="product_name_clean")
    p.add_argument("--degs_key", type=str, default="rank_genes_groups")
    p.add_argument("--topk", type=int, default=50)

    p.add_argument("--prediction_layer", type=str, default="auto")
    p.add_argument("--reference_source", type=str, default="counts_to_x", choices=["counts_to_x", "x"])
    p.add_argument("--prediction_source", type=str, default="counts_to_x", choices=["counts_to_x", "x"])
    p.add_argument("--target_size", type=float, default=10000.0)
    p.add_argument("--min_true_cells", type=int, default=5)
    p.add_argument("--min_pred_cells", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_ref = ad.read_h5ad(args.reference_adata_path)
    adata_pred = ad.read_h5ad(args.input_h5ad_path)
    pred_layer = _resolve_prediction_layer(adata_pred, args.prediction_layer)

    required_ref_obs = [
        "indices",
        args.good_smile_col,
        args.cell_type_col,
        args.treatment_col,
        args.control_col,
    ]
    for col in required_ref_obs:
        if col not in adata_ref.obs.columns:
            raise KeyError(f"Missing reference obs column: {col}")
    required_pred_obs = ["indices", args.cell_type_col, args.treatment_col]
    for col in required_pred_obs:
        if col not in adata_pred.obs.columns:
            raise KeyError(f"Missing prediction obs column: {col}")

    pred_indices = adata_pred.obs["indices"].to_numpy()
    split_key, split_report = _resolve_split_key(
        adata_ref=adata_ref,
        pred_indices=pred_indices,
        fold=args.fold,
        split_key=args.split_key,
        prediction_split=args.prediction_split,
        good_smile_col=args.good_smile_col,
    )

    deg_map = _build_deg_map(adata_ref, args.degs_key, int(args.topk))
    pred_idx_set = set(int(x) for x in pred_indices)
    ref_obs = adata_ref.obs
    pred_obs = adata_pred.obs

    good_mask = (
        pd.to_numeric(ref_obs[args.good_smile_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    )
    eval_mask = good_mask & (ref_obs[split_key].astype(str).to_numpy() == str(args.prediction_split))

    ref_cell_type = ref_obs[args.cell_type_col].astype(str).to_numpy()
    ref_treatment = ref_obs[args.treatment_col].astype(str).to_numpy()
    ref_indices = ref_obs["indices"].astype(int).to_numpy()
    pred_cell_type = pred_obs[args.cell_type_col].astype(str).to_numpy()
    pred_treatment = pred_obs[args.treatment_col].astype(str).to_numpy()
    pred_indices_arr = pred_obs["indices"].astype(int).to_numpy()

    control_mask = ref_obs[args.control_col].astype(str).to_numpy() == str(args.control_label)
    control_cache: Dict[str, np.ndarray] = {}
    for cell_type in sorted(np.unique(ref_cell_type[control_mask]).tolist()):
        idx_ctrl = np.where(control_mask & (ref_cell_type == str(cell_type)))[0]
        control_cache[str(cell_type)] = _matrix_from_indices(
            adata_obj=adata_ref,
            row_idx=idx_ctrl,
            layer_name="counts" if args.reference_source == "counts_to_x" else None,
            target_sum=float(args.target_size),
            source_mode=str(args.reference_source),
        ).mean(axis=0)

    pair_df = (
        ref_obs.loc[eval_mask, [args.cell_type_col, args.treatment_col]]
        .astype(str)
        .drop_duplicates()
        .sort_values([args.cell_type_col, args.treatment_col])
        .reset_index(drop=True)
    )

    rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []

    for row in pair_df.itertuples(index=False):
        cell_type = str(getattr(row, args.cell_type_col))
        treatment = str(getattr(row, args.treatment_col))
        deg_idx = deg_map.get(treatment)

        true_row_idx = np.where(eval_mask & (ref_cell_type == cell_type) & (ref_treatment == treatment))[0]
        pred_row_idx = np.where((pred_cell_type == cell_type) & (pred_treatment == treatment))[0]

        reason = ""
        if deg_idx is None or deg_idx.size == 0:
            reason = "missing_deg_genes"
        elif cell_type not in control_cache:
            reason = "missing_celltype_control"
        elif true_row_idx.size < int(args.min_true_cells):
            reason = "small_true_group"
        elif pred_row_idx.size < int(args.min_pred_cells):
            reason = "missing_prediction_pair"

        pred_pair_indices = set(int(x) for x in pred_indices_arr[pred_row_idx])
        true_pair_indices = set(int(x) for x in ref_indices[true_row_idx])
        coverage_rows.append(
            {
                "cell_type": cell_type,
                "treatment": treatment,
                "n_true_cells": int(true_row_idx.size),
                "n_pred_cells": int(pred_row_idx.size),
                "index_intersection": int(len(pred_pair_indices & true_pair_indices)),
                "pred_coverage_ratio": float(len(pred_pair_indices & true_pair_indices) / max(len(true_pair_indices), 1)),
                "used_for_eval": reason == "",
                "skip_reason": reason,
            }
        )
        if reason:
            continue

        true_mat = _matrix_from_indices(
            adata_obj=adata_ref,
            row_idx=true_row_idx,
            layer_name="counts" if args.reference_source == "counts_to_x" else None,
            target_sum=float(args.target_size),
            source_mode=str(args.reference_source),
        )
        pred_mat = _matrix_from_indices(
            adata_obj=adata_pred,
            row_idx=pred_row_idx,
            layer_name=pred_layer if args.prediction_source == "counts_to_x" else pred_layer,
            target_sum=float(args.target_size),
            source_mode=str(args.prediction_source),
        )
        true_raw_mat = _raw_matrix_from_indices(
            adata_obj=adata_ref,
            row_idx=true_row_idx,
            layer_name="counts" if args.reference_source == "counts_to_x" else None,
        )
        pred_raw_mat = _raw_matrix_from_indices(
            adata_obj=adata_pred,
            row_idx=pred_row_idx,
            layer_name=pred_layer,
        )

        true_vec = true_mat.mean(axis=0)
        pred_vec = pred_mat.mean(axis=0)
        ctrl_vec = control_cache[cell_type]
        true_delta_all = true_vec - ctrl_vec
        pred_delta_all = pred_vec - ctrl_vec
        true_delta_deg = true_delta_all[deg_idx]
        pred_delta_deg = pred_delta_all[deg_idx]
        if (
            str(args.reference_source) == "counts_to_x"
            and str(args.prediction_source) == "counts_to_x"
        ):
            r2_all, pear_all = _profile_metrics_norm_by_all(
                true_raw_mat,
                pred_raw_mat,
                target_sum=float(args.target_size),
                gene_idx=None,
            )
            r2_deg, pear_deg = _profile_metrics_norm_by_all(
                true_raw_mat,
                pred_raw_mat,
                target_sum=float(args.target_size),
                gene_idx=deg_idx,
            )
        else:
            r2_all, pear_all = _profile_metrics(true_mat, pred_mat)
            r2_deg, pear_deg = _profile_metrics(true_mat[:, deg_idx], pred_mat[:, deg_idx])

        rows.append(
            {
                "cell_type": cell_type,
                "treatment": treatment,
                "n_true_cells": int(true_row_idx.size),
                "n_pred_cells": int(pred_row_idx.size),
                "n_de_genes": int(deg_idx.size),
                "r2": r2_all,
                "pear": pear_all,
                "r2_deg": r2_deg,
                "pear_deg": pear_deg,
                "direction_accuracy_deg": _direction_acc(true_delta_deg, pred_delta_deg),
                "pr_delta_deg": _safe_corr(true_delta_deg, pred_delta_deg, "pearson"),
                "sp_delta_deg": _safe_corr(true_delta_deg, pred_delta_deg, "spearman"),
                "direction_accuracy_all": _direction_acc(true_delta_all, pred_delta_all),
                "pr_delta_all": _safe_corr(true_delta_all, pred_delta_all, "pearson"),
                "sp_delta_all": _safe_corr(true_delta_all, pred_delta_all, "spearman"),
            }
        )

    pair_columns = [
        "cell_type",
        "treatment",
        "n_true_cells",
        "n_pred_cells",
        "n_de_genes",
        "r2",
        "pear",
        "r2_deg",
        "pear_deg",
        "direction_accuracy_deg",
        "pr_delta_deg",
        "sp_delta_deg",
        "direction_accuracy_all",
        "pr_delta_all",
        "sp_delta_all",
    ]
    coverage_columns = [
        "cell_type",
        "treatment",
        "n_true_cells",
        "n_pred_cells",
        "index_intersection",
        "pred_coverage_ratio",
        "used_for_eval",
        "skip_reason",
    ]
    pair_df_out = pd.DataFrame(rows, columns=pair_columns)
    coverage_df = pd.DataFrame(coverage_rows, columns=coverage_columns)

    aggregate_df = pd.DataFrame(
        [
            {
                "split_key": split_key,
                "prediction_split": str(args.prediction_split),
                "n_pairs_total": int(len(pair_df)),
                "n_pairs_scored": int(pair_df_out.shape[0]),
                "r2_mean": _series_stat(pair_df_out, "r2", "mean"),
                "r2_median": _series_stat(pair_df_out, "r2", "median"),
                "pear_mean": _series_stat(pair_df_out, "pear", "mean"),
                "pear_median": _series_stat(pair_df_out, "pear", "median"),
                "r2_deg_mean": _series_stat(pair_df_out, "r2_deg", "mean"),
                "r2_deg_median": _series_stat(pair_df_out, "r2_deg", "median"),
                "pear_deg_mean": _series_stat(pair_df_out, "pear_deg", "mean"),
                "pear_deg_median": _series_stat(pair_df_out, "pear_deg", "median"),
                "direction_accuracy_deg_mean": _series_stat(pair_df_out, "direction_accuracy_deg", "mean"),
                "direction_accuracy_deg_median": _series_stat(pair_df_out, "direction_accuracy_deg", "median"),
                "pr_delta_deg_mean": _series_stat(pair_df_out, "pr_delta_deg", "mean"),
                "pr_delta_deg_median": _series_stat(pair_df_out, "pr_delta_deg", "median"),
                "sp_delta_deg_mean": _series_stat(pair_df_out, "sp_delta_deg", "mean"),
                "sp_delta_deg_median": _series_stat(pair_df_out, "sp_delta_deg", "median"),
                "direction_accuracy_all_mean": _series_stat(pair_df_out, "direction_accuracy_all", "mean"),
                "direction_accuracy_all_median": _series_stat(pair_df_out, "direction_accuracy_all", "median"),
                "pr_delta_all_mean": _series_stat(pair_df_out, "pr_delta_all", "mean"),
                "pr_delta_all_median": _series_stat(pair_df_out, "pr_delta_all", "median"),
                "sp_delta_all_mean": _series_stat(pair_df_out, "sp_delta_all", "mean"),
                "sp_delta_all_median": _series_stat(pair_df_out, "sp_delta_all", "median"),
            }
        ]
    )

    cell_type_rows: List[Dict[str, object]] = []
    for cell_type, group in pair_df_out.groupby("cell_type", observed=False):
        cell_type_rows.append(
            {
                "cell_type": str(cell_type),
                "n_pairs": int(group.shape[0]),
                "r2_mean": _series_stat(group, "r2", "mean"),
                "r2_median": _series_stat(group, "r2", "median"),
                "pear_mean": _series_stat(group, "pear", "mean"),
                "pear_median": _series_stat(group, "pear", "median"),
                "r2_deg_mean": _series_stat(group, "r2_deg", "mean"),
                "r2_deg_median": _series_stat(group, "r2_deg", "median"),
                "pear_deg_mean": _series_stat(group, "pear_deg", "mean"),
                "pear_deg_median": _series_stat(group, "pear_deg", "median"),
                "direction_accuracy_deg_mean": _series_stat(group, "direction_accuracy_deg", "mean"),
                "direction_accuracy_deg_median": _series_stat(group, "direction_accuracy_deg", "median"),
                "pr_delta_deg_mean": _series_stat(group, "pr_delta_deg", "mean"),
                "pr_delta_deg_median": _series_stat(group, "pr_delta_deg", "median"),
                "sp_delta_deg_mean": _series_stat(group, "sp_delta_deg", "mean"),
                "sp_delta_deg_median": _series_stat(group, "sp_delta_deg", "median"),
                "direction_accuracy_all_mean": _series_stat(group, "direction_accuracy_all", "mean"),
                "direction_accuracy_all_median": _series_stat(group, "direction_accuracy_all", "median"),
                "pr_delta_all_mean": _series_stat(group, "pr_delta_all", "mean"),
                "pr_delta_all_median": _series_stat(group, "pr_delta_all", "median"),
                "sp_delta_all_mean": _series_stat(group, "sp_delta_all", "mean"),
                "sp_delta_all_median": _series_stat(group, "sp_delta_all", "median"),
            }
        )
    cell_type_df = pd.DataFrame(cell_type_rows)

    run_config = {
        **vars(args),
        "prediction_layer_resolved": pred_layer,
        "split_resolution": split_report,
        "n_pairs_total": int(len(pair_df)),
        "n_pairs_scored": int(pair_df_out.shape[0]),
        "n_controls_total": int(control_mask.sum()),
        "n_celltype_controls": int(len(control_cache)),
    }

    pair_path = output_dir / f"{args.output_prefix}_pair_metrics.csv"
    aggregate_path = output_dir / f"{args.output_prefix}_aggregate_metrics.csv"
    cell_path = output_dir / f"{args.output_prefix}_cell_type_summary.csv"
    coverage_path = output_dir / f"{args.output_prefix}_coverage.csv"
    run_cfg_path = output_dir / f"{args.output_prefix}_run_config.json"

    run_cfg_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(run_config, indent=2, ensure_ascii=False))
        return

    pair_df_out.to_csv(pair_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    cell_type_df.to_csv(cell_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)

    print("Saved:")
    print(f"  pair metrics:  {pair_path}")
    print(f"  aggregate:     {aggregate_path}")
    print(f"  cell-type:     {cell_path}")
    print(f"  coverage:      {coverage_path}")
    print(f"  run config:    {run_cfg_path}")
    print("\nAggregate:")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
