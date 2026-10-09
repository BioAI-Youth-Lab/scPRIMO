#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from primo.paths import ARTIFACT_DIR, OUTPUT_DIR

DEFAULT_REFERENCE_ADATA_PATH = str(OUTPUT_DIR / "nips" / "nips_ref_global_hvg5000_globaldeg1000.h5ad")
DEFAULT_OURS_H5AD_PATH = str(OUTPUT_DIR / "nips" / "predicted.h5ad")
DEFAULT_BIOLORD_PRED_PATH = str(ARTIFACT_DIR / "baselines" / "biolord" / "nips_pred_dict.pkl")


def _to_numpy_2d(x: object) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        arr = x.numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray(arr, dtype=np.float32)


def _extract_matrix(adata_obj: ad.AnnData, idx: np.ndarray, layer_name: Optional[str]) -> np.ndarray:
    if idx.size == 0:
        return np.zeros((0, adata_obj.n_vars), dtype=np.float32)
    if layer_name and layer_name in adata_obj.layers:
        mat = adata_obj.layers[layer_name][idx]
    else:
        mat = adata_obj.X[idx]
    if sparse.issparse(mat):
        mat = mat.toarray()
    else:
        mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    return np.asarray(mat, dtype=np.float32)


def _counts_to_x_mean(counts: np.ndarray, target_sum: float) -> np.ndarray:
    if counts.shape[0] == 0:
        return np.array([], dtype=np.float32)
    x = np.nan_to_num(np.asarray(counts, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    lib = x.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(x / lib * float(target_sum)).mean(axis=0)


def _legacy_nonzero_log_cpm_matrix(counts: np.ndarray, target_sum: float) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_sum = arr.sum(axis=1)
    arr = arr[row_sum != 0, :]
    if arr.shape[0] == 0:
        return np.zeros((0, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)
    norm = arr.sum(axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return np.log1p(arr / norm * float(target_sum)).astype(np.float32, copy=False)


def _legacy_profile_metrics(counts_true: np.ndarray, counts_pred: np.ndarray, target_sum: float) -> Tuple[float, float]:
    true_norm = _legacy_nonzero_log_cpm_matrix(counts_true, target_sum)
    pred_norm = _legacy_nonzero_log_cpm_matrix(counts_pred, target_sum)
    if true_norm.shape[0] == 0 or pred_norm.shape[0] == 0:
        return np.nan, np.nan
    true_vec = true_norm.mean(axis=0)
    pred_vec = pred_norm.mean(axis=0)
    if np.std(true_vec) == 0:
        true_vec = true_vec.copy()
        true_vec[0] += 1e-6
    return _safe_r2(true_vec, pred_vec), _safe_corr(true_vec, pred_vec, "pearson")


def _mean_input(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return np.array([], dtype=np.float32)
    x = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return x.mean(axis=0)


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        if kind == "pearson":
            v, _ = pearsonr(x, y)
        elif kind == "spearman":
            from scipy.stats import spearmanr

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


def _direction_accuracy_match(true_logfc: np.ndarray, pred_logfc: np.ndarray) -> float:
    if true_logfc.size == 0 or pred_logfc.size == 0 or true_logfc.shape != pred_logfc.shape:
        return np.nan
    if np.isnan(true_logfc).any() or np.isnan(pred_logfc).any():
        return np.nan
    return float((np.sign(true_logfc) == np.sign(pred_logfc)).mean())


def _parse_key_with_cell_type_prefix(
    key: str,
    known_cell_types: Iterable[str],
) -> Tuple[Optional[str], Optional[str]]:
    sorted_ct = sorted([str(x) for x in known_cell_types], key=len, reverse=True)
    for ct in sorted_ct:
        prefix = f"{ct}_"
        if key.startswith(prefix):
            return ct, key[len(prefix) :]
    return None, None


def _build_deg_index_cache(
    adata_ref: ad.AnnData,
    degs_key: str,
    expected_topk: int,
) -> Dict[str, np.ndarray]:
    if degs_key not in adata_ref.uns:
        raise KeyError(f"{degs_key} not found in adata.uns")
    gene_to_idx = {str(g): i for i, g in enumerate(adata_ref.var_names.astype(str))}
    deg_cache: Dict[str, np.ndarray] = {}
    for k, genes in adata_ref.uns[degs_key].items():
        mapped = [gene_to_idx[str(g)] for g in genes if str(g) in gene_to_idx]
        arr = np.asarray(mapped, dtype=int)
        if expected_topk > 0 and arr.size > expected_topk:
            arr = arr[:expected_topk]
        deg_cache[str(k)] = arr
    return deg_cache


def _build_control_cache(
    adata_ref: ad.AnnData,
    cell_types: List[str],
    cell_type_col: str,
    condition_col: str,
    control_label: str,
    control_layer: Optional[str],
    split_key: Optional[str],
    control_tags: Optional[List[str]],
) -> Dict[str, np.ndarray]:
    obs = adata_ref.obs
    cache: Dict[str, np.ndarray] = {}
    split_mask = np.ones(obs.shape[0], dtype=bool)
    if split_key and split_key in obs.columns and control_tags:
        split_mask &= obs[split_key].astype(str).isin(control_tags).values
    for ct in cell_types:
        mask = (
            split_mask
            & (obs[cell_type_col].astype(str).values == str(ct))
            & (obs[condition_col].astype(str).values == str(control_label))
        )
        idx = np.where(mask)[0]
        cache[ct] = _extract_matrix(adata_ref, idx, control_layer)
    return cache


def _split_csv_tags(text: str) -> Optional[List[str]]:
    text = text.strip()
    if not text:
        return None
    vals = [x.strip() for x in text.split(",") if x.strip()]
    return vals or None


def _iter_h5ad_items(
    adata_pred: ad.AnnData,
    cell_type_col: str,
    condition_col: str,
    control_label: str,
    pred_layer: Optional[str],
    split_key: Optional[str],
    eval_tags: Optional[List[str]],
):
    obs = adata_pred.obs
    mask = np.ones(obs.shape[0], dtype=bool)
    if split_key and split_key in obs.columns and eval_tags:
        mask &= obs[split_key].astype(str).isin(eval_tags).values
    grouped = obs.loc[mask].groupby([cell_type_col, condition_col], observed=False).indices
    for (ct, cond), idx in grouped.items():
        if str(cond) == str(control_label):
            continue
        idx_arr = np.asarray(idx, dtype=int)
        y_pred = _extract_matrix(adata_pred, idx_arr, pred_layer)
        yield {
            "cell_type": str(ct),
            "condition": str(cond),
            "y_pred": y_pred,
            "source_key": f"{ct}_{cond}",
        }


def _iter_biolord_items(
    pred_dict: Dict[str, object],
    known_cell_types: List[str],
    control_label: str,
    pred_index: int,
):
    for raw_key, value in pred_dict.items():
        ct, cond = _parse_key_with_cell_type_prefix(str(raw_key), known_cell_types)
        if ct is None or cond is None or cond == str(control_label):
            continue
        if not isinstance(value, (list, tuple)) or len(value) <= pred_index:
            continue
        y_pred = _to_numpy_2d(value[pred_index])
        yield {
            "cell_type": ct,
            "condition": cond,
            "y_pred": y_pred,
            "source_key": str(raw_key),
        }


def _resolve_pred_layer(adata_pred: ad.AnnData, requested: str) -> Optional[str]:
    if requested and requested != "auto":
        return requested if requested in adata_pred.layers else None
    for key in ("predicted", "prediction"):
        if key in adata_pred.layers:
            return key
    return None


def _series_stat(df: pd.DataFrame, column: str, fn: str) -> float:
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return np.nan
    if fn == "mean":
        return float(series.mean())
    if fn == "median":
        return float(series.median())
    raise ValueError(f"Unsupported stat function: {fn}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NIPS direction benchmark for ours / biolord.")
    p.add_argument("--input_type", type=str, required=True, choices=["h5ad", "biolord_dict"])
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--reference_adata_path", type=str, default=DEFAULT_REFERENCE_ADATA_PATH)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--output_prefix", type=str, default="direction_benchmark")

    p.add_argument("--cell_type_col", type=str, default="cell_type")
    p.add_argument("--condition_col", type=str, default="condition")
    p.add_argument("--control_label", type=str, default="DMSO")
    p.add_argument("--degs_key", type=str, default="rank_genes_groups_cov")
    p.add_argument("--expected_topk", type=int, default=50)
    p.add_argument("--counts_target_sum", type=float, default=1_000_000.0)
    p.add_argument(
        "--reference_space",
        type=str,
        default="mappedx",
        choices=["mappedx", "x"],
        help="mappedx uses counts->log1p(CPM), x uses direct mean(adata.X).",
    )

    p.add_argument("--canonical_split_key", type=str, default="split_82_1_cpa")
    p.add_argument("--canonical_eval_tags", type=str, default="ood")
    p.add_argument("--canonical_control_tags", type=str, default="train")
    p.add_argument("--pred_layer", type=str, default="auto")
    p.add_argument("--biolord_pred_index", type=int, default=2)
    p.add_argument("--min_cells_true", type=int, default=5)
    p.add_argument("--min_valid_pred_cells", type=int, default=1)
    p.add_argument("--min_pred_coverage", type=float, default=0.0)
    p.add_argument("--max_conditions", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_tags = _split_csv_tags(args.canonical_eval_tags)
    control_tags = _split_csv_tags(args.canonical_control_tags)
    split_key = args.canonical_split_key.strip() or None

    ref_layer = "counts" if args.reference_space == "mappedx" else None
    adata_ref = ad.read_h5ad(args.reference_adata_path, backed="r")
    cell_types = list(map(str, adata_ref.obs[args.cell_type_col].unique()))
    deg_cache = _build_deg_index_cache(adata_ref, args.degs_key, args.expected_topk)
    control_cache = _build_control_cache(
        adata_ref=adata_ref,
        cell_types=cell_types,
        cell_type_col=args.cell_type_col,
        condition_col=args.condition_col,
        control_label=args.control_label,
        control_layer=ref_layer,
        split_key=split_key,
        control_tags=control_tags,
    )

    if args.input_type == "h5ad":
        adata_pred = ad.read_h5ad(args.input_path, backed="r")
        pred_layer = _resolve_pred_layer(adata_pred, args.pred_layer)
        items = _iter_h5ad_items(
            adata_pred=adata_pred,
            cell_type_col=args.cell_type_col,
            condition_col=args.condition_col,
            control_label=args.control_label,
            pred_layer=pred_layer,
            split_key=split_key,
            eval_tags=eval_tags,
        )
    else:
        adata_pred = None
        with open(args.input_path, "rb") as fh:
            pred_dict = pickle.load(fh)
        if not isinstance(pred_dict, dict):
            raise ValueError("biolord_dict input must be a dict.")
        items = _iter_biolord_items(
            pred_dict=pred_dict,
            known_cell_types=cell_types,
            control_label=args.control_label,
            pred_index=args.biolord_pred_index,
        )

    rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    skipped = {
        "missing_deg_key": 0,
        "empty_deg_idx": 0,
        "missing_control": 0,
        "small_true_cells": 0,
        "empty_pred_rows": 0,
        "small_valid_pred_cells": 0,
        "low_pred_coverage": 0,
        "dim_mismatch": 0,
    }
    n_done = 0

    for item in items:
        cell_type = str(item["cell_type"])
        condition = str(item["condition"])
        source_key = str(item["source_key"])
        y_pred = _to_numpy_2d(item["y_pred"])

        deg_key = f"{cell_type}_{condition}"
        n_pred_total = int(y_pred.shape[0])
        pred_valid_mask = np.isfinite(y_pred).all(axis=1) if n_pred_total > 0 else np.zeros((0,), dtype=bool)
        n_pred_valid = int(pred_valid_mask.sum())
        pred_coverage = float(n_pred_valid / n_pred_total) if n_pred_total > 0 else np.nan
        skip_reason = ""

        if deg_key not in deg_cache:
            skipped["missing_deg_key"] += 1
            skip_reason = "missing_deg_key"
        elif deg_cache[deg_key].size == 0:
            skipped["empty_deg_idx"] += 1
            skip_reason = "empty_deg_idx"
        else:
            ctrl_mat = control_cache.get(cell_type)
            if ctrl_mat is None or ctrl_mat.shape[0] == 0:
                skipped["missing_control"] += 1
                skip_reason = "missing_control"
            else:
                obs = adata_ref.obs
                mask = (
                    (obs[args.cell_type_col].astype(str).values == cell_type)
                    & (obs[args.condition_col].astype(str).values == condition)
                )
                if split_key and split_key in obs.columns and eval_tags:
                    mask &= obs[split_key].astype(str).isin(eval_tags).values
                true_idx = np.where(mask)[0]
                true_mat = _extract_matrix(adata_ref, true_idx, ref_layer)

                if true_mat.shape[0] < args.min_cells_true:
                    skipped["small_true_cells"] += 1
                    skip_reason = "small_true_cells"
                elif n_pred_total == 0:
                    skipped["empty_pred_rows"] += 1
                    skip_reason = "empty_pred_rows"
                elif n_pred_valid < args.min_valid_pred_cells:
                    skipped["small_valid_pred_cells"] += 1
                    skip_reason = "small_valid_pred_cells"
                elif pred_coverage < float(args.min_pred_coverage):
                    skipped["low_pred_coverage"] += 1
                    skip_reason = "low_pred_coverage"
                else:
                    y_pred_valid = y_pred[pred_valid_mask]
                    if args.reference_space == "mappedx":
                        true_vec = _counts_to_x_mean(true_mat, args.counts_target_sum)
                        ctrl_vec = _counts_to_x_mean(ctrl_mat, args.counts_target_sum)
                        r2_value, pear_value = _legacy_profile_metrics(
                            true_mat,
                            y_pred_valid,
                            args.counts_target_sum,
                        )
                    else:
                        true_vec = _mean_input(true_mat)
                        ctrl_vec = _mean_input(ctrl_mat)
                        r2_value, pear_value = _safe_r2(true_vec, _mean_input(y_pred_valid)), _safe_corr(
                            true_vec, _mean_input(y_pred_valid), "pearson"
                        )
                    if args.input_type == "h5ad" and args.reference_space == "mappedx":
                        pred_vec = _counts_to_x_mean(y_pred_valid, args.counts_target_sum)
                    else:
                        pred_vec = _mean_input(y_pred_valid)

                    if pred_vec.shape[0] != ctrl_vec.shape[0]:
                        skipped["dim_mismatch"] += 1
                        skip_reason = "dim_mismatch"
                    else:
                        de_idx = deg_cache[deg_key]
                        true_logfc = true_vec - ctrl_vec
                        pred_logfc = pred_vec - ctrl_vec

                        rows.append(
                            {
                                "source_key": source_key,
                                "pair_key": source_key,
                                "cell_type": cell_type,
                                "condition": condition,
                                "deg_key": deg_key,
                                "n_true_cells": int(true_mat.shape[0]),
                                "n_pred_cells_total": n_pred_total,
                                "n_pred_cells_valid": n_pred_valid,
                                "pred_coverage_ratio": pred_coverage,
                                "n_de_genes": int(de_idx.size),
                                "r2": r2_value,
                                "pear": pear_value,
                                "r2_deg": (
                                    _legacy_profile_metrics(
                                        true_mat[:, de_idx],
                                        y_pred_valid[:, de_idx],
                                        args.counts_target_sum,
                                    )[0]
                                    if args.reference_space == "mappedx"
                                    else _safe_r2(true_vec[de_idx], pred_vec[de_idx])
                                ),
                                "pear_deg": (
                                    _legacy_profile_metrics(
                                        true_mat[:, de_idx],
                                        y_pred_valid[:, de_idx],
                                        args.counts_target_sum,
                                    )[1]
                                    if args.reference_space == "mappedx"
                                    else _safe_corr(true_vec[de_idx], pred_vec[de_idx], "pearson")
                                ),
                                "direction_accuracy_top50": _direction_accuracy_match(
                                    true_logfc[de_idx], pred_logfc[de_idx]
                                ),
                                "pr_delta_de": _safe_corr(
                                    true_logfc[de_idx], pred_logfc[de_idx], "pearson"
                                ),
                                "sp_delta_de": _safe_corr(
                                    true_logfc[de_idx], pred_logfc[de_idx], "spearman"
                                ),
                                "direction_accuracy_all": _direction_accuracy_match(
                                    true_logfc, pred_logfc
                                ),
                                "pr_delta_all": _safe_corr(true_logfc, pred_logfc, "pearson"),
                                "sp_delta_all": _safe_corr(true_logfc, pred_logfc, "spearman"),
                            }
                        )
                        n_done += 1

        coverage_rows.append(
            {
                "source_key": source_key,
                "deg_key": deg_key,
                "cell_type": cell_type,
                "condition": condition,
                "n_pred_cells_total": n_pred_total,
                "n_pred_cells_valid": n_pred_valid,
                "pred_coverage_ratio": pred_coverage,
                "used_for_eval": skip_reason == "",
                "skip_reason": skip_reason,
            }
        )
        if args.max_conditions > 0 and n_done >= args.max_conditions:
            break

    df = pd.DataFrame(rows)
    cov_df = pd.DataFrame(coverage_rows)

    per_path = output_dir / f"{args.output_prefix}_pair_metrics.csv"
    agg_path = output_dir / f"{args.output_prefix}_aggregate_metrics.csv"
    cell_path = output_dir / f"{args.output_prefix}_cell_type_summary.csv"
    cov_path = output_dir / f"{args.output_prefix}_coverage.csv"
    run_cfg_path = output_dir / f"{args.output_prefix}_run_config.json"

    df.to_csv(per_path, index=False)
    cov_df.to_csv(cov_path, index=False)
    run_cfg_path.write_text(
        json.dumps(
            {
                **vars(args),
                "n_pairs_scored": int(df.shape[0]),
                "skipped": skipped,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if df.empty:
        agg_df = pd.DataFrame(
            [
                {
                    "n_pairs_total": int(cov_df["source_key"].nunique()),
                    "n_pairs_scored": 0,
                    "r2_mean": np.nan,
                    "r2_median": np.nan,
                    "pear_mean": np.nan,
                    "pear_median": np.nan,
                    "r2_deg_mean": np.nan,
                    "r2_deg_median": np.nan,
                    "pear_deg_mean": np.nan,
                    "pear_deg_median": np.nan,
                    "direction_accuracy_top50_mean": np.nan,
                    "direction_accuracy_top50_median": np.nan,
                    "pr_delta_de_mean": np.nan,
                    "pr_delta_de_median": np.nan,
                    "sp_delta_de_mean": np.nan,
                    "sp_delta_de_median": np.nan,
                    "direction_accuracy_all_mean": np.nan,
                    "direction_accuracy_all_median": np.nan,
                    "pr_delta_all_mean": np.nan,
                    "pr_delta_all_median": np.nan,
                    "sp_delta_all_mean": np.nan,
                    "sp_delta_all_median": np.nan,
                }
            ]
        )
        cell_df = pd.DataFrame(
            columns=[
                "cell_type",
                "n_pairs",
                "r2_mean",
                "r2_median",
                "pear_mean",
                "pear_median",
                "r2_deg_mean",
                "r2_deg_median",
                "pear_deg_mean",
                "pear_deg_median",
                "direction_accuracy_top50_mean",
                "direction_accuracy_top50_median",
                "pr_delta_de_mean",
                "pr_delta_de_median",
                "sp_delta_de_mean",
                "sp_delta_de_median",
                "direction_accuracy_all_mean",
                "direction_accuracy_all_median",
                "pr_delta_all_mean",
                "pr_delta_all_median",
                "sp_delta_all_mean",
                "sp_delta_all_median",
            ]
        )
    else:
        agg_df = pd.DataFrame(
            [
                {
                    "n_pairs_total": int(cov_df["source_key"].nunique()),
                    "n_pairs_scored": int(df.shape[0]),
                    "r2_mean": _series_stat(df, "r2", "mean"),
                    "r2_median": _series_stat(df, "r2", "median"),
                    "pear_mean": _series_stat(df, "pear", "mean"),
                    "pear_median": _series_stat(df, "pear", "median"),
                    "r2_deg_mean": _series_stat(df, "r2_deg", "mean"),
                    "r2_deg_median": _series_stat(df, "r2_deg", "median"),
                    "pear_deg_mean": _series_stat(df, "pear_deg", "mean"),
                    "pear_deg_median": _series_stat(df, "pear_deg", "median"),
                    "direction_accuracy_top50_mean": _series_stat(df, "direction_accuracy_top50", "mean"),
                    "direction_accuracy_top50_median": _series_stat(df, "direction_accuracy_top50", "median"),
                    "pr_delta_de_mean": _series_stat(df, "pr_delta_de", "mean"),
                    "pr_delta_de_median": _series_stat(df, "pr_delta_de", "median"),
                    "sp_delta_de_mean": _series_stat(df, "sp_delta_de", "mean"),
                    "sp_delta_de_median": _series_stat(df, "sp_delta_de", "median"),
                    "direction_accuracy_all_mean": _series_stat(df, "direction_accuracy_all", "mean"),
                    "direction_accuracy_all_median": _series_stat(df, "direction_accuracy_all", "median"),
                    "pr_delta_all_mean": _series_stat(df, "pr_delta_all", "mean"),
                    "pr_delta_all_median": _series_stat(df, "pr_delta_all", "median"),
                    "sp_delta_all_mean": _series_stat(df, "sp_delta_all", "mean"),
                    "sp_delta_all_median": _series_stat(df, "sp_delta_all", "median"),
                }
            ]
        )
        cell_rows: List[Dict[str, object]] = []
        for cell_type, group in df.groupby("cell_type", observed=False):
            cell_rows.append(
                {
                    "cell_type": cell_type,
                    "n_pairs": int(group.shape[0]),
                    "r2_mean": _series_stat(group, "r2", "mean"),
                    "r2_median": _series_stat(group, "r2", "median"),
                    "pear_mean": _series_stat(group, "pear", "mean"),
                    "pear_median": _series_stat(group, "pear", "median"),
                    "r2_deg_mean": _series_stat(group, "r2_deg", "mean"),
                    "r2_deg_median": _series_stat(group, "r2_deg", "median"),
                    "pear_deg_mean": _series_stat(group, "pear_deg", "mean"),
                    "pear_deg_median": _series_stat(group, "pear_deg", "median"),
                    "direction_accuracy_top50_mean": _series_stat(group, "direction_accuracy_top50", "mean"),
                    "direction_accuracy_top50_median": _series_stat(group, "direction_accuracy_top50", "median"),
                    "pr_delta_de_mean": _series_stat(group, "pr_delta_de", "mean"),
                    "pr_delta_de_median": _series_stat(group, "pr_delta_de", "median"),
                    "sp_delta_de_mean": _series_stat(group, "sp_delta_de", "mean"),
                    "sp_delta_de_median": _series_stat(group, "sp_delta_de", "median"),
                    "direction_accuracy_all_mean": _series_stat(group, "direction_accuracy_all", "mean"),
                    "direction_accuracy_all_median": _series_stat(group, "direction_accuracy_all", "median"),
                    "pr_delta_all_mean": _series_stat(group, "pr_delta_all", "mean"),
                    "pr_delta_all_median": _series_stat(group, "pr_delta_all", "median"),
                    "sp_delta_all_mean": _series_stat(group, "sp_delta_all", "mean"),
                    "sp_delta_all_median": _series_stat(group, "sp_delta_all", "median"),
                }
            )
        cell_df = pd.DataFrame(cell_rows)

    agg_df.to_csv(agg_path, index=False)
    cell_df.to_csv(cell_path, index=False)

    print("Saved:")
    print(f"  pair metrics:  {per_path}")
    print(f"  aggregate:     {agg_path}")
    print(f"  cell-type:     {cell_path}")
    print(f"  coverage:      {cov_path}")
    print("\nAggregate:")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
