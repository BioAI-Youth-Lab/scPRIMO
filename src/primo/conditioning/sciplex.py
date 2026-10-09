from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from primo.conditioning.enhanced_covariates import build_train_enhanced_cov, fit_train_stats_pca
from primo.paths import ARTIFACT_DIR


DEFAULT_SCIPLEX_CELL_CONTEXT_MODE = "control_pca"
DEFAULT_SCIPLEX_CELL_CONTEXT_DIM = 16
DEFAULT_SCIPLEX_STATS_PCA_DIM = 64
DEFAULT_SCIPLEX_CONTROL_COL = "product_name_clean"
DEFAULT_SCIPLEX_CONTROL_LABEL = "Vehicle"
DEFAULT_SCIPLEX_RESIDUAL_SHRINKAGE = 10.0
DEFAULT_SCIPLEX_CACHE_ROOT = str(ARTIFACT_DIR / "sciplex")


def _slug_text(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "", str(value).strip().lower())
    return text or "na"


def _format_tag_number(value: float | int) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_sciplex_cache_tag(
    *,
    cell_context_mode: str = DEFAULT_SCIPLEX_CELL_CONTEXT_MODE,
    cell_context_dim: int = DEFAULT_SCIPLEX_CELL_CONTEXT_DIM,
    stats_pca_dim: int = DEFAULT_SCIPLEX_STATS_PCA_DIM,
    control_label: str = DEFAULT_SCIPLEX_CONTROL_LABEL,
    residualize_to_control: bool = True,
    residual_shrinkage: float = DEFAULT_SCIPLEX_RESIDUAL_SHRINKAGE,
    perturbation_key: str = "perturb_string_processed",
) -> str:
    mode = str(cell_context_mode).strip().lower()
    if mode == "control_pca":
        context_tag = f"controlpca{int(cell_context_dim)}"
    else:
        context_tag = _slug_text(mode)

    if residualize_to_control:
        stats_tag = f"residualshrink{_format_tag_number(residual_shrinkage)}"
    else:
        stats_tag = "absolute"

    return (
        f"{context_tag}_{_slug_text(control_label)}_"
        f"{stats_tag}_pca{int(stats_pca_dim)}_{_slug_text(perturbation_key)}"
    )


def default_sciplex_cache_dir(
    *,
    test_size: float,
    cell_context_mode: str = DEFAULT_SCIPLEX_CELL_CONTEXT_MODE,
    cell_context_dim: int = DEFAULT_SCIPLEX_CELL_CONTEXT_DIM,
    stats_pca_dim: int = DEFAULT_SCIPLEX_STATS_PCA_DIM,
    control_label: str = DEFAULT_SCIPLEX_CONTROL_LABEL,
    residualize_to_control: bool = True,
    residual_shrinkage: float = DEFAULT_SCIPLEX_RESIDUAL_SHRINKAGE,
    perturbation_key: str = "perturb_string_processed",
    root_dir: str = DEFAULT_SCIPLEX_CACHE_ROOT,
) -> str:
    tag = build_sciplex_cache_tag(
        cell_context_mode=cell_context_mode,
        cell_context_dim=cell_context_dim,
        stats_pca_dim=stats_pca_dim,
        control_label=control_label,
        residualize_to_control=residualize_to_control,
        residual_shrinkage=residual_shrinkage,
        perturbation_key=perturbation_key,
    )
    return str(Path(root_dir) / f"cached_data_{test_size}_{tag}")


def requested_sciplex_covariate_config(
    *,
    cell_context_mode: str = DEFAULT_SCIPLEX_CELL_CONTEXT_MODE,
    cell_context_dim: int = DEFAULT_SCIPLEX_CELL_CONTEXT_DIM,
    stats_pca_dim: int = DEFAULT_SCIPLEX_STATS_PCA_DIM,
    control_col: str = DEFAULT_SCIPLEX_CONTROL_COL,
    control_label: str = DEFAULT_SCIPLEX_CONTROL_LABEL,
    residualize_to_control: bool = True,
    residual_shrinkage: float = DEFAULT_SCIPLEX_RESIDUAL_SHRINKAGE,
) -> dict[str, Any]:
    return {
        "n_components": int(stats_pca_dim),
        "cell_type_col": "cell_type",
        "dose_col": "dose",
        "gene_layer": "counts",
        "random_state": 42,
        "cell_type_rep": str(cell_context_mode),
        "cell_type_pca_n": int(cell_context_dim),
        "control_col": str(control_col),
        "control_label": str(control_label),
        "residualize_to_control": bool(residualize_to_control),
        "residual_shrinkage": float(residual_shrinkage),
    }


def summarize_train_fit(train_fit: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(train_fit, dict):
        return {}
    categories = train_fit.get("cell_type_categories")
    return {
        "cell_context_mode": train_fit.get("cell_context_mode", "onehot"),
        "cell_context_dim": train_fit.get(
            "cell_context_dim",
            len(categories) if isinstance(categories, (list, tuple)) else None,
        ),
        "stats_space": train_fit.get("stats_space", "absolute"),
        "n_components": train_fit.get("n_components"),
        "control_col": train_fit.get("control_col"),
        "control_label": train_fit.get("control_label"),
        "residualize_to_control": bool(train_fit.get("residualize_to_control", False)),
        "residual_shrinkage": train_fit.get("residual_shrinkage"),
    }


def infer_sciplex_covariate_layout(
    enhanced_cov_train: np.ndarray,
    train_fit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cov_dim = int(enhanced_cov_train.shape[1])
    stats_pca_dim = (
        int(train_fit.get("n_components", DEFAULT_SCIPLEX_STATS_PCA_DIM))
        if isinstance(train_fit, dict)
        else DEFAULT_SCIPLEX_STATS_PCA_DIM
    )

    if isinstance(train_fit, dict) and train_fit.get("cell_context_dim") is not None:
        cell_context_dim = int(train_fit["cell_context_dim"])
    else:
        inferred = cov_dim - 1 - stats_pca_dim
        cell_context_dim = inferred if inferred > 0 else max(1, cov_dim - 1)

    return {
        "cov_dim": cov_dim,
        "stats_pca_dim": stats_pca_dim,
        "cell_context_dim": cell_context_dim,
        "cell_context_mode": (
            train_fit.get("cell_context_mode", "onehot")
            if isinstance(train_fit, dict)
            else "onehot"
        ),
        "stats_space": (
            train_fit.get("stats_space", "absolute")
            if isinstance(train_fit, dict)
            else "absolute"
        ),
    }


def _cache_key(test_size: float, fold: int) -> str:
    return f"{test_size}_{fold}"


def cache_paths(cache_dir: str | Path, fold: int, test_size: float) -> dict[str, Path]:
    base = Path(cache_dir)
    key = _cache_key(test_size=test_size, fold=fold)
    return {
        "cache_dir": base,
        "enhanced_cov_train": base / f"enhanced_cov_train_{key}.npy",
        "train_fit": base / f"train_fit_{key}.pkl",
        "trt_seenpert_list": base / f"trt_seenpert_list_{key}.pkl",
    }


def load_cached_train_artifacts(
    *,
    cache_dir: str | Path,
    fold: int,
    test_size: float,
) -> dict[str, Any] | None:
    paths = cache_paths(cache_dir=cache_dir, fold=fold, test_size=test_size)
    if not paths["enhanced_cov_train"].exists() or not paths["train_fit"].exists():
        return None

    with paths["train_fit"].open("rb") as f:
        train_fit = pickle.load(f)

    trt_seenpert_list = None
    if paths["trt_seenpert_list"].exists():
        with paths["trt_seenpert_list"].open("rb") as f:
            trt_seenpert_list = pickle.load(f)

    return {
        "cache_dir": str(paths["cache_dir"]),
        "enhanced_cov_train": np.load(paths["enhanced_cov_train"]),
        "train_fit": train_fit,
        "trt_seenpert_list": trt_seenpert_list,
    }


def save_cached_train_artifacts(
    *,
    cache_dir: str | Path,
    fold: int,
    test_size: float,
    enhanced_cov_train: np.ndarray,
    train_fit: dict[str, Any],
    trt_seenpert_list: Any = None,
) -> dict[str, Path]:
    paths = cache_paths(cache_dir=cache_dir, fold=fold, test_size=test_size)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    np.save(paths["enhanced_cov_train"], enhanced_cov_train)
    with paths["train_fit"].open("wb") as f:
        pickle.dump(train_fit, f)
    if trt_seenpert_list is not None:
        with paths["trt_seenpert_list"].open("wb") as f:
            pickle.dump(trt_seenpert_list, f)
    return paths


def train_fit_matches_requested(
    *,
    train_fit: dict[str, Any],
    requested_config: dict[str, Any],
) -> tuple[bool, str]:
    summary = summarize_train_fit(train_fit)
    requested_mode = str(requested_config["cell_type_rep"])
    requested_dim = int(requested_config["cell_type_pca_n"])
    requested_pca_dim = int(requested_config["n_components"])
    requested_residual = bool(requested_config["residualize_to_control"])
    requested_stats_space = "control_residual" if requested_residual else "absolute"

    checks = [
        (summary.get("cell_context_mode", "onehot") == requested_mode, "cell_context_mode"),
        (int(summary.get("n_components") or requested_pca_dim) == requested_pca_dim, "n_components"),
        (summary.get("stats_space", "absolute") == requested_stats_space, "stats_space"),
        (
            bool(summary.get("residualize_to_control", False)) == requested_residual,
            "residualize_to_control",
        ),
    ]

    if requested_mode == "control_pca":
        checks.extend(
            [
                (int(summary.get("cell_context_dim") or 0) == requested_dim, "cell_context_dim"),
                (summary.get("control_col") == requested_config["control_col"], "control_col"),
                (str(summary.get("control_label")) == str(requested_config["control_label"]), "control_label"),
            ]
        )

    bad = [name for ok, name in checks if not ok]
    if bad:
        return False, ", ".join(bad)
    return True, "ok"


def build_or_load_sciplex_train_artifacts(
    *,
    adata_train,
    adata_control_ref,
    cond_stage_data,
    fold: int,
    test_size: float,
    cache_dir: str | Path | None,
    requested_config: dict[str, Any],
    force_rebuild: bool = False,
) -> dict[str, Any]:
    cache_dir_str = str(cache_dir) if cache_dir else ""

    if cache_dir_str and not force_rebuild:
        cached = load_cached_train_artifacts(
            cache_dir=cache_dir_str,
            fold=fold,
            test_size=test_size,
        )
        if cached is not None:
            matches, reason = train_fit_matches_requested(
                train_fit=cached["train_fit"],
                requested_config=requested_config,
            )
            if matches:
                cached["cache_action"] = "loaded"
                cached["cache_mismatch_reason"] = ""
                return cached
            cached["cache_mismatch_reason"] = reason

    fit_kwargs = dict(requested_config)
    fit_kwargs["control_adata"] = adata_control_ref
    train_fit = fit_train_stats_pca(
        adata_train=adata_train,
        cond_stage_data_train=cond_stage_data,
        **fit_kwargs,
    )
    enhanced_cov_train = build_train_enhanced_cov(
        adata_train=adata_train,
        cond_stage_data_train=cond_stage_data,
        train_fit=train_fit,
        cell_type_col=requested_config["cell_type_col"],
        dose_col=requested_config["dose_col"],
    )

    result = {
        "cache_dir": cache_dir_str,
        "enhanced_cov_train": enhanced_cov_train,
        "train_fit": train_fit,
        "trt_seenpert_list": cond_stage_data,
        "cache_action": "built",
        "cache_mismatch_reason": "",
    }
    if cache_dir_str:
        save_cached_train_artifacts(
            cache_dir=cache_dir_str,
            fold=fold,
            test_size=test_size,
            enhanced_cov_train=enhanced_cov_train,
            train_fit=train_fit,
            trt_seenpert_list=cond_stage_data,
        )
    return result


def resolve_diag_adata_path(
    *,
    adata_path_arg: str,
    default_train_adata_path: str,
) -> str:
    adata_path_arg = str(adata_path_arg or "").strip()
    if adata_path_arg:
        return adata_path_arg
    return default_train_adata_path


def legacy_sciplex_cache_dir() -> str:
    return str(Path(DEFAULT_SCIPLEX_CACHE_ROOT) / "cached_data")


def maybe_resolve_existing_cache_dir(
    *,
    cache_dir_arg: str,
    fold: int,
    test_size: float,
) -> str:
    if cache_dir_arg:
        return cache_dir_arg

    versioned = default_sciplex_cache_dir(test_size=test_size)
    if os.path.exists(os.path.join(versioned, f"train_fit_{test_size}_{fold}.pkl")):
        return versioned

    legacy = legacy_sciplex_cache_dir()
    legacy_prefixed = os.path.join(
        legacy,
        f"train_fit_control_pca16_vehicle_residual10_{test_size}_{fold}.pkl",
    )
    if os.path.exists(legacy_prefixed):
        return legacy

    return versioned
