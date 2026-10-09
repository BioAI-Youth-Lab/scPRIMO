#!/usr/bin/env python3
"""Train scVI decoders used by the PRIMO/GRCA release pipeline."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

from primo.config import _load_yaml
from primo.data.splits import KFoldSplitConfig, load_kfold_split
from primo.paths import CHECKPOINT_DIR, CONFIG_DIR, DATA_DIR, REPO_ROOT


@dataclass(frozen=True)
class DatasetDefaults:
    h5ad_path: str
    split_dir: str
    k: int
    test_size: float
    n_latent: int
    n_hidden: int
    n_layers: int
    epochs: int
    n_epochs_kl_warmup: int
    output_base: str
    good_smile_key: str = ""
    good_smile_value: Any = 1


DATASET_DEFAULTS: Dict[str, DatasetDefaults] = {
    "nips": DatasetDefaults(
        h5ad_path=str(DATA_DIR / "nips" / "nips_processed_min50_split82.h5ad"),
        split_dir=str(DATA_DIR / "nips" / "splits_82"),
        k=3,
        test_size=0.2,
        n_latent=20,
        n_hidden=256,
        n_layers=2,
        epochs=500,
        n_epochs_kl_warmup=300,
        output_base=str(CHECKPOINT_DIR / "nips" / "scvi"),
    ),
    "sciplex": DatasetDefaults(
        h5ad_path=str(DATA_DIR / "sciplex" / "sciplex_processed.h5ad"),
        split_dir=str(DATA_DIR / "sciplex" / "splits_82"),
        k=5,
        test_size=0.2,
        n_latent=10,
        n_hidden=128,
        n_layers=1,
        epochs=400,
        n_epochs_kl_warmup=400,
        output_base=str(CHECKPOINT_DIR / "sciplex" / "scvi"),
        good_smile_key="good_smile_chemvae",
        good_smile_value=1,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the scVI decoder checkpoint consumed by PRIMO/GRCA."
    )
    p.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional release config. Defaults to configs/{dataset}.yaml when present.",
    )
    p.add_argument("--adata_path", type=str, default="")
    p.add_argument("--split_dir", type=str, default="")
    p.add_argument("--split_col", type=str, default="")
    p.add_argument("--split_train_label", type=str, default="train")
    p.add_argument("--split_test_label", type=str, default="test")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--test_size", type=float, default=None)

    p.add_argument("--good_smile_key", type=str, default="")
    p.add_argument("--good_smile_value", type=str, default="")

    p.add_argument("--output_dir", type=str, default="")
    p.add_argument(
        "--output_dir_template",
        type=str,
        default="",
        help=(
            "Optional template with {dataset}, {fold}, and {test_size}. "
            "Ignored when --output_dir is set."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_anndata", action="store_true")

    p.add_argument("--layer", type=str, default="counts")
    p.add_argument(
        "--batch_key",
        type=str,
        default="",
        help=(
            "Optional scVI batch key. Leave empty for the default PRIMO pipeline; "
            "downstream GRCA loading must use the same setup if this is enabled."
        ),
    )
    p.add_argument("--n_hvg", type=int, default=0)
    p.add_argument("--hvg_flavor", type=str, default="cell_ranger")

    p.add_argument("--n_latent", type=int, default=None)
    p.add_argument("--n_hidden", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument(
        "--dispersion",
        type=str,
        default="gene",
        choices=["gene", "gene-batch", "gene-label", "gene-cell"],
    )
    p.add_argument(
        "--gene_likelihood",
        type=str,
        default="zinb",
        choices=["zinb", "nb", "poisson"],
    )
    p.add_argument(
        "--latent_distribution",
        type=str,
        default="normal",
        choices=["normal", "ln"],
    )

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train_size", type=float, default=0.9)
    p.add_argument("--inner_test_size", type=float, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_epochs_kl_warmup", type=int, default=None)
    p.add_argument("--frequency", type=int, default=20)
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--es_metric", type=str, default="elbo")
    p.add_argument(
        "--es_on",
        type=str,
        default="test_set",
        choices=["train_set", "test_set", "validation_set"],
    )
    p.add_argument("--es_patience", type=int, default=100)
    p.add_argument("--es_threshold", type=float, default=3.0)
    p.add_argument("--es_save_best_state", action="store_true")
    p.add_argument("--es_reduce_lr_on_plateau", action="store_true")
    p.add_argument("--es_lr_patience", type=int, default=10)
    p.add_argument("--es_lr_factor", type=float, default=0.5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def _as_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _default_config_path(dataset: str) -> Path:
    return CONFIG_DIR / f"{dataset}.yaml"


def _load_release_config(dataset: str, config_path: str) -> Dict[str, Any]:
    path = Path(config_path) if config_path else _default_config_path(dataset)
    if not path.exists():
        return {}
    return _load_yaml(path)


def _resolve_from_cli_config_default(
    cli_value: Any,
    config_section: Dict[str, Any],
    key: str,
    default_value: Any,
) -> Any:
    if cli_value not in ("", None):
        return cli_value
    if key in config_section and config_section[key] not in ("", None):
        return config_section[key]
    return default_value


def _format_output_path(template: str, dataset: str, fold: int, test_size: float) -> str:
    return template.format(dataset=dataset, fold=fold, test_size=test_size)


def resolve_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = DATASET_DEFAULTS[args.dataset]
    raw_config = _load_release_config(args.dataset, args.config)
    dataset_cfg = raw_config.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    paths_cfg = raw_config.get("paths", {})
    if not isinstance(paths_cfg, dict):
        paths_cfg = {}

    fold = int(args.fold)
    k = int(_resolve_from_cli_config_default(args.k, dataset_cfg, "k", defaults.k))
    test_size = float(
        _resolve_from_cli_config_default(
            args.test_size, dataset_cfg, "test_size", defaults.test_size
        )
    )
    h5ad_path = _as_repo_path(
        str(
            _resolve_from_cli_config_default(
                args.adata_path, dataset_cfg, "h5ad_path", defaults.h5ad_path
            )
        )
    )
    split_dir = _as_repo_path(
        str(
            _resolve_from_cli_config_default(
                args.split_dir, dataset_cfg, "split_dir", defaults.split_dir
            )
        )
    )
    good_smile_key = str(
        _resolve_from_cli_config_default(
            args.good_smile_key,
            dataset_cfg,
            "good_smile_key",
            defaults.good_smile_key,
        )
    )
    good_smile_value = _resolve_from_cli_config_default(
        args.good_smile_value,
        dataset_cfg,
        "good_smile_value",
        defaults.good_smile_value,
    )

    if args.output_dir:
        output_dir = _as_repo_path(args.output_dir)
    elif args.output_dir_template:
        output_dir = _as_repo_path(
            _format_output_path(args.output_dir_template, args.dataset, fold, test_size)
        )
    else:
        base_or_template = str(paths_cfg.get("scvi_model_dir", defaults.output_base))
        if "{" in base_or_template:
            output_dir = _as_repo_path(
                _format_output_path(base_or_template, args.dataset, fold, test_size)
            )
        else:
            output_dir = _as_repo_path(base_or_template) / f"fold{fold}_{test_size}"

    return {
        "raw_config_path": str(Path(args.config) if args.config else _default_config_path(args.dataset)),
        "h5ad_path": str(h5ad_path),
        "split_dir": str(split_dir),
        "fold": fold,
        "k": k,
        "test_size": test_size,
        "good_smile_key": good_smile_key,
        "good_smile_value": good_smile_value,
        "output_dir": str(output_dir),
        "n_latent": int(args.n_latent or defaults.n_latent),
        "n_hidden": int(args.n_hidden or defaults.n_hidden),
        "n_layers": int(args.n_layers or defaults.n_layers),
        "epochs": int(args.epochs or defaults.epochs),
        "n_epochs_kl_warmup": int(
            args.n_epochs_kl_warmup or defaults.n_epochs_kl_warmup
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(scvi, "settings"):
        scvi.settings.seed = int(seed)


def _load_split_from_obs(
    adata_obj: ad.AnnData,
    split_col: str,
    train_label: str,
    test_label: str,
) -> Dict[str, np.ndarray]:
    if split_col not in adata_obj.obs.columns:
        raise KeyError(f"split_col '{split_col}' not found in adata.obs")
    values = adata_obj.obs[split_col].astype(str).to_numpy()
    train_idx = np.where(values == str(train_label))[0].astype(np.int64)
    test_idx = np.where(values == str(test_label))[0].astype(np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Empty split from {split_col}: train={train_idx.size}, test={test_idx.size}"
        )
    return {"train_idx": train_idx, "test_idx": test_idx}


def _filter_good_smiles(
    adata_obj: ad.AnnData,
    good_smile_key: str,
    good_smile_value: Any,
) -> ad.AnnData:
    if not good_smile_key:
        return adata_obj
    if good_smile_key not in adata_obj.obs.columns:
        raise KeyError(f"good_smile_key '{good_smile_key}' not found in adata.obs")

    series = adata_obj.obs[good_smile_key]
    mask = np.asarray(series == good_smile_value)
    if not mask.any():
        mask = np.asarray(series.astype(str) == str(good_smile_value))
    if not mask.any():
        raise ValueError(
            f"Filtering by {good_smile_key}={good_smile_value!r} removed all cells."
        )
    return adata_obj[mask, :].copy()


def load_train_test(args: argparse.Namespace, runtime: Dict[str, Any]) -> tuple[ad.AnnData, ad.AnnData, str]:
    print(f"[INFO] loading adata: {runtime['h5ad_path']}")
    adata_obj = ad.read_h5ad(runtime["h5ad_path"])

    if args.dataset == "sciplex":
        before = adata_obj.n_obs
        adata_obj = _filter_good_smiles(
            adata_obj,
            str(runtime["good_smile_key"]),
            runtime["good_smile_value"],
        )
        print(f"[INFO] sciplex good-smile filter: {before} -> {adata_obj.n_obs} cells")

    if args.split_col:
        split = _load_split_from_obs(
            adata_obj,
            args.split_col,
            args.split_train_label,
            args.split_test_label,
        )
        split_source = f"obs:{args.split_col}"
    else:
        split = load_kfold_split(
            KFoldSplitConfig(
                fold=int(runtime["fold"]),
                k=int(runtime["k"]),
                split_dir=str(runtime["split_dir"]),
                test_size=float(runtime["test_size"]),
            )
        )
        split_source = (
            f"pickle:{runtime['split_dir']}/"
            f"kfold_{runtime['k']}_split_{runtime['fold']}_testsize_{runtime['test_size']}.pkl"
        )

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    adata_train = adata_obj[train_idx, :].copy()
    adata_test = adata_obj[test_idx, :].copy()
    print(f"[INFO] train={adata_train.shape}, test={adata_test.shape}")
    return adata_train, adata_test, split_source


def subset_by_hvg(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    n_hvg: int,
    layer: str,
    flavor: str,
) -> tuple[ad.AnnData, ad.AnnData, int]:
    if n_hvg <= 0 or n_hvg >= adata_train.n_vars:
        return adata_train.copy(), adata_test.copy(), int(adata_train.n_vars)

    adata_work = adata_train.copy()
    sc.pp.highly_variable_genes(
        adata_work,
        n_top_genes=int(n_hvg),
        layer=layer,
        flavor=flavor,
        subset=False,
    )
    if "highly_variable" not in adata_work.var.columns:
        raise RuntimeError("scanpy did not produce a highly_variable mask.")
    mask = np.asarray(adata_work.var["highly_variable"])
    if int(mask.sum()) <= 0:
        raise RuntimeError("HVG mask is empty.")
    return adata_train[:, mask].copy(), adata_test[:, mask].copy(), int(mask.sum())


def setup_scvi_anndata(adata_obj: ad.AnnData, layer: str, batch_key: str) -> None:
    if layer and layer not in adata_obj.layers:
        raise KeyError(f"layer '{layer}' not found in adata.layers")
    if batch_key:
        if batch_key not in adata_obj.obs.columns:
            raise KeyError(f"batch_key '{batch_key}' not found in adata.obs")
        if int(adata_obj.obs[batch_key].nunique()) <= 1:
            raise ValueError(f"batch_key '{batch_key}' has <=1 category in train data")
        scvi.data.setup_anndata(adata_obj, layer=layer, batch_key=batch_key)
    else:
        scvi.data.setup_anndata(adata_obj, layer=layer)


def resolve_use_cuda(device: str) -> bool:
    if device == "cpu":
        return False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    return bool(torch.cuda.is_available())


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime_config(args)
    set_seed(args.seed)
    use_cuda = resolve_use_cuda(args.device)

    print(f"[INFO] scvi-tools version={scvi.__version__}")
    print(f"[INFO] use_cuda={use_cuda}")
    print(f"[INFO] output_dir={runtime['output_dir']}")

    output_dir = Path(runtime["output_dir"])
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite to retrain.")
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_train_base, adata_test_base, split_source = load_train_test(args, runtime)
    adata_train, adata_test, n_vars_used = subset_by_hvg(
        adata_train_base,
        adata_test_base,
        n_hvg=int(args.n_hvg),
        layer=args.layer,
        flavor=args.hvg_flavor,
    )
    print(
        "[INFO] training scVI: "
        f"n_vars={n_vars_used}, n_latent={runtime['n_latent']}, "
        f"n_hidden={runtime['n_hidden']}, n_layers={runtime['n_layers']}"
    )

    setup_scvi_anndata(adata_train, layer=args.layer, batch_key=args.batch_key)

    model = scvi.model.SCVI(
        adata_train,
        n_hidden=int(runtime["n_hidden"]),
        n_latent=int(runtime["n_latent"]),
        n_layers=int(runtime["n_layers"]),
        dropout_rate=float(args.dropout_rate),
        dispersion=args.dispersion,
        gene_likelihood=args.gene_likelihood,
        latent_distribution=args.latent_distribution,
        use_cuda=use_cuda,
    )

    trainer_kwargs: Dict[str, Any] = {}
    if args.early_stopping:
        trainer_kwargs["early_stopping_kwargs"] = {
            "early_stopping_metric": args.es_metric,
            "save_best_state_metric": args.es_metric if args.es_save_best_state else None,
            "on": args.es_on,
            "patience": int(args.es_patience),
            "threshold": float(args.es_threshold),
            "reduce_lr_on_plateau": bool(args.es_reduce_lr_on_plateau),
            "lr_patience": int(args.es_lr_patience),
            "lr_factor": float(args.es_lr_factor),
        }
        print(f"[INFO] early_stopping_kwargs={trainer_kwargs['early_stopping_kwargs']}")

    train_kwargs: Dict[str, Any] = {
        "n_epochs": int(runtime["epochs"]),
        "train_size": float(args.train_size),
        "lr": float(args.lr),
        "n_epochs_kl_warmup": int(runtime["n_epochs_kl_warmup"]),
        "frequency": int(args.frequency),
    }
    if args.inner_test_size is not None:
        train_kwargs["test_size"] = float(args.inner_test_size)
    train_kwargs.update(trainer_kwargs)

    t0 = time.time()
    model.train(**train_kwargs)
    train_seconds = float(time.time() - t0)

    model.save(str(output_dir), overwrite=True, save_anndata=bool(args.save_anndata))
    pd.Series(adata_train.var_names.astype(str)).to_csv(
        output_dir / "var_names.csv",
        index=False,
        header=False,
    )

    run_config = {
        "dataset": args.dataset,
        "runtime": runtime,
        "split_source": split_source,
        "layer": args.layer,
        "batch_key": args.batch_key,
        "n_hvg": int(args.n_hvg),
        "hvg_flavor": args.hvg_flavor,
        "dropout_rate": float(args.dropout_rate),
        "dispersion": args.dispersion,
        "gene_likelihood": args.gene_likelihood,
        "latent_distribution": args.latent_distribution,
        "train_size": float(args.train_size),
        "inner_test_size": args.inner_test_size,
        "lr": float(args.lr),
        "frequency": int(args.frequency),
        "early_stopping": bool(args.early_stopping),
        "seed": int(args.seed),
        "use_cuda": bool(use_cuda),
        "train_seconds": train_seconds,
        "n_train_cells": int(adata_train.n_obs),
        "n_test_cells": int(adata_test.n_obs),
        "n_vars_used": int(n_vars_used),
    }
    (output_dir / "train_scvi_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] scVI model saved to: {output_dir}")


if __name__ == "__main__":
    main()
