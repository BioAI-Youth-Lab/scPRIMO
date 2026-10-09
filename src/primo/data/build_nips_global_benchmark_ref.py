#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from primo.paths import DATA_DIR, OUTPUT_DIR


def _write_gene_list(genes: np.ndarray, txt_path: Path, npy_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(txt_path, genes, fmt="%s")
    np.save(npy_path, genes)


def _gene_hash(genes: np.ndarray) -> str:
    joined = "\n".join(genes.tolist()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _to_list(x) -> list[str]:
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(v) for v in list(x)]


def _compute_deg_stats(deg_map: dict, top_n: int) -> dict[str, float]:
    lengths = [len(v) for v in deg_map.values()]
    if not lengths:
        return {
            "groups_total": 0.0,
            "groups_non_empty": 0.0,
            "groups_full_topn": 0.0,
            "deg_len_min": 0.0,
            "deg_len_median": 0.0,
            "deg_len_max": 0.0,
        }
    return {
        "groups_total": float(len(lengths)),
        "groups_non_empty": float(sum(1 for x in lengths if x > 0)),
        "groups_full_topn": float(sum(1 for x in lengths if x >= top_n)),
        "deg_len_min": float(min(lengths)),
        "deg_len_median": float(np.median(lengths)),
        "deg_len_max": float(max(lengths)),
    }


def _rank_genes_groups_by_cov_hvg(
    adata: ad.AnnData,
    groupby: str,
    covariate: str,
    control_group: str,
    n_genes: int,
    rankby_abs: bool,
) -> tuple[dict, dict]:
    gene_dict: dict[str, np.ndarray] = {}
    logfc_dict: dict[str, np.ndarray] = {}
    cov_categories = adata.obs[covariate].astype(str).unique().tolist()

    for cov_cat in cov_categories:
        control_group_cov = "_".join([cov_cat, control_group])
        adata_cov = adata[adata.obs[covariate].astype(str) == cov_cat, :].copy()
        groups_here = set(adata_cov.obs[groupby].astype(str).tolist())
        if control_group_cov not in groups_here:
            raise ValueError(
                f"Control group '{control_group_cov}' not found in groupby='{groupby}' "
                f"for covariate '{covariate}={cov_cat}'."
            )

        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        logfc_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["logfoldchanges"])
        for group in de_genes.columns:
            gene_dict[group] = np.asarray(de_genes[group].tolist(), dtype=object)
            logfc_dict[group] = np.asarray(logfc_genes[group].tolist(), dtype=float)

    return gene_dict, logfc_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a global benchmark NIPS reference: full-data HVG5000 + full-data DEG1000."
    )
    p.add_argument(
        "--adata_path",
        type=Path,
        default=DATA_DIR / "nips" / "nips_processed_min50_split82.h5ad",
    )
    p.add_argument("--n_hvg", type=int, default=5000)
    p.add_argument("--layer", type=str, default="counts")
    p.add_argument("--flavor", type=str, default="cell_ranger")
    p.add_argument(
        "--hvg_selection_mode",
        type=str,
        default="mask_order",
        choices=["mask_order", "dispersion_order"],
    )
    p.add_argument("--deg_top_n", type=int, default=1000)
    p.add_argument("--deg_groupby", type=str, default="cov_drug_name")
    p.add_argument("--deg_covariate", type=str, default="cell_type")
    p.add_argument("--deg_control_group", type=str, default="DMSO")
    p.add_argument(
        "--no_deg_rankby_abs",
        action="store_true",
        help="Disable rankby_abs when recomputing HVG-space DEGs.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_DIR / "nips" / "global_benchmark_hvg5000_deg1000",
    )
    p.add_argument(
        "--output_h5ad",
        type=Path,
        default=OUTPUT_DIR / "nips" / "nips_ref_global_hvg5000_globaldeg1000.h5ad",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.adata_path)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_hvg,
        layer=args.layer,
        flavor=args.flavor,
        subset=False,
    )

    if args.hvg_selection_mode == "mask_order":
        hv_mask = adata.var["highly_variable"].to_numpy()
        genes = adata.var_names[hv_mask][: args.n_hvg]
    else:
        ranked = adata.var["dispersions_norm"].fillna(-np.inf)
        genes = ranked.sort_values(ascending=False, kind="mergesort").index[: args.n_hvg]

    genes = np.asarray(genes, dtype=str)
    if genes.shape[0] != args.n_hvg:
        raise ValueError(f"Expected {args.n_hvg} HVGs, got {genes.shape[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    genes_txt = args.output_dir / f"global_hvg{args.n_hvg}.txt"
    genes_npy = args.output_dir / f"global_hvg{args.n_hvg}.npy"
    _write_gene_list(genes, genes_txt, genes_npy)

    adata_ref = adata[:, genes].copy()
    deg_hvg, logfc_hvg = _rank_genes_groups_by_cov_hvg(
        adata=adata_ref,
        groupby=args.deg_groupby,
        covariate=args.deg_covariate,
        control_group=args.deg_control_group,
        n_genes=args.deg_top_n,
        rankby_abs=(not args.no_deg_rankby_abs),
    )
    deg_stats = _compute_deg_stats(deg_hvg, top_n=args.deg_top_n)

    adata_ref.uns["rank_genes_groups_cov"] = deg_hvg
    adata_ref.uns["rank_genes_groups_cov_logfc"] = logfc_hvg
    adata_ref.write_h5ad(args.output_h5ad)

    summary = {
        "adata_path": str(args.adata_path),
        "output_h5ad": str(args.output_h5ad),
        "n_obs_total": int(adata.n_obs),
        "n_vars_total": int(adata.n_vars),
        "n_hvg": int(args.n_hvg),
        "hvg_hash_sha256": _gene_hash(genes),
        "genes_txt": str(genes_txt),
        "genes_npy": str(genes_npy),
        "layer": args.layer,
        "flavor": args.flavor,
        "hvg_selection_mode": args.hvg_selection_mode,
        "deg_top_n": int(args.deg_top_n),
        "deg_groupby": args.deg_groupby,
        "deg_covariate": args.deg_covariate,
        "deg_control_group": args.deg_control_group,
        "deg_groups_total": int(deg_stats["groups_total"]),
        "deg_groups_non_empty": int(deg_stats["groups_non_empty"]),
        "deg_groups_full_topn": int(deg_stats["groups_full_topn"]),
        "deg_len_min": int(deg_stats["deg_len_min"]),
        "deg_len_median": float(deg_stats["deg_len_median"]),
        "deg_len_max": int(deg_stats["deg_len_max"]),
    }
    summary_path = args.output_dir / "global_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] global benchmark ref: {args.output_h5ad}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
