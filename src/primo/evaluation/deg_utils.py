"""
DEG (Differentially Expressed Gene) identification and utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Dict, List, Optional, Tuple, Literal


def get_deg_indices_sciplex(
    adata: AnnData,
    perturbation: str,
    n_genes: int = 50,
) -> np.ndarray:
    """
    Get DEG indices for sciplex dataset.
    
    Sciplex uses rank_genes_groups with treatment (trt) as key.
    Treatment is like 'S1001', 'S1002', which maps to SMILES.
    
    Args:
        adata: AnnData with pre-computed rank_genes_groups
        perturbation: Treatment key (e.g., 'S1001')
        n_genes: Number of top DEGs
    
    Returns:
        Array of gene indices
    """
    if "rank_genes_groups" not in adata.uns:
        raise ValueError("No rank_genes_groups found in adata.uns")
    
    if perturbation not in adata.uns["rank_genes_groups"]["names"].dtype.names:
        raise KeyError(f"Treatment '{perturbation}' not found in rank_genes_groups")
    
    deg_gene_names = adata.uns["rank_genes_groups"]["names"][perturbation][:n_genes]
    
    if "gene_idx" in adata.var.columns:
        deg_indices = np.array(adata.var.loc[deg_gene_names]["gene_idx"])
        print(f"Deg indices: {deg_indices}")
    else:
        deg_indices = np.array([
            np.where(adata.var_names == gene)[0][0]
            for gene in deg_gene_names
            if gene in adata.var_names
        ])
    
    return deg_indices


def get_deg_indices_nips(
    adata: AnnData,
    cell_type: str,
    perturbation: str,
    n_genes: int = 50,
) -> np.ndarray:
    """
    Get DEG indices for nips dataset.
    
    Nips uses rank_genes_groups_cov with key format: "{cell_type}_{perturbation}".
    
    Args:
        adata: AnnData with pre-computed rank_genes_groups_cov
        cell_type: Cell type name
        perturbation: Perturbation name
        n_genes: Number of top DEGs
    
    Returns:
        Array of gene indices
    """
    key_added = "rank_genes_groups_cov"
    if key_added not in adata.uns:
        raise ValueError(f"No {key_added} found in adata.uns")
    
    deg_key = f"{cell_type}_{perturbation}"
    if deg_key not in adata.uns[key_added]:
        # Try simplified drug name (remove non-alphanumeric)
        import re
        simple_pert = re.sub(r'[^a-zA-Z0-9]', '', perturbation)
        
        # Iterate keys to find match
        found_key = None
        for k in adata.uns[key_added].keys():
            if k.startswith(f"{cell_type}_"):
                k_pert = k.split("_", 1)[1]
                simple_k_pert = re.sub(r'[^a-zA-Z0-9]', '', k_pert)
                if simple_k_pert == simple_pert:
                    found_key = k
                    break
        
        if found_key:
            deg_key = found_key
        else:
            raise KeyError(f"Key '{deg_key}' not found in {key_added}")
    
    deg_gene_names = adata.uns[key_added][deg_key][:n_genes]
    
    if "gene_idx" in adata.var.columns:
        deg_indices = np.array(adata.var.loc[deg_gene_names]["gene_idx"])
    else:
        deg_indices = np.array([
            np.where(adata.var_names == gene)[0][0]
            for gene in deg_gene_names
            if gene in adata.var_names
        ])
    
    return deg_indices


def build_smile_to_trt_mapping(adata: AnnData) -> Dict[str, str]:
    """
    Build mapping from SMILES to treatment (trt) for sciplex.
    
    Uses ordered_all_smile_chemvae and ordered_all_trt_chemvae from adata.uns.
    
    Args:
        adata: AnnData with uns['ordered_all_smile_chemvae'] and uns['ordered_all_trt_chemvae']
    
    Returns:
        Dictionary mapping SMILES -> trt
    """
    if "ordered_all_smile_chemvae" not in adata.uns:
        raise ValueError("ordered_all_smile_chemvae not found in adata.uns")
    if "ordered_all_trt_chemvae" not in adata.uns:
        raise ValueError("ordered_all_trt_chemvae not found in adata.uns")
    
    smiles = np.array(adata.uns["ordered_all_smile_chemvae"]).astype(str)
    trts = np.array(adata.uns["ordered_all_trt_chemvae"]).astype(str)
    
    return dict(zip(smiles, trts))


def build_product_to_smile_mapping(adata: AnnData) -> Dict[str, str]:
    """
    Build mapping from product_name_clean to SMILES for sciplex.
    
    Args:
        adata: AnnData with obs['product_name_clean'] and obs['perturb_string_processed']
    
    Returns:
        Dictionary mapping product_name_clean -> SMILES
    """
    if "product_name_clean" not in adata.obs.columns:
        raise ValueError("product_name_clean not found in adata.obs")
    if "perturb_string_processed" not in adata.obs.columns:
        raise ValueError("perturb_string_processed not found in adata.obs")
    
    return dict(zip(
        adata.obs["product_name_clean"].astype(str),
        adata.obs["perturb_string_processed"].astype(str)
    ))


def build_trt_to_smile_mapping(adata: AnnData) -> Dict[str, str]:
    """
    Build mapping from treatment (trt) to SMILES for sciplex.
    
    Uses ordered_all_trt_chemvae and ordered_all_smile_chemvae from adata.uns.
    
    Args:
        adata: AnnData with uns['ordered_all_trt_chemvae'] and uns['ordered_all_smile_chemvae']
    
    Returns:
        Dictionary mapping trt -> SMILES
    """
    if "ordered_all_trt_chemvae" not in adata.uns:
        raise ValueError("ordered_all_trt_chemvae not found in adata.uns")
    if "ordered_all_smile_chemvae" not in adata.uns:
        raise ValueError("ordered_all_smile_chemvae not found in adata.uns")
    
    trts = np.array(adata.uns["ordered_all_trt_chemvae"]).astype(str)
    smiles = np.array(adata.uns["ordered_all_smile_chemvae"]).astype(str)
    
    return dict(zip(trts, smiles))


def build_smile_to_product_mapping(adata: AnnData) -> Dict[str, str]:
    """
    Build mapping from SMILES to product_name_clean for sciplex.
    
    Args:
        adata: AnnData with obs['perturb_string_processed'] and obs['product_name_clean']
    
    Returns:
        Dictionary mapping SMILES -> product_name_clean
    """
    if "perturb_string_processed" not in adata.obs.columns:
        raise ValueError("perturb_string_processed not found in adata.obs")
    if "product_name_clean" not in adata.obs.columns:
        raise ValueError("product_name_clean not found in adata.obs")
    
    return dict(zip(
        adata.obs["perturb_string_processed"].astype(str),
        adata.obs["product_name_clean"].astype(str)
    ))


def get_trt_from_perturbation_sciplex(
    adata: AnnData,
    perturbation: str,
    perturbation_col: str = "cov_drug_dose_name",
) -> Optional[str]:
    """
    Get treatment (trt) key from perturbation name for sciplex.
    
    Maps: product_name_clean -> SMILES -> trt
    
    Args:
        adata: AnnData with required uns and obs columns
        perturbation: Perturbation name (e.g., from cov_drug_dose_name)
        perturbation_col: Column name for perturbation
    
    Returns:
        Treatment key (e.g., 'S1001') or None if not found
    """
    product_to_smile = build_product_to_smile_mapping(adata)
    smile_to_trt = build_smile_to_trt_mapping(adata)
    
    if perturbation in product_to_smile:
        smile = product_to_smile[perturbation]
        if smile in smile_to_trt:
            return smile_to_trt[smile]
    
    return None


def get_perturbation_from_trt_sciplex(
    adata: AnnData,
    trt: str,
) -> Optional[str]:
    """
    Get perturbation name (product_name_clean) from treatment (trt) key for sciplex.
    
    Maps: trt -> SMILES -> product_name_clean (reverse direction)
    
    Args:
        adata: AnnData with required uns and obs columns
        trt: Treatment key (e.g., 'S1001')
    
    Returns:
        Perturbation name (product_name_clean) or None if not found
    """
    trt_to_smile = build_trt_to_smile_mapping(adata)
    smile_to_product = build_smile_to_product_mapping(adata)
    
    if trt in trt_to_smile:
        smile = trt_to_smile[trt]
        if smile in smile_to_product:
            return smile_to_product[smile]
    
    return None



def identify_degs_per_perturbation(
    adata: AnnData,
    perturbation_col: str = "condition",
    control_label: str = "control",
    n_genes: int = 50,
    method: str = "t-test",
) -> Dict[str, np.ndarray]:

    sc.tl.rank_genes_groups(
        adata,
        groupby=perturbation_col,
        reference=control_label,
        method=method,
        n_genes=n_genes,
    )
    
    deg_dict = {}
    for perturb in adata.uns["rank_genes_groups"]["names"].dtype.names:
        if perturb == control_label:
            continue
        
        deg_gene_names = adata.uns["rank_genes_groups"]["names"][perturb][:n_genes]
        deg_indices = np.array([
            np.where(adata.var_names == gene)[0][0]
            for gene in deg_gene_names
            if gene in adata.var_names
        ])
        deg_dict[perturb] = deg_indices
    
    return deg_dict


def get_deg_indices_from_precomputed(
    adata: AnnData,
    perturbation: str,
    n_genes: int = 50,
    dataset_mode: Literal["sciplex", "nips"] = "sciplex",
    cell_type: Optional[str] = None,
) -> np.ndarray:


    if dataset_mode == "sciplex":
        return get_deg_indices_sciplex(adata, perturbation, n_genes)
    elif dataset_mode == "nips":
        if cell_type is None:
            raise ValueError("cell_type is required for nips dataset")
        return get_deg_indices_nips(adata, cell_type, perturbation, n_genes)
    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")


def filter_degs_by_logfoldchange(
    adata: AnnData,
    perturbation: str,
    deg_indices: np.ndarray,
    min_log_fc: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if "rank_genes_groups" not in adata.uns:
        raise ValueError("No rank_genes_groups results in adata.uns")
    
    logfc = adata.uns["rank_genes_groups"]["logfoldchanges"][perturbation]
    
    # Filter by magnitude
    large_effect_mask = np.abs(logfc) >= min_log_fc
    large_effect_idx = deg_indices[large_effect_mask[:len(deg_indices)]]
    
    # Split by direction
    logfc_filtered = logfc[:len(deg_indices)]
    upregulated_mask = logfc_filtered > 0
    downregulated_mask = logfc_filtered < 0
    
    upregulated_idx = deg_indices[upregulated_mask]
    downregulated_idx = deg_indices[downregulated_mask]
    
    return large_effect_idx, upregulated_idx, downregulated_idx

