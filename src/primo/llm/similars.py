import os
import json
from typing import Any, Dict, List, Optional, Sequence
import anndata as ad
import numpy as np
import pandas as pd

from primo.llm.selector import batch_llm_drug_perturbation_selector_optimized

from primo.llm.similar_utils import extract_cell_drug_combinations, similars_entry_has_candidates


def load_llm_credentials() -> tuple[str, str]:
    """
    Load LLM API credentials from environment variables.
    
    Returns:
        tuple: (api_key, base_url)
    
    Raises:
        ValueError: If required credentials are not found.
    """
    api_key = (
        os.getenv("LLM_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "LLM API key not found. Please set one of: LLM_API_KEY, "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY"
        )

    base_url = os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
    if not base_url:
        if os.getenv("ANTHROPIC_API_KEY"):
            base_url = "https://api.anthropic.com"
        elif os.getenv("GEMINI_API_KEY"):
            base_url = "https://generativelanguage.googleapis.com"
        else:
            base_url = "https://api.openai.com/v1"
    return api_key, base_url


def build_unseen_perturbations_sciplex(
    adata: ad.AnnData,
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    target_cell_types: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Build unseen perturbations list for sciplex dataset.
    
    Args:
        adata: Full AnnData object
        adata_train: Training AnnData
        adata_test: Test AnnData
        target_cell_types: List of cell types to process. If None, uses all unique cell types.
    
    Returns:
        Dict mapping cell_type -> list of perturbation dicts with keys:
        - drug_name: str (from product_name_clean)
        - smiles: str (from perturb_string_processed)
        - cell_type: str
    """
    perturb_train = np.array(adata_train.obs["perturb_string_processed"])
    perturb_test = np.array(adata_test.obs["perturb_string_processed"])
    
    trt_unseen_list = np.unique(perturb_test)
    
    if target_cell_types is None:
        target_cell_types = adata.obs["cell_type"].unique().tolist()
    
    valid_perturbations = adata.obs[
        (adata.obs["perturb_string_processed"].isin(trt_unseen_list))
        & (adata.obs["cell_type"].isin(target_cell_types))
    ][["cell_type", "perturb_string_processed", "product_name_clean"]].drop_duplicates()
    
    unseen_perturbations_all = {}
    for cell_type in target_cell_types:
        cell_valid = valid_perturbations[valid_perturbations["cell_type"] == cell_type]
        unseen_perturbations_all[cell_type] = [
            {
                "drug_name": row["product_name_clean"],
                "smiles": row["perturb_string_processed"],
                "cell_type": cell_type,
            }
            for _, row in cell_valid.iterrows()
        ]
    
    return unseen_perturbations_all


def build_unseen_perturbations_nips(
    adata: ad.AnnData,
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    target_cell_types: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Build unseen perturbations list for nips dataset.
    
    Args:
        adata: Full AnnData object
        adata_train: Training AnnData
        adata_test: Test AnnData
        target_cell_types: List of cell types to process. If None, uses default nips cell types.
    
    Returns:
        Dict mapping cell_type -> list of perturbation dicts
    """
    if target_cell_types is None:
        target_cell_types = [
            "NK cells",
            "T cells CD4+",
            "T regulatory cells",
            "T cells CD8+",
            "B cells",
            "Myeloid cells",
        ]
    
    perturb_train = np.array(adata_train.obs["SMILES"])
    perturb_test = np.array(adata_test.obs["SMILES"])
    trt_unseen_list = np.unique(perturb_test)
    
    valid_perturbations = adata.obs[
        (adata.obs["SMILES"].isin(trt_unseen_list))
        & (adata.obs["cell_type"].isin(target_cell_types))
    ][["cell_type", "condition", "SMILES"]].drop_duplicates()
    
    unseen_perturbations_all = {}
    for cell_type in target_cell_types:
        cell_valid = valid_perturbations[valid_perturbations["cell_type"] == cell_type]
        unseen_perturbations_all[cell_type] = [
            {
                "drug_name": row["condition"],
                "smiles": row["SMILES"],
                "cell_type": cell_type,
            }
            for _, row in cell_valid.iterrows()
        ]
    
    return unseen_perturbations_all


def extract_train_drug_list_sciplex(
    adata_train: ad.AnnData, cell_type: str
) -> List[str]:
    """Extract train drug list for sciplex dataset."""
    counts = adata_train.obs["cov_drug_dose_name"].unique()
    return extract_cell_drug_combinations(counts)


def extract_train_drug_list_nips(
    adata_train: ad.AnnData, cell_type: str
) -> List[str]:
    """Extract train drug list for nips dataset."""
    counts = adata_train.obs[adata_train.obs["cell_type"] == cell_type][
        "cov_drug_name"
    ].unique()
    return extract_cell_drug_combinations(counts)


def batch_generate_similars(
    unseen_perturbations: List[Dict],
    train_drug_list: List[str],
    max_candidates: int = 3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_format: str = "auto",
    request_timeout: Optional[int] = None,
    anthropic_enable_thinking: bool = True,
    anthropic_thinking_budget_tokens: Optional[int] = None,
    anthropic_max_tokens: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate similars_map using LLM for a batch of unseen perturbations.
    
    Args:
        unseen_perturbations: List of dicts with keys: drug_name, smiles, cell_type
        train_drug_list: List of training drug combinations (format: "CELLTYPE_DRUG")
        max_candidates: Maximum number of similar drugs to return per perturbation
        api_key: LLM API key. If None, loads from environment.
        base_url: LLM API base URL. If None, uses default or from environment.
        model: Optional model name override.
        api_format: API format to use. Supports auto/openai/gemini/anthropic.
        request_timeout: Optional request timeout override in seconds.
        anthropic_enable_thinking: Whether to enable Anthropic thinking blocks.
        anthropic_thinking_budget_tokens: Optional Anthropic thinking budget override.
        anthropic_max_tokens: Optional Anthropic max_tokens override.
    
    Returns:
        Dict mapping "cell_type_drug_name" -> structured LLM-selected candidates
    """
    if api_key is None or base_url is None:
        loaded_key, loaded_url = load_llm_credentials()
        api_key = api_key or loaded_key
        base_url = base_url or loaded_url
    
    return batch_llm_drug_perturbation_selector_optimized(
        unseen_perturbations=unseen_perturbations,
        train_drug_list=train_drug_list,
        max_candidates=max_candidates,
        api_key=api_key,
        base_url=base_url,
        model=model,
        api_format=api_format,
        request_timeout=request_timeout,
        anthropic_enable_thinking=anthropic_enable_thinking,
        anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
        anthropic_max_tokens=anthropic_max_tokens,
    )


def save_similars_map(similars_map: Dict[str, Any], filepath: str) -> None:
    """Save similars_map to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(similars_map, f, ensure_ascii=False, indent=2)
    print(f"similars_map saved to: {filepath}")


def load_similars_map(filepath: str) -> Optional[Dict[str, Any]]:
    """Load similars_map from JSON file."""
    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return None
    
    with open(filepath, "r", encoding="utf-8") as f:
        similars_map = json.load(f)
    print(f"similars_map loaded from: {filepath}")
    return similars_map


def retry_empty_perturbations(
    filepath: str,
    adata: ad.AnnData,
    train_drug_list: List[str],
    cell_type: str,
    dataset_mode: str = "nips",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_candidates: int = 3,
) -> Dict[str, Any]:
    """
    Retry generating similars for perturbations that returned empty lists.
    
    Args:
        filepath: Path to similars_map JSON file
        adata: Full AnnData object
        train_drug_list: Training drug list
        cell_type: Target cell type
        dataset_mode: "sciplex" or "nips"
        api_key: LLM API key (optional)
        base_url: LLM API base URL (optional)
        max_candidates: Maximum candidates per perturbation
    
    Returns:
        Updated similars_map dict
    """
    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return None
    
    similars_map = load_similars_map(filepath)
    if similars_map is None:
        return None
    
    empty_perturbations = [
        key for key, value in similars_map.items()
        if not similars_entry_has_candidates(value)
    ]
    
    if not empty_perturbations:
        print("No empty perturbations found")
        return similars_map
    
    print(f"Found {len(empty_perturbations)} empty perturbations: {empty_perturbations}")
    
    retry_unseen_list = []
    original_keys_map = {}
    
    for perturbation in empty_perturbations:
        perturbation_name = (
            perturbation.split("_", 1)[1] if "_" in perturbation else perturbation
        )
        
        try:
            if dataset_mode == "nips":
                target_smiles = adata.obs[adata.obs["condition"] == perturbation_name][
                    "SMILES"
                ].unique()[0]
                drug_name = perturbation_name
            else:
                target_smiles = adata.obs[
                    adata.obs["product_name_clean"] == perturbation_name
                ]["perturb_string_processed"].unique()[0]
                drug_name = perturbation_name
            
            retry_unseen_list.append(
                {
                    "drug_name": drug_name,
                    "smiles": target_smiles,
                    "cell_type": cell_type,
                }
            )
            generated_key = f"{cell_type}_{perturbation_name}"
            original_keys_map[generated_key] = perturbation
        except Exception as e:
            print(f"Cannot get SMILES for {perturbation_name}: {e}")
    
    if not retry_unseen_list:
        print("Cannot build retry list")
        return similars_map
    
    print(f"Retrying {len(retry_unseen_list)} perturbations...")
    
    new_results = batch_generate_similars(
        retry_unseen_list,
        train_drug_list,
        max_candidates=max_candidates,
        api_key=api_key,
        base_url=base_url,
    )
    
    updated_count = 0
    for gen_key, candidates in new_results.items():
        original_key = original_keys_map.get(gen_key)
        if original_key and similars_entry_has_candidates(candidates):
            similars_map[original_key] = candidates
            n_selected = len((candidates or {}).get("selected_drugs", []))
            print(f"  Successfully found {n_selected} similar drugs for {original_key}")
            updated_count += 1
        elif original_key:
            print(f"  {original_key} still has no similar drugs after retry")
    
    if updated_count > 0:
        save_similars_map(similars_map, filepath)
        print(f"\nUpdated {updated_count} perturbations, file saved")
    else:
        print("\nRetry did not update any perturbations")
    
    return similars_map
