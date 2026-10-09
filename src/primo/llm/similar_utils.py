from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from primo.paths import ARTIFACT_DIR


def load_similars_map(filepath=None):
    """
    从JSON文件加载similars_map

    Args:
        filepath: str, 文件路径

    Returns:
        dict, 相似药物映射字典
    """
    filepath = filepath or str(ARTIFACT_DIR / "similars_map.json")
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        similars_map = json.load(f)
    print(f"similars_map已从 {filepath} 加载")
    return similars_map


def _normalize_similars_score(
    raw_score: Any,
    default: float = 1.0,
    min_score: float = 0.0,
    max_score: float = 10.0,
) -> float:
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(score):
        return float(default)
    return float(min(max_score, max(min_score, score)))


def _transform_similars_score_to_weight(
    raw_score: Any,
    *,
    mode: str = "linear",
    threshold: float = 0.0,
) -> float:
    score = _normalize_similars_score(raw_score, default=1.0)
    threshold = float(max(0.0, threshold))
    mode_norm = str(mode or "linear").strip().lower()

    if mode_norm == "linear":
        return float(score)

    if mode_norm == "threshold_uniform_kept":
        if score < threshold:
            return 0.0
        return 1.0

    if mode_norm == "threshold_ceil_sqrt_ceil":
        if score < threshold:
            return 0.0
        return float(np.ceil(np.sqrt(np.ceil(score))))

    raise ValueError(f"Unsupported similars weight mode: {mode}")


def _extract_similars_drug_name(
    full_key: str,
    cell_type: Optional[str] = None,
) -> Optional[str]:
    text = str(full_key or "").strip()
    if not text:
        return None
    if cell_type:
        prefix = f"{cell_type}_"
        if text.startswith(prefix):
            return text[len(prefix) :]
    if "_" in text:
        return text.split("_", 1)[1]
    return text


def extract_similars_candidates(
    entry: Any,
    cell_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if isinstance(entry, dict):
        candidates = (
            entry.get("selected_drugs")
            or entry.get("candidates")
            or entry.get("similar_drugs")
            or []
        )
    elif isinstance(entry, list):
        candidates = entry
    elif entry in (None, "", False):
        candidates = []
    else:
        candidates = [entry]

    normalized: List[Dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, str):
            full_key = item.strip()
            if not full_key:
                continue
            normalized.append(
                {
                    "full_key": full_key,
                    "drug_name": _extract_similars_drug_name(full_key, cell_type),
                    "score": 1.0,
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        full_key = item.get("full_key") or item.get("candidate") or item.get("key")
        drug_name = item.get("drug_name") or item.get("drug")
        if not full_key and drug_name:
            full_key = f"{cell_type}_{drug_name}" if cell_type else str(drug_name)
        if not full_key:
            continue

        normalized.append(
            {
                "full_key": str(full_key),
                "drug_name": (
                    str(drug_name)
                    if drug_name
                    else _extract_similars_drug_name(str(full_key), cell_type)
                ),
                "score": _normalize_similars_score(
                    item.get("score", item.get("confidence", item.get("weight"))),
                    default=1.0,
                ),
            }
        )

    return normalized


def similars_entry_has_candidates(entry: Any) -> bool:
    return len(extract_similars_candidates(entry)) > 0


def extract_cell_drug_combinations(cov_drug_dose_names):
    """
    从 'CELLTYPE_DRUGNAME_DOSE' 格式的列表中提取唯一的 'CELLTYPE_DRUGNAME' 组合

    Parameters:
    -----------
    cov_drug_dose_names : list or array
        包含 'CELLTYPE_DRUGNAME_DOSE' 格式字符串的列表

    Returns:
    --------
    list[str]
        包含唯一 'CELLTYPE_DRUGNAME' 组合的列表
    """
    cell_drug_combinations = set()

    for item in cov_drug_dose_names:
        parts = item.split("_")
        if len(parts) >= 2:  # 确保至少有 cell_type 和 drug_name
            cell_type_drug = f"{parts[0]}_{parts[1]}"
            cell_drug_combinations.add(cell_type_drug)

    # 转换为排序的列表
    return sorted(list(cell_drug_combinations))
