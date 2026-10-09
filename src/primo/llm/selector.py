#!/usr/bin/env python3

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, parse, request

from primo.paths import REPO_ROOT

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-preview"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4.6"
DEFAULT_ANTHROPIC_SYSTEM_PROMPT = "You are an expert in 'context-aware drug perturbation analysis', specializing in matching drug responses across specific biological contexts. You possess deep knowledge of medicinal chemistry (SMILES), pharmacological mechanisms (MOA), and how cell-line specific genetic backgrounds (e.g., gene expression, mutations in CCLE/DepMap) influence drug sensitivity."
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_ANTHROPIC_REQUEST_TIMEOUT = 900
DEFAULT_ANTHROPIC_MAX_TOKENS = 65536
DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS = 2048
ANTHROPIC_VERSION = "2023-06-01"
PROMPT_TEMPLATE_PATH = REPO_ROOT / "assets" / "prompts" / "llm_drug_select_prompt.txt"


def _make_target_key(drug: Dict[str, str]) -> str:
    return f"{drug['cell_type']}_{drug['drug_name']}"


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def _compact_text(text: str) -> str:
    return re.sub(r"[\s_]+", "", _normalize_text(text))


def _cell_variants(cell_type: str) -> List[str]:
    variants = [cell_type, cell_type.replace(" ", "_"), cell_type.replace("_", " ")]
    return list(dict.fromkeys(v for v in variants if v))


def _looks_like_dose(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?(?:[a-zA-Z%]+)?", token.strip()))


def _normalize_confidence_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not score == score:
        return None
    return float(min(10.0, max(0.0, score)))


def load_prompt_template(prompt_template_path: Optional[str] = None) -> str:
    template_path = Path(prompt_template_path) if prompt_template_path else PROMPT_TEMPLATE_PATH
    return template_path.read_text(encoding="utf-8")


class LLMDrugSelector:
    def __init__(
        self,
        api_format: str = "auto",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 3,
        enable_web: bool = False,
        per_target_allowed_list: bool = True,
        prompt_template_path: Optional[str] = None,
        request_timeout: Optional[int] = None,
        anthropic_enable_thinking: bool = True,
        anthropic_thinking_budget_tokens: Optional[int] = None,
        anthropic_max_tokens: Optional[int] = None,
    ):
        self.enable_web = bool(enable_web)
        self.per_target_allowed_list = bool(per_target_allowed_list)
        self.api_format = self._resolve_api_format(api_format, base_url)
        self.base_url = self._resolve_base_url(base_url, self.api_format)
        self.model = model or self._default_model(self.api_format)
        self.api_key = (
            api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        self.batch_size = batch_size
        self.prompt_template = load_prompt_template(prompt_template_path)
        self.anthropic_enable_thinking = bool(anthropic_enable_thinking)
        self.anthropic_thinking_budget_tokens = int(
            anthropic_thinking_budget_tokens
            or os.getenv(
                "ANTHROPIC_THINKING_BUDGET_TOKENS",
                str(DEFAULT_ANTHROPIC_THINKING_BUDGET_TOKENS),
            )
        )
        self.anthropic_max_tokens = int(
            anthropic_max_tokens
            or os.getenv(
                "ANTHROPIC_MAX_TOKENS",
                str(DEFAULT_ANTHROPIC_MAX_TOKENS),
            )
        )
        if request_timeout is not None:
            self.request_timeout = int(request_timeout)
        elif self.api_format == "anthropic":
            self.request_timeout = int(
                os.getenv(
                    "ANTHROPIC_REQUEST_TIMEOUT",
                    os.getenv(
                        "LLM_REQUEST_TIMEOUT",
                        str(DEFAULT_ANTHROPIC_REQUEST_TIMEOUT),
                    ),
                )
            )
        else:
            self.request_timeout = int(
                os.getenv("LLM_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT))
            )

        if not self.api_key:
            raise ValueError(
                "Missing API key. Pass api_key or set ANTHROPIC_API_KEY / "
                "OPENAI_API_KEY / GEMINI_API_KEY / LLM_API_KEY."
            )

    def _resolve_api_format(self, api_format: str, base_url: Optional[str]) -> str:
        if api_format not in {"auto", "openai", "gemini", "anthropic"}:
            raise ValueError("api_format must be one of: auto, openai, gemini, anthropic")

        if api_format != "auto":
            return api_format

        normalized_base = (base_url or "").strip().lower()
        if "anthropic" in normalized_base:
            return "anthropic"
        if base_url and "/v1beta" in base_url:
            return "gemini"
        if self.enable_web:
            return "gemini"
        return "openai"

    def _resolve_base_url(self, base_url: Optional[str], api_format: str) -> str:
        if api_format == "openai":
            url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
            return url if url.endswith("/v1") else f"{url}/v1"

        if api_format == "anthropic":
            url = (base_url or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")
            if url.endswith("/v1"):
                return url[: -len("/v1")]
            return url

        url = (base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")
        if url.endswith("/v1beta"):
            return url[: -len("/v1beta")]
        if url.endswith("/v1"):
            return url[: -len("/v1")]
        return url

    def _default_model(self, api_format: str) -> str:
        if api_format == "anthropic":
            return DEFAULT_ANTHROPIC_MODEL
        if api_format == "gemini":
            return DEFAULT_GEMINI_MODEL
        return DEFAULT_OPENAI_MODEL

    def _post_json(
        self,
        url: str,
        payload: Dict,
        include_auth_header: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if include_auth_header:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)

        req = request.Request(url, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code}: {details}") from exc

    def _call_openai_chat(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        raw = self._post_json(url, payload)
        message = raw.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
        return str(content).strip()

    def _call_gemini_generate_content(self, prompt: str) -> str:
        model_name = parse.quote(self.model, safe="-._")
        api_key = parse.quote(self.api_key, safe="")
        url = f"{self.base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        if self.enable_web:
            payload["tools"] = [{"urlContext": {}}]

        raw = self._post_json(url, payload, include_auth_header=False)
        if "error" in raw:
            raise RuntimeError(str(raw["error"]))

        parts = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()

    def _call_anthropic_messages(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": self.anthropic_max_tokens,
            "temperature": 0.0,
            "system": DEFAULT_ANTHROPIC_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.anthropic_enable_thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.anthropic_thinking_budget_tokens,
            }
        raw = self._post_json(
            url,
            payload,
            include_auth_header=False,
            extra_headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
        )
        content = raw.get("content", [])
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if text:
            return text

        content_types = [
            str(block.get("type", ""))
            for block in content
            if isinstance(block, dict)
        ]
        stop_reason = raw.get("stop_reason")
        if content_types:
            raise RuntimeError(
                "Anthropic response contained no text blocks "
                f"(content_types={content_types}, stop_reason={stop_reason}). "
                "If thinking is enabled, the model may have exhausted max_tokens "
                "before producing the final answer."
            )
        return text

    def _call_llm(self, prompt: str) -> str:
        try:
            if self.api_format == "anthropic":
                return self._call_anthropic_messages(prompt)
            if self.api_format == "gemini":
                return self._call_gemini_generate_content(prompt)
            return self._call_openai_chat(prompt)
        except Exception as exc:
            print(f"LLM API Error: {exc}")
            return ""

    def _format_drug_name_list(self, drug_names: List[str]) -> str:
        return ", ".join(drug_names) if drug_names else "None"

    def _build_batch_prompt(
        self,
        target_drugs: List[Dict],
        target_allowed_drug_names: Dict[str, List[str]],
        train_drug_names: List[str],
        max_candidates: int,
    ) -> str:
        targets_info = []
        for index, drug in enumerate(target_drugs, start=1):
            key = _make_target_key(drug)
            target_block = (
                f"Target {index}:\n"
                f"Name: {drug['drug_name']}\n"
                f"SMILES: {drug['smiles']}\n"
                f"Cell Type: {drug['cell_type']}"
            )
            if self.per_target_allowed_list:
                allowed_list = self._format_drug_name_list(
                    target_allowed_drug_names.get(key, [])
                )
                target_block += f"\nAllowed Drugs In This Cell: {allowed_list}"
            targets_info.append(target_block)

        if self.per_target_allowed_list:
            available_drug_summary = (
                "Each target block includes its own allowed candidate drugs in the current cell context. "
                "You must only select drugs from that target's own 'Allowed Drugs In This Cell' list. "
                "Do not borrow candidates across targets."
            )
        else:
            available_drug_summary = self._format_drug_name_list(train_drug_names)

        prompt = self.prompt_template
        prompt = prompt.replace("__TARGETS_TEXT__", "\n---\n".join(targets_info))
        prompt = prompt.replace("__TRAIN_DRUG_LIST__", available_drug_summary)
        prompt = prompt.replace("__AVAILABLE_DRUG_SUMMARY__", available_drug_summary)
        prompt = prompt.replace("__MAX_CANDIDATES__", str(max_candidates))
        return prompt

    def _extract_drug_name_for_cell(
        self, full_key: str, cell_type: str
    ) -> Optional[str]:
        for variant in _cell_variants(cell_type):
            prefix = f"{variant}_"
            if not full_key.startswith(prefix):
                continue

            remainder = full_key[len(prefix) :]
            parts = remainder.split("_")
            if len(parts) >= 2 and _looks_like_dose(parts[-1]):
                return "_".join(parts[:-1])
            return remainder

        return None

    def _build_cell_drug_index(
        self, train_drug_list: List[str], cell_types: List[str]
    ) -> Tuple[Dict[str, Dict[str, List[str]]], List[str], Dict[str, List[str]]]:
        cell_drug_index = {cell_type: {} for cell_type in cell_types}
        available_names = set()
        available_names_by_cell = {cell_type: set() for cell_type in cell_types}

        for full_key in train_drug_list:
            for cell_type in cell_types:
                drug_name = self._extract_drug_name_for_cell(full_key, cell_type)
                if not drug_name:
                    continue

                available_names.add(drug_name)
                available_names_by_cell[cell_type].add(drug_name)
                norm_name = _normalize_text(drug_name)
                cell_drug_index[cell_type].setdefault(norm_name, []).append(full_key)

        return (
            cell_drug_index,
            sorted(available_names),
            {
                cell_type: sorted(names)
                for cell_type, names in available_names_by_cell.items()
            },
        )

    def _match_full_keys_for_name(
        self, recommended_name: str, drug_index_for_cell: Dict[str, List[str]]
    ) -> List[str]:
        clean_name = recommended_name.strip(".,\"'")
        norm_name = _normalize_text(clean_name)
        compact_name = _compact_text(clean_name)

        if norm_name in drug_index_for_cell:
            return drug_index_for_cell[norm_name]

        for indexed_name, full_keys in drug_index_for_cell.items():
            if compact_name == _compact_text(indexed_name):
                return full_keys

        for indexed_name, full_keys in drug_index_for_cell.items():
            compact_indexed = _compact_text(indexed_name)
            if abs(len(compact_name) - len(compact_indexed)) >= 5:
                continue
            if compact_name in compact_indexed or compact_indexed in compact_name:
                return full_keys

        return []

    def _resolve_target_key(
        self,
        target_label: str,
        target_drugs: List[Dict],
        block_index: int,
        used_keys: set,
    ) -> Optional[str]:
        normalized_label = _normalize_text(target_label)
        compact_label = _compact_text(target_label)
        contextual_matches = []
        drug_matches = []

        for drug in target_drugs:
            key = _make_target_key(drug)
            if key in used_keys:
                continue

            drug_name = drug["drug_name"]
            cell_type = drug["cell_type"]

            if _normalize_text(drug_name) == normalized_label:
                drug_matches.append(key)
            elif _compact_text(drug_name) == compact_label:
                drug_matches.append(key)

            if (
                _compact_text(drug_name) in compact_label
                and _compact_text(cell_type) in compact_label
            ):
                contextual_matches.append(key)

        if len(contextual_matches) == 1:
            return contextual_matches[0]
        if len(drug_matches) == 1:
            return drug_matches[0]

        if block_index < len(target_drugs):
            positional_key = _make_target_key(target_drugs[block_index])
            if positional_key not in used_keys:
                return positional_key

        if contextual_matches:
            return contextual_matches[0]
        if drug_matches:
            return drug_matches[0]
        return None

    def _parse_selected_drugs(
        self,
        selected_text: str,
    ) -> List[Dict[str, Any]]:
        selected_text = selected_text.strip()
        if not selected_text or selected_text.lower() == "none":
            return []

        parsed: List[Dict[str, Any]] = []
        line_pattern = re.compile(
            r"^(?:[-*]|\d+[.)])?\s*(?P<drug>.+?)\s*\|\s*"
            r"(?:confidence|score)\s*:\s*(?P<score>-?\d+(?:\.\d+)?)\s*$",
            re.IGNORECASE,
        )

        for raw_line in selected_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = line_pattern.match(line)
            if not match:
                continue
            parsed.append(
                {
                    "drug_name": match.group("drug").strip(),
                    "score": _normalize_confidence_score(match.group("score")),
                }
            )

        if parsed:
            return parsed

        global_pattern = re.compile(
            r"(?P<drug>[^,\n|]+?)\s*\|\s*(?:confidence|score)\s*:\s*"
            r"(?P<score>-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        for match in global_pattern.finditer(selected_text):
            parsed.append(
                {
                    "drug_name": match.group("drug").strip(),
                    "score": _normalize_confidence_score(match.group("score")),
                }
            )

        if parsed:
            return parsed

        return [
            {
                "drug_name": name.strip(),
                "score": None,
            }
            for name in re.split(r"[,，\n]", selected_text)
            if name.strip()
        ]

    def _parse_batch_response(
        self, response: str, target_drugs: List[Dict]
    ) -> Dict[str, Dict[str, Any]]:
        results_map = {
            _make_target_key(drug): {"reasoning": "", "selected_drugs": []}
            for drug in target_drugs
        }
        used_keys = set()
        blocks = [block.strip() for block in response.split("===") if block.strip()]

        for block_index, block in enumerate(blocks):
            target_match = re.search(r"Target Drug:\s*(.+)", block, re.IGNORECASE)
            if not target_match:
                continue

            selected_match = re.search(
                r"Selected Drugs:\s*(.*)",
                block,
                re.IGNORECASE | re.DOTALL,
            )
            if not selected_match:
                continue

            reasoning_match = re.search(
                r"Reasoning:\s*(.*?)\s*Selected Drugs:",
                block,
                re.IGNORECASE | re.DOTALL,
            )
            selected_text = selected_match.group(1).strip()
            parsed_selected = self._parse_selected_drugs(selected_text)

            resolved_key = self._resolve_target_key(
                target_match.group(1).strip(),
                target_drugs,
                block_index,
                used_keys,
            )
            if not resolved_key:
                continue

            results_map[resolved_key] = {
                "reasoning": (
                    reasoning_match.group(1).strip() if reasoning_match else ""
                ),
                "selected_drugs": parsed_selected,
            }
            used_keys.add(resolved_key)

        return results_map

    def process_batch(
        self,
        unseen_perturbations: List[Dict],
        train_drug_list: List[str],
        max_candidates: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        all_results: Dict[str, Dict[str, Any]] = {}
        total_items = len(unseen_perturbations)

        for index in range(0, total_items, self.batch_size):
            batch = unseen_perturbations[index : index + self.batch_size]
            batch_number = index // self.batch_size + 1
            batch_cell_types = list(dict.fromkeys(drug["cell_type"] for drug in batch))
            target_lookup = {_make_target_key(drug): drug for drug in batch}
            (
                cell_drug_index,
                train_drug_names,
                allowed_names_by_cell,
            ) = self._build_cell_drug_index(
                train_drug_list, batch_cell_types
            )
            target_allowed_drug_names = {
                _make_target_key(drug): allowed_names_by_cell.get(drug["cell_type"], [])
                for drug in batch
            }

            print(f"\nProcessing batch {batch_number} ({len(batch)} drugs)...")
            print(
                f"Available drugs in current batch cell contexts: {len(train_drug_names)}"
            )

            for drug in batch:
                all_results.setdefault(
                    _make_target_key(drug),
                    {"reasoning": "", "selected_drugs": []},
                )

            if not train_drug_names:
                print("No train drugs found for these cell contexts, skipping LLM call.")
                continue

            prompt = self._build_batch_prompt(
                batch,
                target_allowed_drug_names,
                train_drug_names,
                max_candidates,
            )
            response = self._call_llm(prompt)
            print(f"Raw LLM Response (First 200 chars):\n{response[:200]}...\n")

            batch_results_names = self._parse_batch_response(response, batch)

            for key, payload in batch_results_names.items():
                target_cell_type = target_lookup[key]["cell_type"]
                drug_index_for_cell = cell_drug_index.get(target_cell_type, {})
                final_candidates: List[Dict[str, Any]] = []
                score_by_full_key: Dict[str, Dict[str, Any]] = {}

                for rec_index, rec in enumerate(payload.get("selected_drugs", [])):
                    recommended_name = str(rec.get("drug_name", "")).strip()
                    if not recommended_name:
                        continue
                    matched_keys = self._match_full_keys_for_name(
                        recommended_name,
                        drug_index_for_cell,
                    )
                    score = _normalize_confidence_score(rec.get("score"))
                    if score is None:
                        score = 1.0

                    for matched_key in matched_keys:
                        existing = score_by_full_key.get(matched_key)
                        candidate_payload = {
                            "drug_name": recommended_name,
                            "full_key": matched_key,
                            "score": score,
                            "_order": rec_index,
                        }
                        if (
                            existing is None
                            or float(candidate_payload["score"])
                            > float(existing["score"])
                        ):
                            score_by_full_key[matched_key] = candidate_payload

                final_candidates = sorted(
                    score_by_full_key.values(),
                    key=lambda item: (-float(item["score"]), int(item["_order"])),
                )[:max_candidates]
                for rank, item in enumerate(final_candidates, start=1):
                    item["rank"] = rank
                    item.pop("_order", None)

                all_results[key] = {
                    "reasoning": str(payload.get("reasoning", "")).strip(),
                    "selected_drugs": final_candidates,
                }
                print(f"  {key}: found {len(final_candidates)} valid candidates")

            if index + self.batch_size < total_items:
                time.sleep(1)

        return all_results


def batch_llm_drug_perturbation_selector_optimized(
    unseen_perturbations: List[Dict],
    train_drug_list: List[str],
    max_candidates: int = 3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_format: str = "auto",
    enable_web: bool = True,
    per_target_allowed_list: bool = True,
    batch_size: int = 3,
    prompt_template_path: Optional[str] = None,
    request_timeout: Optional[int] = None,
    anthropic_enable_thinking: bool = True,
    anthropic_thinking_budget_tokens: Optional[int] = None,
    anthropic_max_tokens: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    selector = LLMDrugSelector(
        api_format=api_format,
        model=model,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size,
        enable_web=enable_web,
        per_target_allowed_list=per_target_allowed_list,
        prompt_template_path=prompt_template_path,
        request_timeout=request_timeout,
        anthropic_enable_thinking=anthropic_enable_thinking,
        anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
        anthropic_max_tokens=anthropic_max_tokens,
    )
    return selector.process_batch(unseen_perturbations, train_drug_list, max_candidates)
