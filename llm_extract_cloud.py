import json
import re
from typing import Any, Dict

import requests

from cloud_config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TIMEOUT,
)
from prompts import SYSTEM_PROMPT

REQUIRED_FIELDS = [
    "Age_at_onset",
    "disease_duration_baseline",
    "GBA1_mutation",
    "T2D",
    "DBS",
    "UPDRS_Part_III",
    "HY_Stage",
    "Falls",
    "Depression",
    "Cognitive_dysfunction",
    "LEDD",
]

NUMERIC_FIELDS = ["disease_duration_baseline", "UPDRS_Part_III", "LEDD"]
ALLOWED_LEVELS = {
    "Age_at_onset": ["≤50", ">50"],
    "GBA1_mutation": ["No", "Yes"],
    "T2D": ["No", "Yes"],
    "DBS": ["No", "Yes"],
    "HY_Stage": ["1", "2", "2.5", "3", "4", "5"],
    "Falls": ["No", "Yes"],
    "Depression": ["No", "Yes"],
    "Cognitive_dysfunction": ["No", "Yes"],
}


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    return json.loads(_strip_code_fence(text))


def _to_float_or_none(value: Any):
    if value in [None, "", []]:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            return float(match.group(0))
        return value
    try:
        return float(value)
    except Exception:
        return value


def _normalize_output(data: Dict[str, Any]) -> Dict[str, Any]:
    result = {field: data.get(field) for field in REQUIRED_FIELDS}
    result["missing_fields"] = data.get("missing_fields", []) or []
    result["uncertainties"] = data.get("uncertainties", []) or []

    for field in NUMERIC_FIELDS:
        result[field] = _to_float_or_none(result.get(field))

    for field, levels in ALLOWED_LEVELS.items():
        value = result.get(field)
        if value in [None, "", []]:
            result[field] = None
        else:
            value = str(value).strip()
            if value not in levels:
                result["uncertainties"].append(f"{field}: unsupported value '{value}'")
                result[field] = None
            else:
                result[field] = value

    missing = [k for k in REQUIRED_FIELDS if result.get(k) in [None, "", []]]
    result["missing_fields"] = sorted(list(set(result.get("missing_fields", []) + missing)))
    result["uncertainties"] = sorted(list(set(result.get("uncertainties", []))))
    result["can_predict"] = len(result["missing_fields"]) == 0
    return result


def extract_variables_from_text(case_text: str) -> Dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        return {"ok": False, "error": "DeepSeek API key was not detected in Streamlit secrets or environment variables.", "data": None}

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract the PD survival model variables from the de-identified case summary below. Do not calculate risk:\n\n{case_text.strip()}"},
        ],
        "temperature": 0,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = _safe_json_loads(content)
        return {"ok": True, "error": None, "data": _normalize_output(parsed)}
    except requests.HTTPError as e:
        return {"ok": False, "error": f"DeepSeek API error: {str(e)}; response: {getattr(e.response, 'text', '')[:500]}", "data": None}
    except Exception as e:
        return {"ok": False, "error": f"AI extraction failed: {str(e)}", "data": None}
