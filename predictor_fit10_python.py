"""Deterministic prediction engine for the final MI20-pooled Cox survival model.

The JSON model export is generated in R after multiple imputation and pooling.
AI-assisted modules in the Streamlit interface do not calculate risk, impute
missing values, fit models, or assign risk groups.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

MODEL_PATH = Path(__file__).with_name("fit10_export_for_python.json")
with MODEL_PATH.open("r", encoding="utf-8") as f:
    MODEL_EXPORT: Dict[str, Any] = json.load(f)

COEFS = MODEL_EXPORT["coefficients"]
FACTOR_LEVELS = MODEL_EXPORT.get("factor_levels", {})
EXPECTED_HORIZONS = [2, 4, 6]
HORIZONS = [int(x) for x in MODEL_EXPORT.get("horizons_years", [])]
if HORIZONS != EXPECTED_HORIZONS:
    raise ValueError(
        "Wrong model JSON loaded: expected horizons_years = [2, 4, 6], "
        f"but found {HORIZONS}. Please replace fit10_export_for_python.json "
        "in the GitHub repository root with the final MI20 pooled Cox JSON."
    )
REQUIRED_FIELDS = list(MODEL_EXPORT.get("predictors", []))
NUMERIC_FIELDS = [field for field in REQUIRED_FIELDS if field not in FACTOR_LEVELS]

RISK_STRATIFICATION = MODEL_EXPORT.get("risk_stratification", {}) or {}
POINT_UNIT = RISK_STRATIFICATION.get("point_unit")
LP_MIN_FOR_POINTS = RISK_STRATIFICATION.get("lp_min_for_points")
POINT_UNIT = float(POINT_UNIT) if POINT_UNIT is not None else None
LP_MIN_FOR_POINTS = float(LP_MIN_FOR_POINTS) if LP_MIN_FOR_POINTS is not None else None
LP_CENTERING_CONSTANT = float(
    MODEL_EXPORT.get("lp_centering_constant", MODEL_EXPORT.get("LP_CENTERING_CONSTANT", 0.0))
)


def _load_baseline_survival() -> Dict[str, float]:
    """Read baseline survival at the requested horizons from the JSON export."""
    if "baseline_survival_at_horizons" in MODEL_EXPORT:
        return {
            f"{int(item['horizon'])}y": float(item["baseline_survival"])
            for item in MODEL_EXPORT["baseline_survival_at_horizons"]
        }

    if "baseline_survival_at_lp0" in MODEL_EXPORT:
        return {str(k): float(v) for k, v in MODEL_EXPORT["baseline_survival_at_lp0"].items()}

    if "baseline_cumulative_hazard" in MODEL_EXPORT:
        table = MODEL_EXPORT["baseline_cumulative_hazard"]
        xs = [float(row["time"]) for row in table]
        ys = [float(row["hazard"]) for row in table]

        def interpolate_hazard(t: float) -> float:
            if t <= xs[0]:
                return ys[0]
            if t >= xs[-1]:
                return ys[-1]
            for i in range(1, len(xs)):
                if xs[i] >= t:
                    x0, x1 = xs[i - 1], xs[i]
                    y0, y1 = ys[i - 1], ys[i]
                    if x1 == x0:
                        return y1
                    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
            return ys[-1]

        return {f"{h}y": math.exp(-interpolate_hazard(float(h))) for h in HORIZONS}

    raise ValueError("No baseline survival information was found in fit10_export_for_python.json.")


BASELINE_SURVIVAL = _load_baseline_survival()


def model_summary() -> Dict[str, Any]:
    """Expose model provenance for display in the web interface."""
    return {
        "model_type": MODEL_EXPORT.get("model_type", "Cox proportional hazards model"),
        "time_origin": MODEL_EXPORT.get("time_origin"),
        "outcome": MODEL_EXPORT.get("outcome"),
        "imputation_method": MODEL_EXPORT.get("imputation_method"),
        "number_of_imputations": MODEL_EXPORT.get("number_of_imputations"),
        "predictor_set": MODEL_EXPORT.get("selected_predictor_set_name"),
        "horizons_years": HORIZONS,
        "predictors": REQUIRED_FIELDS,
        "risk_cutoff_available_for_manuscript_descriptive_analysis": bool(RISK_STRATIFICATION),
    }


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if payload.get(field) in [None, "", []]]


def _coef(*names: str) -> float:
    for name in names:
        if name in COEFS:
            return float(COEFS[name])
    return 0.0


def _to_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value}") from exc


def _assert_model_ready() -> None:
    if not REQUIRED_FIELDS:
        raise ValueError("The model export does not define any predictors.")
    missing_numeric_coef = [field for field in NUMERIC_FIELDS if field not in COEFS]
    if missing_numeric_coef:
        raise ValueError(
            "The model export does not contain the required numeric coefficients: "
            + ", ".join(missing_numeric_coef)
        )
    missing_baseline = [h for h in HORIZONS if f"{h}y" not in BASELINE_SURVIVAL]
    if missing_baseline:
        raise ValueError(f"Baseline survival is unavailable for horizons: {missing_baseline}.")


def _assert_allowed(payload: Dict[str, Any]) -> None:
    _assert_model_ready()
    missing = validate_payload(payload)
    if missing:
        raise ValueError(
            "The following fields are missing and prediction cannot be calculated: "
            + ", ".join(missing)
        )
    for field, levels in FACTOR_LEVELS.items():
        if field in REQUIRED_FIELDS and payload.get(field) not in levels:
            raise ValueError(
                f"Invalid value for {field}: {payload.get(field)}. Allowed values are {levels}."
            )
    for field in NUMERIC_FIELDS:
        _to_float(payload[field], field)


def compute_raw_score(payload: Dict[str, Any]) -> float:
    """Calculate the raw uncentered Cox linear predictor exported from R."""
    _assert_allowed(payload)
    score = 0.0
    for field in REQUIRED_FIELDS:
        value = payload[field]
        if field in NUMERIC_FIELDS:
            score += _to_float(value, field) * _coef(field)
        else:
            value_str = str(value)
            levels = FACTOR_LEVELS.get(field, [])
            if levels and value_str == str(levels[0]):
                continue
            score += _coef(f"{field}{value_str}", f"{field}={value_str}", f"{field}= {value_str}")
    return score


def compute_linear_predictor(payload: Dict[str, Any]) -> float:
    return compute_raw_score(payload) - LP_CENTERING_CONSTANT


def compute_nomogram_points_from_lp(raw_lp: float) -> float | None:
    if POINT_UNIT is None or LP_MIN_FOR_POINTS is None or POINT_UNIT == 0:
        return None
    return (raw_lp - LP_MIN_FOR_POINTS) / POINT_UNIT


def _survival_for_horizon(horizon: int, hazard_multiplier: float) -> float:
    baseline = BASELINE_SURVIVAL.get(f"{horizon}y")
    if baseline is None:
        raise ValueError(f"Baseline survival for {horizon} years is missing from the model export.")
    return float(baseline) ** hazard_multiplier


def predict_fit10_python(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return deterministic 2-/4-/6-year survival and mortality risk estimates."""
    _assert_allowed(payload)
    raw_lp = compute_raw_score(payload)
    lp = raw_lp - LP_CENTERING_CONSTANT
    hazard_multiplier = math.exp(lp)
    points = compute_nomogram_points_from_lp(raw_lp)
    survival = {h: _survival_for_horizon(h, hazard_multiplier) for h in HORIZONS}

    predictions: Dict[str, Any] = {
        "linear_predictor": round(lp, 6),
        "raw_linear_predictor": round(raw_lp, 6),
        "relative_hazard_vs_lp0": round(hazard_multiplier, 6),
        "nomogram_points": round(points, 2) if points is not None else None,
        "risk_group_note": (
            "The web interface intentionally does not display low/high risk labels. "
            "Any exported cutoff is retained only for manuscript-level descriptive plots, "
            "not as an independent treatment-decision threshold."
        ),
    }
    for horizon in HORIZONS:
        predictions[f"survival_{horizon}y"] = round(survival[horizon], 6)
        predictions[f"risk_{horizon}y"] = round(1 - survival[horizon], 6)

    return {
        "model": model_summary(),
        "input": payload,
        "predictions": predictions,
    }
