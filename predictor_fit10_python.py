import json
import math
from pathlib import Path
from typing import Any, Dict, List

MODEL_PATH = Path(__file__).with_name("fit10_export_for_python.json")
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    MODEL_EXPORT = json.load(f)

COEFS = MODEL_EXPORT["coefficients"]
FACTOR_LEVELS = MODEL_EXPORT.get("factor_levels", {})
HORIZONS = [int(x) for x in MODEL_EXPORT.get("horizons_years", [3, 5, 7])]
REQUIRED_FIELDS = list(MODEL_EXPORT.get("predictors", [
    "Age_at_onset", "disease_duration_baseline", "GBA1_mutation", "T2D", "DBS",
    "UPDRS_Part_III", "HY_Stage", "Falls", "Depression", "Cognitive_dysfunction", "LEDD"
]))
NUMERIC_FIELDS = [field for field in REQUIRED_FIELDS if field not in FACTOR_LEVELS]

RISK_STRATIFICATION = MODEL_EXPORT.get("risk_stratification", {}) or {}
LP_CUTOFF = RISK_STRATIFICATION.get("lp_cutoff", None)
POINTS_CUTOFF = RISK_STRATIFICATION.get("points_cutoff", None)
POINT_UNIT = RISK_STRATIFICATION.get("point_unit", None)
LP_MIN_FOR_POINTS = RISK_STRATIFICATION.get("lp_min_for_points", None)
LP_CUTOFF = float(LP_CUTOFF) if LP_CUTOFF is not None else None
POINTS_CUTOFF = float(POINTS_CUTOFF) if POINTS_CUTOFF is not None else None
POINT_UNIT = float(POINT_UNIT) if POINT_UNIT is not None else None
LP_MIN_FOR_POINTS = float(LP_MIN_FOR_POINTS) if LP_MIN_FOR_POINTS is not None else None
LP_CENTERING_CONSTANT = float(MODEL_EXPORT.get("lp_centering_constant", MODEL_EXPORT.get("LP_CENTERING_CONSTANT", 0.0)))


def _load_baseline_survival() -> Dict[str, float]:
    if "baseline_survival_at_horizons" in MODEL_EXPORT:
        out = {}
        for item in MODEL_EXPORT["baseline_survival_at_horizons"]:
            h = int(item["horizon"])
            out[f"{h}y"] = float(item["baseline_survival"])
        return out
    if "baseline_survival_at_lp0" in MODEL_EXPORT:
        return {str(k): float(v) for k, v in MODEL_EXPORT["baseline_survival_at_lp0"].items()}
    if "baseline_cumulative_hazard" in MODEL_EXPORT:
        table = MODEL_EXPORT["baseline_cumulative_hazard"]
        xs = [float(r["time"]) for r in table]
        ys = [float(r["hazard"]) for r in table]

        def interp(t: float) -> float:
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

        return {f"{h}y": math.exp(-interp(float(h))) for h in HORIZONS}
    raise ValueError("No baseline survival information found in fit10_export_for_python.json.")


BASELINE_SURVIVAL = _load_baseline_survival()


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    return [k for k in REQUIRED_FIELDS if payload.get(k) in [None, "", []]]


def _coef(*names: str) -> float:
    for name in names:
        if name in COEFS:
            return float(COEFS[name])
    return 0.0


def _assert_model_ready() -> None:
    missing_coef = [field for field in NUMERIC_FIELDS if field not in COEFS]
    if missing_coef:
        raise ValueError(
            "The current fit10_export_for_python.json does not contain required numeric coefficients: "
            + ", ".join(missing_coef)
        )


def _to_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except Exception:
        raise ValueError(f"Invalid numeric value for {field}: {value}")


def _assert_allowed(payload: Dict[str, Any]) -> None:
    _assert_model_ready()
    missing = validate_payload(payload)
    if missing:
        raise ValueError("The following fields are missing and prediction cannot be calculated: " + ", ".join(missing))
    for field, levels in FACTOR_LEVELS.items():
        if field in REQUIRED_FIELDS and payload.get(field) not in levels:
            raise ValueError(f"Invalid value for {field}: {payload.get(field)}. Allowed values are {levels}.")
    for field in NUMERIC_FIELDS:
        _to_float(payload[field], field)


def compute_raw_score(payload: Dict[str, Any]) -> float:
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
    # R export uses centered=FALSE baseline hazard and raw uncentered LP.
    # The centering constant is kept for backward compatibility, but is normally zero.
    return compute_raw_score(payload) - LP_CENTERING_CONSTANT


def compute_nomogram_points_from_lp(raw_lp: float) -> float | None:
    if POINT_UNIT is None or LP_MIN_FOR_POINTS is None or POINT_UNIT == 0:
        return None
    return (raw_lp - LP_MIN_FOR_POINTS) / POINT_UNIT


def classify_risk_by_lp(lp: float) -> str:
    if LP_CUTOFF is None:
        return "Not assigned"
    return "High risk" if lp > LP_CUTOFF else "Low risk"


def _survival_for_horizon(horizon: int, hr: float) -> float:
    key = f"{horizon}y"
    if key not in BASELINE_SURVIVAL:
        raise ValueError(f"Baseline survival for {key} is missing from model export.")
    return float(BASELINE_SURVIVAL[key]) ** hr


def predict_fit10_python(payload: Dict[str, Any]) -> Dict[str, Any]:
    missing = validate_payload(payload)
    if missing:
        raise ValueError(f"The following fields are missing and prediction cannot be calculated: {', '.join(missing)}")

    raw_lp = compute_raw_score(payload)
    lp = raw_lp - LP_CENTERING_CONSTANT
    hr = math.exp(lp)
    points = compute_nomogram_points_from_lp(raw_lp)
    surv = {h: _survival_for_horizon(h, hr) for h in HORIZONS}
    risk_label = classify_risk_by_lp(raw_lp if (RISK_STRATIFICATION.get("lp_scale") == "raw_uncentered_linear_predictor") else lp)

    preds = {
        "linear_predictor": round(lp, 4),
        "raw_linear_predictor": round(raw_lp, 4),
        "hazard_ratio_vs_lp0": round(hr, 4),
        "nomogram_points": round(points, 1) if points is not None else None,
        "risk_group": risk_label,
        "lp_cutoff_for_risk_group": round(LP_CUTOFF, 4) if LP_CUTOFF is not None else None,
        "points_cutoff_for_risk_group": round(POINTS_CUTOFF, 1) if POINTS_CUTOFF is not None else None,
    }

    for h in HORIZONS:
        preds[f"survival_{h}y"] = round(surv[h], 4)
        preds[f"risk_{h}y"] = round(1 - surv[h], 4)

    return {"input": payload, "predictions": preds}
