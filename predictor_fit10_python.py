import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

MODEL_PATH = Path(__file__).with_name("fit10_export_for_python.json")
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    MODEL_EXPORT = json.load(f)

COEFS = MODEL_EXPORT["coefficients"]
FACTOR_LEVELS = MODEL_EXPORT["factor_levels"]
HORIZONS = [int(x) for x in MODEL_EXPORT.get("horizons_years", [3, 5, 7])]

# New R export: baseline_survival_at_horizons = [{"horizon": 3, "baseline_survival": ...}, ...]
# Old export: baseline_survival_at_lp0 = {"3y": ..., "5y": ..., "7y": ...}
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
        # Fallback: linear interpolation of H0(t), then S0(t)=exp(-H0(t)).
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
                    x0, x1 = xs[i-1], xs[i]
                    y0, y1 = ys[i-1], ys[i]
                    if x1 == x0:
                        return y1
                    return y0 + (y1-y0) * (t-x0) / (x1-x0)
            return ys[-1]
        return {f"{h}y": math.exp(-interp(float(h))) for h in HORIZONS}
    raise ValueError("No baseline survival information found in fit10_export_for_python.json.")

BASELINE_SURVIVAL = _load_baseline_survival()
LP_CENTERING_CONSTANT = float(MODEL_EXPORT.get("lp_centering_constant", MODEL_EXPORT.get("LP_CENTERING_CONSTANT", 0.0)))
RISK_STRATIFICATION = MODEL_EXPORT.get("risk_stratification", {}) or {}
LP_CUTOFF = RISK_STRATIFICATION.get("lp_cutoff", None)
POINTS_CUTOFF = RISK_STRATIFICATION.get("points_cutoff", None)
LP_CUTOFF = float(LP_CUTOFF) if LP_CUTOFF is not None else None
POINTS_CUTOFF = float(POINTS_CUTOFF) if POINTS_CUTOFF is not None else None

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
]


def validate_payload(payload: Dict[str, Any]) -> list[str]:
    return [k for k in REQUIRED_FIELDS if payload.get(k) in [None, "", []]]


def _coef(*names: str) -> float:
    """Get a coefficient while allowing both old and new R/Python naming conventions."""
    for name in names:
        if name in COEFS:
            return float(COEFS[name])
    # Treat absent reference-level coefficients as 0.
    return 0.0


def _assert_model_ready() -> None:
    required_any = ["disease_duration_baseline", "UPDRS_Part_III"]
    missing = [k for k in required_any if k not in COEFS]
    if missing:
        raise ValueError(
            "The current fit10_export_for_python.json does not contain required coefficients: "
            + ", ".join(missing)
        )


def _assert_allowed(payload: Dict[str, Any]) -> None:
    _assert_model_ready()
    for field, levels in FACTOR_LEVELS.items():
        if payload.get(field) not in levels:
            raise ValueError(f"Invalid value for {field}: {payload.get(field)}. Allowed values are {levels}.")
    float(payload["disease_duration_baseline"])
    float(payload["UPDRS_Part_III"])


def compute_raw_score(payload: Dict[str, Any]) -> float:
    _assert_allowed(payload)
    s = 0.0

    if payload["Age_at_onset"] == ">50":
        s += _coef("Age_at_onset>50", "Age_at_onset=>50", "Age_at_onset= >50", "Age_at_onset=> 50")

    s += float(payload["disease_duration_baseline"]) * _coef("disease_duration_baseline")

    if payload["GBA1_mutation"] == "Yes":
        s += _coef("GBA1_mutationYes", "GBA1_mutation=Yes")
    if payload["T2D"] == "Yes":
        s += _coef("T2DYes", "T2D=Yes")
    if payload["DBS"] == "Yes":
        s += _coef("DBSYes", "DBS=Yes")

    s += float(payload["UPDRS_Part_III"]) * _coef("UPDRS_Part_III")

    hy_stage = str(payload["HY_Stage"])
    if hy_stage != "1":
        s += _coef(f"HY_Stage{hy_stage}", f"HY_Stage={hy_stage}")

    if payload["Falls"] == "Yes":
        s += _coef("FallsYes", "Falls=Yes")
    if payload["Depression"] == "Yes":
        s += _coef("DepressionYes", "Depression=Yes")
    if payload["Cognitive_dysfunction"] == "Yes":
        s += _coef("Cognitive_dysfunctionYes", "Cognitive_dysfunction=Yes")

    return s


def compute_linear_predictor(payload: Dict[str, Any]) -> float:
    return compute_raw_score(payload) - LP_CENTERING_CONSTANT


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

    lp = compute_linear_predictor(payload)
    hr = math.exp(lp)

    surv = {h: _survival_for_horizon(h, hr) for h in HORIZONS}
    risk_label = classify_risk_by_lp(lp)

    preds = {
        "linear_predictor": round(lp, 4),
        "hazard_ratio_vs_lp0": round(hr, 4),
        "risk_group": risk_label,
        "lp_cutoff_for_risk_group": round(LP_CUTOFF, 4) if LP_CUTOFF is not None else None,
        "points_cutoff_for_risk_group": round(POINTS_CUTOFF, 1) if POINTS_CUTOFF is not None else None,
    }

    for h in HORIZONS:
        preds[f"survival_{h}y"] = round(surv[h], 4)
        preds[f"risk_{h}y"] = round(1 - surv[h], 4)

    return {"input": payload, "predictions": preds}
