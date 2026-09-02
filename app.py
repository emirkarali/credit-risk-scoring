"""Credit risk scoring service.

Serves the calibrated LightGBM pipeline exported by
`credit_risk.ipynb`.

Design notes
------------
* The notebook creates four features *before* the sklearn pipeline runs
  (three financial ratios and an employment-anomaly flag). Those definitions
  are exported into `artifacts/model_config.json` and re-applied here, so
  training and serving cannot silently drift apart. Leaving them unset would
  send NaN for the model's single most important feature.
* The service reports how many columns the caller actually supplied. Any
  column left unset is filled by the pipeline's imputer with a training
  statistic, which anchors the prediction toward an average applicant. That
  count is surfaced to the UI rather than hidden.
* `CODE_GENDER` is deliberately not collected. See the fairness section of
  the notebook.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("credit_risk")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "credit_risk_model.joblib"
CONFIG_PATH = BASE_DIR / "artifacts" / "model_config.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Run credit_risk.ipynb from the project root "
        "first - it writes the artifacts/ directory."
    )
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Model config not found: {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    MODEL_CONFIG: Dict[str, Any] = json.load(config_file)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(MODEL_PATH)


def _resolve_feature_columns(estimator: Any) -> List[str]:
    """Find the training column order, whether or not the pipeline is wrapped.

    The deployed artifact is a CalibratedClassifierCV around the Pipeline, so
    the preprocessor is not reachable at the top level.
    """
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        return list(names)

    steps = getattr(estimator, "named_steps", None)
    if steps and "preprocessor" in steps:
        return list(steps["preprocessor"].feature_names_in_)

    for attribute in ("estimator", "base_estimator"):
        inner = getattr(estimator, attribute, None)
        if inner is not None:
            try:
                return _resolve_feature_columns(inner)
            except AttributeError:
                pass

    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated:
        return _resolve_feature_columns(calibrated[0].estimator)

    raise AttributeError("Could not determine training feature columns.")


FEATURE_COLUMNS: List[str] = _resolve_feature_columns(model)
THRESHOLD = float(MODEL_CONFIG.get("threshold", 0.5))
IS_CALIBRATED = bool(MODEL_CONFIG.get("calibrated", False))
BASE_RATE = float(MODEL_CONFIG.get("train_default_rate", 0.0807))
DAYS_EMPLOYED_PLACEHOLDER = int(MODEL_CONFIG.get("days_employed_placeholder", 365243))

# Fields the UI may submit. WEEKDAY_APPR_PROCESS_START is categorical in the
# training data ("MONDAY", ...) and must not appear in the numeric set, or it
# would be coerced to NaN on every request.
CATEGORICAL_FIELDS: Set[str] = {
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
}

YES_NO_FIELDS: Set[str] = {"FLAG_OWN_CAR", "FLAG_OWN_REALTY"}

# Ranges used only to reject obviously invalid input early, so the caller gets
# a clear 400 instead of a confident prediction built on nonsense.
FIELD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "AMT_INCOME_TOTAL": (0.0, 1e9),
    "AMT_CREDIT": (0.0, 1e9),
    "AMT_ANNUITY": (0.0, 1e8),
    "AMT_GOODS_PRICE": (0.0, 1e9),
    "CNT_CHILDREN": (0.0, 25.0),
    "REGION_RATING_CLIENT": (1.0, 3.0),
    "EXT_SOURCE_1": (0.0, 1.0),
    "EXT_SOURCE_2": (0.0, 1.0),
    "EXT_SOURCE_3": (0.0, 1.0),
    "age_years": (18.0, 100.0),
    "years_employed": (0.0, 60.0),
}

app = FastAPI(title="Credit Risk Scoring API", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --------------------------------------------------------------- value parsing

def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"nan", "none", "null"}
    return False


def _to_float(value: Any, field: str) -> Optional[float]:
    if _is_blank(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a number.")
    if not np.isfinite(number):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a finite number.")

    if field in FIELD_BOUNDS:
        low, high = FIELD_BOUNDS[field]
        if not (low <= number <= high):
            raise HTTPException(
                status_code=400,
                detail=f"'{field}' must be between {low:g} and {high:g}.",
            )
    return number


def _to_yes_no(value: Any) -> Optional[str]:
    if _is_blank(value):
        return None
    text = str(value).strip().upper()
    if text in {"Y", "YES", "TRUE", "1"}:
        return "Y"
    if text in {"N", "NO", "FALSE", "0"}:
        return "N"
    return text


# ------------------------------------------------- feature-engineering parity

def _apply_engineered_features(record: Dict[str, Any]) -> None:
    """Recreate the notebook's pre-pipeline feature engineering.

    Formulas mirror `model_config["engineered_features"]`. Any ratio whose
    inputs are missing or would divide by zero is left as NaN for the
    imputer, which is the same treatment those rows received in training.
    """
    income = record.get("AMT_INCOME_TOTAL")
    credit = record.get("AMT_CREDIT")
    annuity = record.get("AMT_ANNUITY")

    def ratio(numerator: Any, denominator: Any) -> float:
        if numerator is None or denominator is None:
            return np.nan
        try:
            numerator = float(numerator)
            denominator = float(denominator)
        except (TypeError, ValueError):
            return np.nan
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
            return np.nan
        return numerator / denominator

    if "CREDIT_INCOME_RATIO" in record:
        record["CREDIT_INCOME_RATIO"] = ratio(credit, income)
    if "ANNUITY_INCOME_RATIO" in record:
        record["ANNUITY_INCOME_RATIO"] = ratio(annuity, income)
    if "CREDIT_ANNUITY_RATIO" in record:
        record["CREDIT_ANNUITY_RATIO"] = ratio(credit, annuity)

    if "DAYS_EMPLOYED_ANOM" in record:
        days_employed = record.get("DAYS_EMPLOYED")
        if days_employed is None or (
            isinstance(days_employed, float) and np.isnan(days_employed)
        ):
            # Unknown employment history is exactly what the placeholder
            # encoded in the raw data, so the flag is set rather than imputed.
            record["DAYS_EMPLOYED_ANOM"] = 1
        elif int(days_employed) == DAYS_EMPLOYED_PLACEHOLDER:
            record["DAYS_EMPLOYED_ANOM"] = 1
            record["DAYS_EMPLOYED"] = np.nan
        else:
            record["DAYS_EMPLOYED_ANOM"] = 0


def _build_input_frame(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
    record: Dict[str, Any] = {column: np.nan for column in FEATURE_COLUMNS}
    supplied: Set[str] = set()

    age_years = _to_float(payload.get("age_years"), "age_years")
    if age_years is not None:
        record["DAYS_BIRTH"] = -age_years * 365.25
        supplied.add("DAYS_BIRTH")

    employment_years = _to_float(payload.get("years_employed"), "years_employed")
    if employment_years is not None:
        record["DAYS_EMPLOYED"] = -employment_years * 365.25
        supplied.add("DAYS_EMPLOYED")

    for key, value in payload.items():
        if key in {"age_years", "years_employed"}:
            continue
        if key not in record:
            continue
        if _is_blank(value):
            continue

        if key in YES_NO_FIELDS:
            record[key] = _to_yes_no(value)
        elif key in CATEGORICAL_FIELDS:
            record[key] = str(value).strip()
        else:
            number = _to_float(value, key)
            if number is None:
                continue
            record[key] = number
        supplied.add(key)

    _apply_engineered_features(record)

    row = pd.DataFrame([record], columns=FEATURE_COLUMNS)
    return row, len(supplied)


# -------------------------------------------------------------- risk banding

def _risk_band(probability: float) -> Dict[str, Any]:
    """Express risk relative to the portfolio base rate.

    A raw percentage is hard to read on its own: 12% sounds low until you
    know the average applicant sits at ~8%. The lift makes that explicit.
    """
    lift = probability / BASE_RATE if BASE_RATE > 0 else float("nan")

    if probability >= THRESHOLD:
        band = "High Risk"
    elif lift >= 1.0:
        band = "Elevated Risk"
    else:
        band = "Low Risk"

    return {"band": band, "lift_vs_base_rate": round(float(lift), 2)}


# ------------------------------------------------------------------- routes

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_CONFIG.get("model", "unknown"),
        "calibrated": IS_CALIBRATED,
        "threshold": THRESHOLD,
        "n_features": len(FEATURE_COLUMNS),
    }


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """Everything the frontend needs to describe the model honestly."""
    return {
        "model": MODEL_CONFIG.get("model", "unknown"),
        "calibrated": IS_CALIBRATED,
        "threshold": THRESHOLD,
        "threshold_metric": MODEL_CONFIG.get("threshold_metric"),
        "threshold_alternatives": MODEL_CONFIG.get("threshold_alternatives", {}),
        "base_rate": BASE_RATE,
        "total_columns": len(FEATURE_COLUMNS),
        "test_metrics": MODEL_CONFIG.get("test_metrics", {}),
        "versions": MODEL_CONFIG.get("versions", {}),
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/predict")
async def predict(payload: Dict[str, Any]):
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Request body must be a non-empty object.")

    input_frame, supplied_count = _build_input_frame(payload)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probability = float(model.predict_proba(input_frame)[0, 1])
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001 - surfaced to the caller as 500
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    band = _risk_band(probability)

    return {
        "probability": probability,
        "calibrated": IS_CALIBRATED,
        "is_high_risk": probability >= THRESHOLD,
        "threshold": THRESHOLD,
        "base_rate": BASE_RATE,
        "risk_label": band["band"],
        "lift_vs_base_rate": band["lift_vs_base_rate"],
        "features_supplied": supplied_count,
        "total_columns": len(FEATURE_COLUMNS),
    }
