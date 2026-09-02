"""Smoke tests for the scoring service.

These run against the real artifacts, so they are skipped until the notebook
has been executed. Their job is to catch training/serving drift: the derived
features and the categorical field mapping are the two places where the API
can silently disagree with the notebook.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts" / "credit_risk_model.joblib"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="Run credit_risk.ipynb first to build artifacts/.",
)


@pytest.fixture(scope="module")
def client():
    import sys

    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    import app as application

    return TestClient(application.app), application


def test_health(client):
    api, _ = client
    body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] > 100


def test_model_is_calibrated(client):
    """An uncalibrated model must not have its output shown as a percentage."""
    _, application = client
    assert application.IS_CALIBRATED, "Deployed artifact should be the calibrated pipeline."


def test_derived_ratios_match_training_formulas(client):
    _, application = client
    frame, _ = application._build_input_frame(
        {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000}
    )
    assert frame["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(500000 / 150000)
    assert frame["ANNUITY_INCOME_RATIO"].iloc[0] == pytest.approx(25000 / 150000)
    assert frame["CREDIT_ANNUITY_RATIO"].iloc[0] == pytest.approx(500000 / 25000)


def test_engineered_feature_definitions_match_config(client):
    """The config is the contract; the code must implement exactly it."""
    _, application = client
    declared = set(application.MODEL_CONFIG["engineered_features"])
    assert declared == {
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_ANNUITY_RATIO",
        "DAYS_EMPLOYED_ANOM",
    }


def test_unknown_employment_sets_anomaly_flag(client):
    _, application = client
    frame, _ = application._build_input_frame({"AMT_CREDIT": 500000})
    assert frame["DAYS_EMPLOYED_ANOM"].iloc[0] == 1


def test_zero_denominator_yields_nan_not_inf(client):
    _, application = client
    frame, _ = application._build_input_frame({"AMT_INCOME_TOTAL": 0, "AMT_CREDIT": 100000})
    assert np.isnan(frame["CREDIT_INCOME_RATIO"].iloc[0])


def test_weekday_is_not_coerced_to_nan(client):
    """Regression test: this field is categorical, not numeric."""
    _, application = client
    frame, _ = application._build_input_frame({"WEEKDAY_APPR_PROCESS_START": "MONDAY"})
    assert frame["WEEKDAY_APPR_PROCESS_START"].iloc[0] == "MONDAY"


def test_prediction_is_a_plausible_probability(client):
    api, _ = client
    body = api.post(
        "/predict",
        json={
            "AMT_INCOME_TOTAL": 150000,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 25000,
            "age_years": 35,
            "years_employed": 5,
            "EXT_SOURCE_2": 0.62,
        },
    ).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_label"] in {"Low Risk", "Elevated Risk", "High Risk"}


def test_blank_form_is_not_confidently_high_risk(client):
    """A calibrated model should score an unknown applicant near the base rate."""
    api, application = client
    body = api.post("/predict", json={"AMT_CREDIT": 500000}).json()
    assert body["probability"] < 5 * application.BASE_RATE


@pytest.mark.parametrize(
    "payload", [{"AMT_INCOME_TOTAL": "abc"}, {"EXT_SOURCE_1": 5}, {"age_years": 9}, {}]
)
def test_invalid_input_is_rejected(client, payload):
    api, _ = client
    assert api.post("/predict", json=payload).status_code == 400
