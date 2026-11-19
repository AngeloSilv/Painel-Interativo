"""O contrato rejeita dados ambíguos em vez de corrigi-los silenciosamente."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from contracts.forecast_schema import ForecastValidationError, validate_forecast


@pytest.fixture
def payload() -> dict:
    issued_at = datetime(2026, 2, 11, 23, tzinfo=timezone(timedelta(hours=-3)))
    return {
        "model_name": "Modelo de teste",
        "model_version": "1.0",
        "horizon_hours": 24,
        "region": "Região de teste",
        "unit": "MW",
        "issued_at": issued_at.isoformat(),
        "predictions": [
            {"timestamp": (issued_at + timedelta(hours=lead)).isoformat(), "predicted_load": 1000 + lead}
            for lead in range(1, 25)
        ],
    }


def test_valid_forecast_normalizes_utc_and_preserves_issue(payload):
    forecast = validate_forecast(payload)
    assert forecast.issued_at == pd.Timestamp("2026-02-12T02:00:00Z")
    assert forecast.issued_at_original == "2026-02-11T23:00:00-03:00"
    assert str(forecast.predictions["timestamp"].dt.tz) == "UTC"
    assert forecast.predictions.iloc[0]["timestamp"] == pd.Timestamp("2026-02-12T03:00:00Z")
    assert len(forecast.predictions) == 24


def test_valid_48_hour_forecast(payload):
    payload["horizon_hours"] = 48
    start = datetime.fromisoformat(payload["issued_at"])
    payload["predictions"] = [
        {"timestamp": (start + timedelta(hours=lead)).isoformat(), "predicted_load": 1000.0}
        for lead in range(1, 49)
    ]
    assert len(validate_forecast(payload).predictions) == 48


def test_legacy_forecast_has_unspecified_split_without_year_inference(payload):
    assert validate_forecast(payload).dataset_split == "unspecified"


@pytest.mark.parametrize("split", ["validation_2025", "test_2025", "forecast_2026", "unspecified"])
def test_accepts_explicit_dataset_splits(payload, split):
    payload["dataset_split"] = split
    assert validate_forecast(payload).dataset_split == split


@pytest.mark.parametrize("split", [None, "train", "validation", 2025, [], True])
def test_rejects_unknown_dataset_splits(payload, split):
    payload["dataset_split"] = split
    with pytest.raises(ForecastValidationError, match="dataset_split"):
        validate_forecast(payload)


def test_preserves_mwmed_without_unit_conversion(payload):
    payload["unit"] = "MWmed"
    forecast = validate_forecast(payload)
    assert forecast.unit == "MWmed"
    assert forecast.predictions.iloc[0]["predicted_load"] == payload["predictions"][0]["predicted_load"]


@pytest.mark.parametrize("field", [
    "model_name", "model_version", "horizon_hours", "region", "unit", "issued_at", "predictions",
])
def test_rejects_missing_fields(payload, field):
    del payload[field]
    with pytest.raises(ForecastValidationError, match="ausentes"):
        validate_forecast(payload)


@pytest.mark.parametrize("horizon", [0, 12, 23, 25, 49, 24.0, "24", True, None])
def test_rejects_invalid_horizon(payload, horizon):
    payload["horizon_hours"] = horizon
    with pytest.raises(ForecastValidationError, match="horizon_hours"):
        validate_forecast(payload)


@pytest.mark.parametrize("count", [0, 23, 25, 48])
def test_requires_exact_forecast_count(payload, count):
    original = deepcopy(payload["predictions"])
    payload["predictions"] = (original * 2)[:count]
    with pytest.raises(ForecastValidationError, match="exatamente 24"):
        validate_forecast(payload)


@pytest.mark.parametrize("timestamp", [
    "2026-02-12T00:00:00", "2026-02-30T00:00:00-03:00", "ontem",
    "2026-02-12", None, 123, "10000-01-01T00:00:00Z",
])
def test_rejects_invalid_or_naive_target_timestamps(payload, timestamp):
    payload["predictions"][0]["timestamp"] = timestamp
    with pytest.raises(ForecastValidationError, match="timestamp"):
        validate_forecast(payload)


def test_rejects_issue_without_timezone(payload):
    payload["issued_at"] = "2026-02-11T23:00:00"
    with pytest.raises(ForecastValidationError, match="issued_at"):
        validate_forecast(payload)


def test_rejects_duplicate_instants_with_different_offsets(payload):
    payload["predictions"][1]["timestamp"] = "2026-02-12T03:00:00Z"
    with pytest.raises(ForecastValidationError, match="duplicados"):
        validate_forecast(payload)


def test_rejects_unordered_targets(payload):
    payload["predictions"][0], payload["predictions"][1] = payload["predictions"][1], payload["predictions"][0]
    with pytest.raises(ForecastValidationError, match="ordenados"):
        validate_forecast(payload)


def test_rejects_non_hourly_targets(payload):
    payload["predictions"][1]["timestamp"] = "2026-02-12T01:30:00-03:00"
    with pytest.raises(ForecastValidationError, match="consecutivos"):
        validate_forecast(payload)


def test_nanosecond_precision_is_not_silently_truncated(payload):
    payload["predictions"][1]["timestamp"] = "2026-02-12T01:00:00.000000001-03:00"
    with pytest.raises(ForecastValidationError, match="consecutivos"):
        validate_forecast(payload)


def test_rejects_target_at_or_before_issue(payload):
    payload["issued_at"] = payload["predictions"][0]["timestamp"]
    with pytest.raises(ForecastValidationError, match="posteriores"):
        validate_forecast(payload)


@pytest.mark.parametrize("gap_seconds", [3601, 7 * 24 * 3600])
def test_rejects_gap_before_forecast_horizon(payload, gap_seconds):
    first_target = datetime.fromisoformat(payload["predictions"][0]["timestamp"])
    payload["issued_at"] = (first_target - timedelta(seconds=gap_seconds)).isoformat()
    with pytest.raises(ForecastValidationError, match="primeiro alvo"):
        validate_forecast(payload)


def test_accepts_next_full_hour_after_issue_with_minutes(payload):
    payload["issued_at"] = "2026-02-11T23:15:00-03:00"
    forecast = validate_forecast(payload)
    assert (forecast.predictions.iloc[0]["timestamp"] - forecast.issued_at).total_seconds() == 45 * 60


def test_rejects_first_target_more_than_one_hour_even_by_nanosecond(payload):
    payload["issued_at"] = "2026-02-11T22:59:59.999999999-03:00"
    with pytest.raises(ForecastValidationError, match="primeiro alvo"):
        validate_forecast(payload)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "1000", True, None, -1])
def test_rejects_invalid_loads(payload, value):
    payload["predictions"][0]["predicted_load"] = value
    with pytest.raises(ForecastValidationError, match="predicted_load"):
        validate_forecast(payload)


@pytest.mark.parametrize("unit", ["kW", "MWh", "mw", " MW ", None])
def test_rejects_inconsistent_units(payload, unit):
    payload["unit"] = unit
    with pytest.raises(ForecastValidationError, match="unidade"):
        validate_forecast(payload)


@pytest.mark.parametrize("field", ["model_name", "model_version", "region"])
def test_rejects_empty_metadata(payload, field):
    payload[field] = " "
    with pytest.raises(ForecastValidationError, match=field):
        validate_forecast(payload)


def test_rejects_missing_prediction_field(payload):
    del payload["predictions"][0]["predicted_load"]
    with pytest.raises(ForecastValidationError, match="obrigatórios"):
        validate_forecast(payload)


@pytest.mark.parametrize("invalid", [[], None, "payload"])
def test_rejects_non_object_payload(invalid):
    with pytest.raises(ForecastValidationError, match="objeto JSON"):
        validate_forecast(invalid)
