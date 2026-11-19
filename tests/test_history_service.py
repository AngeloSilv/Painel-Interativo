"""A filtragem usa dias locais completos e não expõe observações após emissão."""

from datetime import date
from dataclasses import replace

import pandas as pd
import pytest

from services.data_loader import load_data
from services.history_service import filter_history, forecast_history, future_observation_count, history_summary, region_history


def test_filter_respects_local_day_and_includes_last_hour():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-02T02:00:00Z", periods=26, freq="h"),
        "observed_load": 100.0,
    })
    filtered = filter_history(frame, date(2026, 1, 2), date(2026, 1, 2))
    assert len(filtered) == 24
    assert filtered["timestamp"].iloc[0] == pd.Timestamp("2026-01-02T03:00:00Z")
    assert filtered["timestamp"].iloc[-1] == pd.Timestamp("2026-01-03T02:00:00Z")


def test_filter_rejects_reversed_dates():
    with pytest.raises(ValueError, match="final"):
        filter_history(pd.DataFrame(), date(2026, 2, 2), date(2026, 2, 1))


def test_forecast_context_cannot_include_later_observations():
    bundle = load_data("demo")
    forecast = min(bundle.forecasts, key=lambda run: run.issued_at)
    history = forecast_history(forecast, bundle.observations)
    assert len(history) == 72
    assert history["timestamp"].max() == forecast.issued_at
    assert (history["timestamp"] <= forecast.issued_at).all()
    assert history["region"].eq(forecast.region).all()


def test_gaps_are_reported_without_filling():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T02:00Z"]),
        "observed_load": [100.0, 200.0],
        "region": ["A", "A"], "unit": ["MW", "MW"],
    })
    assert history_summary(frame) == {"count": 2, "mean_load": 150.0, "peak_load": 200.0, "missing_hours": 1}
    assert region_history(frame, "B").empty
    assert len(frame) == 2


def test_future_dated_observations_are_not_presented_as_known():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00Z", "2100-01-01T00:00Z"]),
        "observed_load": [100.0, 900.0],
        "region": ["A", "A"], "unit": ["MW", "MW"],
    })
    history = region_history(frame, "A", as_of=pd.Timestamp("2026-01-02T00:00Z"))
    assert history["observed_load"].tolist() == [100.0]
    assert future_observation_count(frame, "A", as_of=pd.Timestamp("2026-01-02T00:00Z")) == 1


def test_history_mean_handles_large_finite_loads():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        "observed_load": [1e308, 1e308],
    })
    assert history_summary(frame)["mean_load"] == pytest.approx(1e308)


def test_forecast_history_preserves_nanosecond_window_boundary():
    from tests.test_analysis_service import make_forecast

    forecast = replace(make_forecast(), issued_at=pd.Timestamp("2025-01-08T00:00:00.000000001Z"))
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-05T00:00:00.000000001Z", periods=73, freq="h"),
        "observed_load": 100.0, "region": forecast.region, "unit": forecast.unit,
    })
    history = forecast_history(forecast, frame, hours=72)
    assert len(history) == 72
    assert history["timestamp"].min() == pd.Timestamp("2025-01-05T01:00:00.000000001Z")
    assert history["timestamp"].max() == forecast.issued_at
