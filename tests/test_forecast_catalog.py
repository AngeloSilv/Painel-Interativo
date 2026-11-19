"""O seletor permanece leve e materializa somente uma emissão íntegra."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from contracts.forecast_schema import Forecast, ForecastValidationError
from services.analysis_service import forecast_identifier
from services.forecast_catalog import CATALOG_COLUMNS, forecast_catalog, materialize_forecast


def native_run(forecast_id="run24", horizon=24, issue="2025-01-08T03:00:00Z"):
    issued_at = pd.Timestamp(issue)
    return pd.DataFrame({
        "forecast_id": forecast_id, "model_name": f"Modelo original {horizon}",
        "model_version": "não informada", "horizon_hours": horizon,
        "issued_at": issued_at, "timestamp": pd.date_range(issued_at.to_pydatetime() + timedelta(hours=1), periods=horizon, freq="h"),
        "forecast_step": np.arange(1, horizon + 1), "lead_hour": np.arange(1, horizon + 1, dtype=float),
        "predicted_load": np.arange(horizon, dtype=float) + 1000,
        "observed_load": np.nan, "persistence_load": np.nan, "seasonal_load": np.nan,
        "region": "Município do Rio de Janeiro", "unit": "MWmed", "dataset_split": "test_2025",
        "source_file": f"predictions_{horizon}h.csv", "issued_at_original": "2025-01-08 00:00:00",
    })


def bundle(rows=None, forecasts=None):
    return SimpleNamespace(prediction_rows=pd.DataFrame() if rows is None else rows, forecasts=forecasts or [])


def json_run(horizon=24):
    rows = native_run(horizon=horizon)
    return Forecast(
        model_name="Modelo JSON", model_version="1.0", horizon_hours=horizon,
        issued_at=rows.issued_at.iloc[0], predictions=rows[["timestamp", "predicted_load"]],
        region="Município do Rio de Janeiro", unit="MWmed", dataset_split="test_2025",
        source_path=Path("modelo.json"), issued_at_original="2025-01-08T00:00:00-03:00",
    )


def test_native_catalog_has_one_row_per_id_without_materializing_forecasts(monkeypatch):
    rows = pd.concat([native_run(), native_run("run48", 48, "2025-01-09T03:00Z")], ignore_index=True)
    original = rows.copy(deep=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("O catálogo não deve construir ou validar Forecasts do lote.")
    monkeypatch.setattr("services.forecast_catalog.validate_forecast", forbidden)
    monkeypatch.setattr("services.forecast_catalog.forecast_identifier", forbidden)
    catalog = forecast_catalog(bundle(rows))
    assert catalog.columns.tolist() == CATALOG_COLUMNS
    assert catalog.forecast_id.tolist() == ["run48", "run24"]
    assert catalog.horizon_hours.tolist() == [48, 24]
    assert str(catalog.issued_at.dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_frame_equal(rows, original)


@pytest.mark.parametrize("column,value", [
    ("model_name", "Outro modelo"), ("model_version", "1.0"),
    ("horizon_hours", 48), ("issued_at", pd.Timestamp("2025-01-09T03:00Z")),
    ("region", "Outra região"), ("unit", "MW"), ("dataset_split", "validation_2025"),
    ("source_file", "outro.csv"),
])
def test_native_catalog_rejects_conflicting_metadata_for_same_id(column, value):
    rows = native_run()
    rows.loc[0, column] = value
    with pytest.raises(ValueError, match="conflitantes"):
        forecast_catalog(bundle(rows))


def test_native_catalog_marks_absent_version_as_unknown_without_inventing_version():
    rows = native_run().drop(columns="model_version")
    assert forecast_catalog(bundle(rows)).model_version.tolist() == ["não informada"]
    assert materialize_forecast(bundle(rows), "run24").model_version == "não informada"


def test_native_catalog_requires_source_metadata_and_aware_origin():
    with pytest.raises(ValueError, match="obrigatórios"):
        forecast_catalog(bundle(native_run().drop(columns="region")))
    rows = native_run()
    rows["issued_at"] = rows["issued_at"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="fuso"):
        forecast_catalog(bundle(rows))


def test_native_materialization_preserves_original_emission_source_and_48_hours():
    rows = pd.concat([native_run(), native_run("run48", 48)], ignore_index=True)
    forecast = materialize_forecast(bundle(rows), "run48")
    assert forecast.horizon_hours == len(forecast.predictions) == 48
    assert forecast.issued_at == pd.Timestamp("2025-01-08T03:00:00Z")
    assert forecast.issued_at_original == "2025-01-08 00:00:00"
    assert forecast.source_path == Path("predictions_48h.csv")
    assert forecast.model_version == "não informada"
    assert forecast.unit == "MWmed" and forecast.dataset_split == "test_2025"


def test_native_materialization_validates_only_selected_run_once(monkeypatch):
    rows = pd.concat([native_run(), native_run("broken48", 48)], ignore_index=True)
    rows.loc[rows.forecast_id.eq("broken48"), "predicted_load"] = np.nan
    from services import forecast_catalog as module
    original_validate = module.validate_forecast
    calls = []
    def track(payload):
        calls.append(payload)
        return original_validate(payload)
    monkeypatch.setattr(module, "validate_forecast", track)
    forecast = materialize_forecast(bundle(rows), "run24")
    assert forecast.horizon_hours == 24
    assert len(calls) == 1 and len(calls[0]["predictions"]) == 24


def test_native_materialization_accepts_tabular_order_but_preserves_numbered_steps():
    rows = native_run().iloc[::-1]
    forecast = materialize_forecast(bundle(rows), "run24")
    assert forecast.predictions.predicted_load.tolist() == list(np.arange(24) + 1000)


@pytest.mark.parametrize("kind", ["missing", "duplicated", "invalid_step", "invalid_timestamp", "nonfinite_load", "naive_target"])
def test_native_materialization_refuses_invalid_selected_run(kind):
    rows = native_run()
    if kind == "missing":
        rows = rows.iloc[:-1]
    elif kind == "duplicated":
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    elif kind == "invalid_step":
        rows.loc[0, "forecast_step"] = 2
    elif kind == "invalid_timestamp":
        rows.loc[0, "timestamp"] = rows.loc[1, "timestamp"]
    elif kind == "nonfinite_load":
        rows.loc[0, "predicted_load"] = float("inf")
    else:
        rows["timestamp"] = rows["timestamp"].dt.tz_localize(None)
    with pytest.raises((ValueError, ForecastValidationError)):
        materialize_forecast(bundle(rows), "run24")


def test_native_materialization_refuses_conflicting_original_text():
    rows = native_run()
    rows.loc[0, "issued_at_original"] = "2025-01-09 00:00:00"
    with pytest.raises(ValueError, match="temporais textuais conflitantes"):
        materialize_forecast(bundle(rows), "run24")


def test_unknown_id_and_empty_summary_bundle_never_fabricate_forecast():
    assert forecast_catalog(bundle()).empty
    assert forecast_catalog(SimpleNamespace(forecasts=[])).columns.tolist() == CATALOG_COLUMNS
    with pytest.raises(ValueError, match="desconhecida"):
        materialize_forecast(bundle(native_run()), "missing")
    with pytest.raises(ValueError, match="desconhecida"):
        materialize_forecast(bundle(), "summary-only")


def test_json_catalog_and_materialization_keep_content_identifiers_compatible():
    forecast = json_run()
    identifier = forecast_identifier(forecast)
    catalog = forecast_catalog(bundle(forecasts=[forecast]))
    assert catalog.forecast_id.tolist() == [identifier]
    assert materialize_forecast(bundle(forecasts=[forecast]), identifier) is forecast


def test_json_copies_of_identical_content_are_deduplicated_deterministically():
    forecast = json_run()
    copy = replace(forecast, source_path=Path("a_copia.json"))
    catalog = forecast_catalog(bundle(forecasts=[forecast, copy]))
    assert len(catalog) == 1 and catalog.source_file.iloc[0] == "a_copia.json"
    assert materialize_forecast(bundle(forecasts=[forecast, copy]), catalog.forecast_id.iloc[0]) is copy


def test_native_catalog_takes_priority_over_json_list_in_same_bundle():
    native = native_run()
    existing = json_run()
    catalog = forecast_catalog(bundle(native, [existing]))
    assert catalog.forecast_id.tolist() == ["run24"]
    with pytest.raises(ValueError, match="desconhecida"):
        materialize_forecast(bundle(native, [existing]), forecast_identifier(existing))
