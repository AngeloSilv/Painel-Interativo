"""Carregamento, isolamento das fontes e reprodutibilidade da demonstração."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from services.data_loader import (
    DataBundle, DataLoadError, load_calendar, load_data, load_forecast, load_observations, load_weather,
)
from services.demo_data import DEMO_REGION, DEMO_RUNS, HISTORY_HOURS, build_demo_data


@pytest.fixture
def demo_dir(tmp_path):
    build_demo_data(tmp_path / "demo")
    return tmp_path


def write_history(path: Path, rows: str) -> Path:
    path.write_text("timestamp,observed_load,region,unit\n" + rows, encoding="utf-8")
    return path


def write_weather(path: Path, rows: str) -> Path:
    path.write_text("timestamp,temperature_c,station,region\n" + rows, encoding="utf-8")
    return path


def write_calendar(path: Path, rows: str) -> Path:
    path.write_text("date,region,day_type,holiday_scope,event_name\n" + rows, encoding="utf-8")
    return path


def test_missing_forecast_raises_actionable_error(tmp_path):
    with pytest.raises(DataLoadError, match="não encontrado"):
        load_forecast(tmp_path / "missing.json")


def test_missing_history_raises_actionable_error(tmp_path):
    with pytest.raises(DataLoadError, match="não encontrado"):
        load_observations(tmp_path / "missing.csv")


def test_malformed_json_is_reported(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(DataLoadError, match="JSON"):
        load_forecast(path)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"unit":"MW","unit":"kW"}', encoding="utf-8")
    with pytest.raises(DataLoadError, match="duplicada"):
        load_forecast(path)


def test_invalid_json_contract_is_reported(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"model_name": "incompleto"}), encoding="utf-8")
    with pytest.raises(DataLoadError, match="ausentes"):
        load_forecast(path)


def test_load_demo_has_both_horizons_and_preserves_source(demo_dir):
    bundle = load_data("demo", demo_dir)
    assert bundle.mode == "demo"
    assert len(bundle.forecasts) == len(DEMO_RUNS) * 2
    assert {forecast.horizon_hours for forecast in bundle.forecasts} == {24, 48}
    assert len(bundle.observations) == HISTORY_HOURS
    assert bundle.issues == []
    assert all(forecast.source_path.parent == demo_dir / "demo" for forecast in bundle.forecasts)
    assert str(bundle.observations["timestamp"].dt.tz) == "UTC"
    assert set(bundle.observations["unit"]) == {"MWmed"}
    assert set(bundle.observations["region"]) == {DEMO_REGION}
    assert len(bundle.weather) == HISTORY_HOURS * 2
    assert bundle.weather["station"].nunique() == 2
    assert len(bundle.calendar) == 366 + 365 + 365
    assert {forecast.dataset_split for forecast in bundle.forecasts} == {"validation_2025", "test_2025", "forecast_2026"}
    events = bundle.calendar[bundle.calendar["day_type"].isin(["feriado", "ponto_facultativo"])]
    assert events["event_name"].str.startswith("Evento sintético").all()


def test_future_demo_has_no_observations_and_backtest_has_all(demo_dir):
    bundle = load_data("demo", demo_dir)
    observed_instants = set(bundle.observations["timestamp"])
    for forecast in bundle.forecasts:
        targets = set(forecast.predictions["timestamp"])
        if "future" in forecast.source_path.name:
            assert targets.isdisjoint(observed_instants)
            assert forecast.issued_at == bundle.observations["timestamp"].max()
        else:
            assert targets.issubset(observed_instants)


def test_real_mode_never_falls_back_to_demo(demo_dir):
    bundle = load_data("real", demo_dir)
    assert bundle.forecasts == []
    assert bundle.observations.empty
    assert bundle.weather.empty
    assert bundle.calendar.empty
    assert bundle.issues == []


def test_empty_directory_is_a_valid_empty_state(tmp_path):
    bundle = load_data("demo", tmp_path)
    assert bundle.forecasts == []
    assert bundle.observations.empty
    assert list(bundle.observations.columns) == ["timestamp", "observed_load", "region", "unit"]


def test_invalid_data_mode_is_rejected(tmp_path):
    with pytest.raises(DataLoadError, match="Modo"):
        load_data("mixed", tmp_path)


def test_directory_access_error_is_reported(tmp_path, monkeypatch):
    original_iterdir = Path.iterdir

    def guarded_iterdir(path):
        if path == tmp_path / "predictions":
            raise PermissionError("Acesso negado para teste")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    bundle = load_data("real", tmp_path)
    assert bundle.forecasts == []
    assert len(bundle.issues) == 1
    assert "predictions" in bundle.issues[0]
    assert "Acesso negado" in bundle.issues[0]


def test_file_instead_of_data_directory_is_reported(tmp_path):
    (tmp_path / "predictions").write_text("arquivo em vez de pasta", encoding="utf-8")
    bundle = load_data("real", tmp_path)
    assert bundle.forecasts == []
    assert len(bundle.issues) == 1
    assert "predictions" in bundle.issues[0]


def test_unrelated_files_and_subdirectories_are_ignored(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "readme.txt").write_text("instruções", encoding="utf-8")
    (predictions / "nao_e_arquivo.json").mkdir()
    bundle = load_data("real", tmp_path)
    assert bundle.forecasts == []
    assert bundle.issues == []


def test_real_files_load_without_renaming_contract(demo_dir):
    predictions = demo_dir / "predictions"
    historical = demo_dir / "historical"
    predictions.mkdir()
    historical.mkdir()
    source = demo_dir / "demo" / "forecast_future_24h.json"
    (predictions / "colega_modelo_24h.json").write_bytes(source.read_bytes())
    history_source = demo_dir / "demo" / "historical.csv"
    (historical / "carga_observada.csv").write_bytes(history_source.read_bytes())
    bundle = load_data("real", demo_dir)
    assert len(bundle.forecasts) == 1
    assert len(bundle.observations) == HISTORY_HOURS
    assert bundle.forecasts[0].source_path.name == "colega_modelo_24h.json"


def test_invalid_files_are_reported_without_hiding_valid_files(demo_dir):
    (demo_dir / "demo" / "broken.json").write_text("{}", encoding="utf-8")
    bundle = load_data("demo", demo_dir)
    assert len(bundle.forecasts) == len(DEMO_RUNS) * 2
    assert len(bundle.issues) == 1
    assert "broken.json" in bundle.issues[0]


def test_history_missing_columns_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,observed_load\n2026-01-01T00:00:00Z,10\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="colunas"):
        load_observations(path)


@pytest.mark.parametrize("row, message", [
    ("2026-01-01T00:00:00,10,Região,MW\n", "timestamp"),
    ("2026-01-01T00:00:00Z,NaN,Região,MW\n", "observed_load"),
    ("2026-01-01T00:00:00Z,inf,Região,MW\n", "finita"),
    ("2026-01-01T00:00:00Z,-1,Região,MW\n", "não negativa"),
    ("2026-01-01T00:00:00Z,10,Região,kW\n", "unidade"),
    ("2026-01-01T00:00:00Z,10, ,MW\n", "region"),
])
def test_invalid_history_values_are_rejected(tmp_path, row, message):
    path = write_history(tmp_path / "bad.csv", row)
    with pytest.raises(DataLoadError, match=message):
        load_observations(path)


def test_history_duplicates_after_normalization_are_rejected(tmp_path):
    path = write_history(tmp_path / "duplicate.csv", (
        "2026-01-01T00:00:00-03:00,10,Região,MW\n"
        "2026-01-01T03:00:00Z,11,Região,MW\n"
    ))
    with pytest.raises(DataLoadError, match="duplicadas"):
        load_observations(path)


def test_gaps_remain_gaps_and_regions_are_not_merged(tmp_path):
    path = write_history(tmp_path / "valid.csv", (
        "2026-01-01T00:00:00Z,10,A,MW\n"
        "2026-01-01T02:00:00Z,12,A,MW\n"
        "2026-01-01T00:00:00Z,20,B,MW\n"
    ))
    frame = load_observations(path)
    assert len(frame) == 3
    assert frame.loc[frame["region"] == "A", "timestamp"].diff().iloc[1].total_seconds() == 7200


@pytest.mark.parametrize("second_timestamp", [
    "2026-01-01T01:30:00Z", "2026-01-01T00:30:00Z", "2026-01-01T01:00:00.000001Z",
    "2026-01-01T01:00:00.000000001Z",
])
def test_irregular_hourly_grid_is_rejected(tmp_path, second_timestamp):
    path = write_history(tmp_path / "irregular.csv", (
        "2026-01-01T00:00:00Z,10,A,MW\n"
        f"{second_timestamp},12,A,MW\n"
    ))
    with pytest.raises(DataLoadError, match="grade horária irregular"):
        load_observations(path)


def test_regions_may_have_independent_hourly_grid_offsets(tmp_path):
    path = write_history(tmp_path / "valid_offsets.csv", (
        "2026-01-01T00:00:00Z,10,A,MW\n"
        "2026-01-01T01:00:00Z,11,A,MW\n"
        "2026-01-01T00:30:00Z,20,B,MW\n"
        "2026-01-01T02:30:00Z,22,B,MW\n"
    ))
    frame = load_observations(path)
    assert len(frame) == 4
    assert set(frame.loc[frame["region"] == "B", "timestamp"].dt.minute) == {30}


def test_cross_file_singleton_observations_with_irregular_grid_are_rejected(tmp_path):
    historical = tmp_path / "historical"
    historical.mkdir()
    write_history(historical / "first.csv", "2026-01-01T00:00:00Z,10,A,MW\n")
    write_history(historical / "second.csv", "2026-01-01T01:30:00Z,11,A,MW\n")
    bundle = load_data("real", tmp_path)
    assert bundle.observations.empty
    assert len(bundle.issues) == 1
    assert "grade horária irregular" in bundle.issues[0]
    assert "combinado" in bundle.issues[0]


def test_cross_file_singletons_accept_hourly_gaps_and_independent_regions(tmp_path):
    historical = tmp_path / "historical"
    historical.mkdir()
    write_history(historical / "a_first.csv", "2026-01-01T00:00:00Z,10,A,MW\n")
    write_history(historical / "a_later.csv", "2026-01-01T02:00:00Z,12,A,MW\n")
    write_history(historical / "b_first.csv", "2026-01-01T00:30:00Z,20,B,MW\n")
    write_history(historical / "b_later.csv", "2026-01-01T01:30:00Z,21,B,MW\n")
    bundle = load_data("real", tmp_path)
    assert len(bundle.observations) == 4
    assert bundle.issues == []


def test_cross_file_duplicate_history_is_rejected(tmp_path):
    historical = tmp_path / "historical"
    historical.mkdir()
    row = "2026-01-01T00:00:00Z,10,Região,MW\n"
    write_history(historical / "first.csv", row)
    write_history(historical / "second.csv", row)
    bundle = load_data("real", tmp_path)
    assert bundle.observations.empty
    assert len(bundle.issues) == 1
    assert "duplicadas" in bundle.issues[0]


def test_same_region_different_units_remain_independent_series(tmp_path):
    path = write_history(tmp_path / "units.csv", (
        "2026-01-01T00:00:00Z,10,A,MW\n"
        "2026-01-01T01:00:00Z,11,A,MW\n"
        "2026-01-01T00:00:00Z,20,A,MWmed\n"
        "2026-01-01T02:00:00Z,22,A,MWmed\n"
    ))
    frame = load_observations(path)
    assert len(frame) == 4
    assert set(frame["unit"]) == {"MW", "MWmed"}
    assert frame.loc[frame["unit"] == "MWmed", "observed_load"].tolist() == [20, 22]


def test_combined_history_keeps_units_and_grids_separate(tmp_path):
    historical = tmp_path / "historical"
    historical.mkdir()
    write_history(historical / "mw.csv", "2026-01-01T00:00:00Z,10,A,MW\n")
    write_history(historical / "mwmed.csv", "2026-01-01T00:30:00Z,20,A,MWmed\n")
    bundle = load_data("real", tmp_path)
    assert len(bundle.observations) == 2
    assert bundle.issues == []


def test_bundle_four_argument_constructor_keeps_optional_sources_empty():
    first = DataBundle("real", [], pd.DataFrame(), [])
    second = DataBundle("real", [], pd.DataFrame(), [])
    assert first.weather.empty and first.calendar.empty
    assert first.weather is not second.weather
    assert first.calendar is not second.calendar


def test_missing_optional_sources_have_actionable_errors(tmp_path):
    with pytest.raises(DataLoadError, match="não encontrado"):
        load_weather(tmp_path / "missing.csv")
    with pytest.raises(DataLoadError, match="não encontrado"):
        load_calendar(tmp_path / "missing.csv")


def test_weather_preserves_stations_and_normalizes_utc(tmp_path):
    path = write_weather(tmp_path / "weather.csv", (
        "2026-01-01T00:00:00-03:00,-60,Estação A,Rio\n"
        "2026-01-01T03:00:00Z,60,Estação B,Rio\n"
    ))
    weather = load_weather(path)
    assert len(weather) == 2
    assert weather["timestamp"].nunique() == 1
    assert weather["station"].nunique() == 2
    assert str(weather["timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize("row, message", [
    ("2026-01-01T00:00:00,25,A,Rio\n", "timestamp"),
    ("2026-01-01T00:00:00Z,inf,A,Rio\n", "finita"),
    ("2026-01-01T00:00:00Z,61,A,Rio\n", "entre -60 e 60"),
    ("2026-01-01T00:00:00Z,-61,A,Rio\n", "entre -60 e 60"),
    ("2026-01-01T00:00:00Z,quente,A,Rio\n", "temperature_c"),
    ("2026-01-01T00:00:00Z,25, ,Rio\n", "station"),
    ("2026-01-01T00:00:00Z,25,A, \n", "region"),
])
def test_invalid_weather_is_rejected(tmp_path, row, message):
    with pytest.raises(DataLoadError, match=message):
        load_weather(write_weather(tmp_path / "bad.csv", row))


def test_weather_duplicate_instants_with_different_offsets_are_rejected(tmp_path):
    path = write_weather(tmp_path / "duplicate.csv", (
        "2026-01-01T00:00:00-03:00,25,A,Rio\n"
        "2026-01-01T03:00:00Z,26,A,Rio\n"
    ))
    with pytest.raises(DataLoadError, match="duplicadas"):
        load_weather(path)


def test_calendar_preserves_declared_local_dates_and_categories(tmp_path):
    path = write_calendar(tmp_path / "calendar.csv", (
        "2026-01-01,Rio,dia_util,nenhum,\n"
        "2026-01-02,Rio,feriado,municipal,Evento declarado\n"
        "2026-01-03,Rio,ponto_facultativo,estadual,Outro evento\n"
    ))
    calendar = load_calendar(path)
    assert calendar.iloc[0]["date"] == date(2026, 1, 1)
    assert calendar.iloc[0]["day_type"] == "dia_util"
    assert calendar.iloc[0]["event_name"] == ""
    assert calendar.iloc[1]["holiday_scope"] == "municipal"


@pytest.mark.parametrize("row, message", [
    ("2026-02-30,Rio,dia_util,nenhum,\n", "date"),
    ("01/01/2026,Rio,dia_util,nenhum,\n", "date"),
    ("2026-01-01T00:00:00Z,Rio,dia_util,nenhum,\n", "date"),
    ("2026-01-01, ,dia_util,nenhum,\n", "region"),
    ("2026-01-01,Rio,normal,nenhum,\n", "day_type"),
    ("2026-01-01,Rio,dia_util,global,\n", "holiday_scope"),
    ("2026-01-01,Rio,feriado,municipal,\n", "event_name"),
    ("2026-01-01,Rio,ponto_facultativo,municipal,\n", "event_name"),
    ("2026-01-01,Rio,feriado,nenhum,Evento\n", "holiday_scope"),
    ("2026-01-01,Rio,dia_util,nacional,\n", "holiday_scope"),
])
def test_invalid_calendar_is_rejected(tmp_path, row, message):
    with pytest.raises(DataLoadError, match=message):
        load_calendar(write_calendar(tmp_path / "bad.csv", row))


def test_incomplete_calendar_is_not_filled_or_inferred(tmp_path):
    path = write_calendar(tmp_path / "gap.csv", (
        "2026-01-01,Rio,dia_util,nenhum,\n"
        "2026-01-03,Rio,fim_de_semana,nenhum,\n"
    ))
    with pytest.raises(DataLoadError, match="incompleto"):
        load_calendar(path)


def test_duplicate_calendar_dates_are_rejected(tmp_path):
    path = write_calendar(tmp_path / "duplicate.csv", (
        "2026-01-01,Rio,dia_util,nenhum,\n"
        "2026-01-01,Rio,feriado,municipal,Evento\n"
    ))
    with pytest.raises(DataLoadError, match="duplicadas"):
        load_calendar(path)


def test_real_optional_sources_are_loaded_without_demo_fallback(tmp_path):
    (tmp_path / "weather").mkdir()
    (tmp_path / "calendar").mkdir()
    write_weather(tmp_path / "weather" / "estacoes.csv", "2026-01-01T03:00:00Z,25,A,Rio\n")
    write_calendar(tmp_path / "calendar" / "dias.csv", "2026-01-01,Rio,dia_util,nenhum,\n")
    bundle = load_data("real", tmp_path)
    assert bundle.forecasts == []
    assert bundle.observations.empty
    assert len(bundle.weather) == len(bundle.calendar) == 1
    assert bundle.issues == []


def test_cross_file_optional_source_duplicates_are_reported(tmp_path):
    (tmp_path / "weather").mkdir()
    (tmp_path / "calendar").mkdir()
    for name in ("a.csv", "b.csv"):
        write_weather(tmp_path / "weather" / name, "2026-01-01T03:00:00Z,25,A,Rio\n")
        write_calendar(tmp_path / "calendar" / name, "2026-01-01,Rio,dia_util,nenhum,\n")
    bundle = load_data("real", tmp_path)
    assert bundle.weather.empty
    assert bundle.calendar.empty
    assert len(bundle.issues) == 2
    assert all("duplicadas" in issue for issue in bundle.issues)


def test_demo_generation_is_byte_reproducible_and_preserves_existing(tmp_path):
    first = build_demo_data(tmp_path / "first")
    second = build_demo_data(tmp_path / "second")
    assert {path.name: path.read_bytes() for path in first} == {path.name: path.read_bytes() for path in second}
    with pytest.raises(FileExistsError, match="já existem"):
        build_demo_data(tmp_path / "first")
    previous = first[0].read_bytes()
    build_demo_data(tmp_path / "first", overwrite=True)
    assert first[0].read_bytes() == previous
