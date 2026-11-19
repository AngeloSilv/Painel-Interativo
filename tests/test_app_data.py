"""Source-aware caching without private files, cross-mode leaks or shared copies."""

import pandas as pd
import pytest

from services import app_data, data_loader
from services.data_loader import DataBundle
from services.geography_service import TARGET_REGION


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    app_data._load_standard.clear()
    app_data._load_imported.clear()
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)
    yield tmp_path
    app_data._load_standard.clear()
    app_data._load_imported.clear()


def fake_bundle(mode, value=100.0):
    return DataBundle(mode, [], pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01T00:00:00Z"]),
        "observed_load": [value], "region": [TARGET_REGION], "unit": ["MWmed"],
    }), [])


def counting_loader(monkeypatch):
    calls = []

    def load(mode):
        calls.append(mode)
        return fake_bundle(mode, float(len(calls)))

    monkeypatch.setattr(data_loader, "load_data", load)
    return calls


def test_repeated_reads_are_cached_but_demo_and_real_are_separate(monkeypatch):
    calls = counting_loader(monkeypatch)
    demo, _ = app_data.load_app_context("demo")
    real, _ = app_data.load_app_context("real")
    again, _ = app_data.load_app_context("demo")
    assert calls == ["demo", "real"]
    assert demo.mode == again.mode == "demo"
    assert real.mode == "real"
    assert demo.observations.observed_load.iloc[0] == again.observations.observed_load.iloc[0] == 1
    assert real.observations.observed_load.iloc[0] == 2


@pytest.mark.parametrize("mode,directory,name", [
    ("demo", "demo", "forecast_24h.json"),
    ("demo", "demo", "historical.csv"),
    ("demo", "demo", "weather.csv"),
    ("demo", "demo", "calendar.csv"),
    ("real", "predictions", "model_24h.json"),
    ("real", "historical", "load.csv"),
    ("real", "weather", "station.csv"),
    ("real", "calendar", "calendar.csv"),
])
def test_relevant_input_changes_invalidate_the_cache(isolated_cache, monkeypatch, mode, directory, name):
    path = isolated_cache / directory / name
    path.parent.mkdir(parents=True)
    path.write_text("initial", encoding="utf-8")
    calls = counting_loader(monkeypatch)
    app_data.load_app_context(mode)
    app_data.load_app_context(mode)
    assert calls == [mode]
    path.write_text("updated content with a different size", encoding="utf-8")
    refreshed, _ = app_data.load_app_context(mode)
    assert calls == [mode, mode]
    assert refreshed.observations.observed_load.iloc[0] == 2


def test_irrelevant_files_do_not_invalidate_demo_cache(isolated_cache, monkeypatch):
    directory = isolated_cache / "demo"
    directory.mkdir()
    (directory / "forecast.json").write_text("forecast", encoding="utf-8")
    calls = counting_loader(monkeypatch)
    app_data.load_app_context("demo")
    (directory / "readme.txt").write_text("notes", encoding="utf-8")
    (directory / "unused.csv").write_text("not a loaded demo table", encoding="utf-8")
    (directory / "folder.json").mkdir()
    app_data.load_app_context("demo")
    assert calls == ["demo"]


def test_addition_and_removal_of_inputs_invalidate_cache(isolated_cache, monkeypatch):
    directory = isolated_cache / "predictions"
    directory.mkdir()
    calls = counting_loader(monkeypatch)
    app_data.load_app_context("real")
    path = directory / "new_model.JSON"
    path.write_text("new input", encoding="utf-8")
    app_data.load_app_context("real")
    path.unlink()
    app_data.load_app_context("real")
    # The original empty source signature is reused after deletion. Its result
    # is valid for the current empty state, rather than the newer input state.
    assert calls == ["real", "real"]


def test_cached_data_returns_independent_copies(monkeypatch):
    calls = counting_loader(monkeypatch)
    first, _ = app_data.load_app_context("demo")
    first.observations.loc[0, "observed_load"] = 99999.0
    second, _ = app_data.load_app_context("demo")
    assert second.observations.observed_load.iloc[0] == 1.0
    assert calls == ["demo"]


def test_overridden_loader_has_an_independent_cache_key(monkeypatch):
    monkeypatch.setattr(data_loader, "load_data", lambda mode: fake_bundle(mode, 10.0))
    first, _ = app_data.load_app_context("demo")
    monkeypatch.setattr(data_loader, "load_data", lambda mode: fake_bundle(mode, 20.0))
    second, _ = app_data.load_app_context("demo")
    assert first.observations.observed_load.iloc[0] == 10.0
    assert second.observations.observed_load.iloc[0] == 20.0


def test_real_mode_with_missing_files_never_falls_back_to_demo(isolated_cache, monkeypatch):
    original_loader = data_loader.load_data
    (isolated_cache / "demo").mkdir()
    (isolated_cache / "demo" / "irrelevant.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(data_loader, "load_data", lambda mode: original_loader(mode, isolated_cache))
    bundle, catalog = app_data.load_app_context("real")
    assert bundle.mode == "real"
    assert bundle.forecasts == []
    assert bundle.observations.empty
    assert catalog.empty
    assert bundle.issues == []


def test_cache_isolated_when_data_directory_changes(isolated_cache, monkeypatch):
    calls = counting_loader(monkeypatch)
    app_data.load_app_context("demo")
    other = isolated_cache / "another_project"
    other.mkdir()
    monkeypatch.setattr(data_loader, "DATA_DIR", other)
    app_data.load_app_context("demo")
    assert calls == ["demo", "demo"]


def test_invalid_source_mode_has_actionable_error():
    with pytest.raises(data_loader.DataLoadError, match="Modo de dados inválido"):
        app_data.load_app_context("unknown")
