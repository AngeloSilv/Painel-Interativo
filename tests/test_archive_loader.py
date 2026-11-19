"""Integração tabular do arquivo fornecido, usando apenas ZIPs pequenos temporários."""

from datetime import timedelta
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from services.archive_loader import (
    HISTORY_MEMBER, PREDICTION_MEMBERS, SUMMARY_MEMBERS, load_archive_bundle,
)
from services.data_loader import load_data, load_calendar


def _frames() -> dict[str, pd.DataFrame]:
    times = pd.date_range("2025-06-23", periods=12 * 24, freq="h")
    values = pd.Series(1000.0 + np.arange(len(times)), index=times)
    special = times.date == pd.Timestamp("2025-07-02").date()
    history = pd.DataFrame({
        "instante": times.strftime("%Y-%m-%d %H:%M:%S"), "carga_mwmed": values.to_numpy(),
        "temp_a621_c": 25.0, "temp_a601_c": 24.0, "temp_rj_c": 25.0,
        "fonte_temp": "A621_observada", "fim_de_semana": (times.dayofweek >= 5).astype(int),
        "eh_feriado": 0, "eh_ponto_facultativo": special.astype(int),
        "esfera_feriado": "", "nome_evento_calendario": np.where(special, "Evento de teste", ""),
    })
    history.loc[0, "temp_a621_c"] = np.nan
    history.loc[0, "temp_rj_c"] = 24.0
    history.loc[0, "fonte_temp"] = "A601_fallback_t"
    history.loc[1, "temp_a601_c"] = np.nan
    frames = {HISTORY_MEMBER: history}
    issue = pd.Timestamp("2025-07-01 00:00:00")
    for horizon, member in PREDICTION_MEMBERS.items():
        target = pd.date_range(issue + timedelta(hours=1), periods=horizon, freq="h")
        frames[member] = pd.DataFrame({
            "origem_t": issue.strftime("%Y-%m-%d %H:%M:%S"), "passo_h": np.arange(1, horizon + 1),
            "instante_alvo": target.strftime("%Y-%m-%d %H:%M:%S"), "y_real": target.map(values),
            "y_pred": target.map(values) + 5, "y_baseline_semanal": (target - timedelta(hours=168)).map(values),
            "y_baseline_persistencia": values.loc[issue], "modelo": f"integrado_{horizon}h", "etapa": "teste_2025",
        })
    for member in SUMMARY_MEMBERS:
        frames[member] = pd.DataFrame({"etapa": ["Validação 1S2025"], "modelo": ["Modelo fornecido"], "mae_mwmed": [10.25]})
    return frames


def make_archive(tmp_path: Path, transform=None, metadata=None) -> tuple[Path, Path]:
    frames = _frames()
    if transform:
        transform(frames)
    imported = tmp_path / "imported"
    imported.mkdir()
    archive_path = imported / "fixture.zip"
    with ZipFile(archive_path, "w") as archive:
        for name, frame in frames.items():
            archive.writestr(name, frame.to_csv(index=False))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    config = {
        "archive": "fixture.zip", "sha256": digest, "region": "Município do Rio de Janeiro",
        "unit": "MWmed", "timezone": "America/Sao_Paulo",
        "region_source": "informado pelo grupo", "timezone_source": "informado pelo grupo",
    }
    if metadata:
        config.update(metadata)
    config_path = imported / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return config_path, archive_path


def test_native_archive_loads_without_materializing_forecast_objects(tmp_path):
    config, archive = make_archive(tmp_path)
    original = archive.read_bytes()
    bundle = load_archive_bundle(config)
    assert bundle.mode == "provided"
    assert bundle.issues == []
    assert bundle.forecasts == []
    assert len(bundle.observations) == 288
    assert len(bundle.weather) == 288 * 2 - 2
    assert set(bundle.weather["station"]) == {"A621", "A601"}
    assert len(bundle.prediction_rows) == 72
    assert bundle.prediction_rows["forecast_id"].nunique() == 2
    assert set(bundle.prediction_rows["model_version"]) == {"não informada"}
    assert set(bundle.prediction_rows["dataset_split"]) == {"test_2025"}
    assert set(bundle.prediction_rows["unit"]) == {"MWmed"}
    assert set(bundle.prediction_rows["region"]) == {"Município do Rio de Janeiro"}
    assert str(bundle.prediction_rows["timestamp"].dt.tz) == "UTC"
    assert bundle.prediction_rows["issued_at"].iloc[0] == pd.Timestamp("2025-07-01T03:00:00Z")
    assert bundle.prediction_rows["issued_at_original"].iloc[0] == "2025-07-01 00:00:00"
    assert archive.read_bytes() == original
    assert set(path.name for path in config.parent.iterdir()) == {"config.json", "fixture.zip"}


def test_load_data_provided_delegates_and_keeps_legacy_sources_empty(tmp_path):
    make_archive(tmp_path)
    assert len(load_data("provided", tmp_path).prediction_rows) == 72
    assert load_data("real", tmp_path).prediction_rows.empty
    assert load_data("demo", tmp_path).prediction_rows.empty


def test_provenance_preserves_group_declaration_and_temperature_composition(tmp_path):
    config, _ = make_archive(tmp_path)
    provenance = load_archive_bundle(config).provenance
    assert provenance["region_source"] == "informado pelo grupo"
    assert provenance["timezone_source"] == "informado pelo grupo"
    assert provenance["temperature_composite_used_as_observed_station"] is False
    assert provenance["temperature_composition_source_counts"] == {"A621_observada": 287, "A601_fallback_t": 1}
    assert provenance["weather_missing_by_station"] == {"A621": 1, "A601": 1}


def test_unknown_optional_day_scope_remains_explicitly_unknown(tmp_path):
    config, _ = make_archive(tmp_path)
    bundle = load_archive_bundle(config)
    special = bundle.calendar[bundle.calendar["day_type"] == "ponto_facultativo"]
    assert len(special) == 1
    assert special.iloc[0]["holiday_scope"] == "nao_informado"
    assert special.iloc[0]["event_name"] == "Evento de teste"
    assert bundle.provenance["calendar_unknown_scope_days"] == 1
    # O contrato CSV nativo aceita a mesma informação desconhecida explícita.
    native = tmp_path / "calendar.csv"
    bundle.calendar.to_csv(native, index=False)
    assert load_calendar(native).equals(bundle.calendar)


def test_reported_summaries_are_raw_and_keep_source_files(tmp_path):
    config, _ = make_archive(tmp_path)
    reported = load_archive_bundle(config).reported_metrics
    assert len(reported) == 3
    assert set(reported["source_file"]) == set(SUMMARY_MEMBERS)
    assert set(reported["mae_mwmed"]) == {10.25}
    assert "observed_load" not in reported


def test_missing_configuration_is_empty_without_fallback(tmp_path):
    bundle = load_archive_bundle(tmp_path / "missing.json")
    assert bundle.observations.empty and bundle.prediction_rows.empty
    assert len(bundle.issues) == 1
    assert "Configuração" in bundle.issues[0]


@pytest.mark.parametrize("field", ["region", "timezone", "region_source", "timezone_source", "unit", "archive", "sha256"])
def test_required_metadata_is_not_inferred(tmp_path, field):
    config, _ = make_archive(tmp_path)
    content = json.loads(config.read_text(encoding="utf-8"))
    del content[field]
    config.write_text(json.dumps(content), encoding="utf-8")
    bundle = load_archive_bundle(config)
    assert bundle.observations.empty
    assert "declarar" in bundle.issues[0]


@pytest.mark.parametrize("metadata, message", [
    ({"unit": "MW"}, "MWmed"), ({"unit": "kW"}, "MWmed"),
    ({"timezone": "Inventado/Local"}, "Fuso"),
    ({"sha256": "errado"}, "sha256"), ({"sha256": "0" * 64}, "SHA-256"),
    ({"archive": "../fixture.zip"}, "nome relativo"), ({"region": " "}, "não vazios"),
])
def test_invalid_metadata_blocks_archive(tmp_path, metadata, message):
    config, _ = make_archive(tmp_path, metadata=metadata)
    bundle = load_archive_bundle(config)
    assert bundle.observations.empty and bundle.prediction_rows.empty
    assert message in bundle.issues[0]


@pytest.mark.parametrize("column", ["y_real", "y_baseline_persistencia", "y_baseline_semanal"])
def test_values_conflicting_with_history_block_entire_prediction_file(tmp_path, column):
    def corrupt(frames):
        frames[PREDICTION_MEMBERS[24]].loc[0, column] += 1
    config, _ = make_archive(tmp_path, corrupt)
    bundle = load_archive_bundle(config)
    assert set(bundle.prediction_rows["horizon_hours"]) == {48}
    assert len(bundle.issues) == 1
    assert column in bundle.issues[0] and "diverge" in bundle.issues[0]


def test_small_floating_point_rounding_difference_is_accepted(tmp_path):
    def rounding(frames):
        frames[PREDICTION_MEMBERS[24]]["y_real"] += 1e-8
    config, _ = make_archive(tmp_path, rounding)
    assert load_archive_bundle(config).issues == []


@pytest.mark.parametrize("case, message", [("incomplete", "exatamente"), ("duplicate", "duplicados"), ("target", "alvos"), ("split", "etapa"), ("nonfinite", "finitos")])
def test_invalid_forecast_sequences_are_not_repaired(tmp_path, case, message):
    def corrupt(frames):
        member = PREDICTION_MEMBERS[24]
        frame = frames[member]
        if case == "incomplete": frames[member] = frame.iloc[:-1]
        if case == "duplicate": frames[member] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        if case == "target": frame.loc[0, "instante_alvo"] = "2025-07-01 02:00:00"
        if case == "split": frame["etapa"] = "ano_2025_sem_split"
        if case == "nonfinite": frame.loc[0, "y_pred"] = np.inf
    config, _ = make_archive(tmp_path, corrupt)
    bundle = load_archive_bundle(config)
    assert set(bundle.prediction_rows["horizon_hours"]) == {48}
    assert any(message in issue for issue in bundle.issues)


def test_historical_duplicates_block_all_archive_data(tmp_path):
    def corrupt(frames):
        frames[HISTORY_MEMBER] = pd.concat([frames[HISTORY_MEMBER], frames[HISTORY_MEMBER].iloc[[0]]], ignore_index=True)
    config, _ = make_archive(tmp_path, corrupt)
    bundle = load_archive_bundle(config)
    assert bundle.observations.empty and bundle.prediction_rows.empty
    assert "duplicados" in bundle.issues[0]


def test_calendar_priority_holiday_over_optional_and_weekend(tmp_path):
    def flags(frames):
        history = frames[HISTORY_MEMBER]
        day = history["instante"].str.startswith("2025-06-28")
        history.loc[day, "eh_feriado"] = 1
        history.loc[day, "eh_ponto_facultativo"] = 1
        history.loc[day, "esfera_feriado"] = "municipal_rio"
        history.loc[day, "nome_evento_calendario"] = "Evento declarado"
    config, _ = make_archive(tmp_path, flags)
    calendar = load_archive_bundle(config).calendar
    row = calendar[calendar["date"] == pd.Timestamp("2025-06-28").date()].iloc[0]
    assert row["day_type"] == "feriado" and row["holiday_scope"] == "municipal"


def test_unknown_baseline_history_is_unavailable_instead_of_unverified_value(tmp_path):
    def truncate(frames):
        frames[HISTORY_MEMBER] = frames[HISTORY_MEMBER].iloc[7 * 24:].reset_index(drop=True)
    config, _ = make_archive(tmp_path, truncate)
    bundle = load_archive_bundle(config)
    assert len(bundle.prediction_rows) == 72
    assert bundle.prediction_rows["seasonal_load"].isna().all()
    assert len(bundle.issues) == 2
    assert all("sem histórico anterior verificável" in issue for issue in bundle.issues)
