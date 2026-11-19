"""Adaptador vetorizado do ZIP fornecido pelo grupo, sem extrair seus membros.

O recorte geográfico e o fuso vêm de metadados explícitos, atribuídos ao grupo.
Os arquivos originais são somente lidos; previsões tabulares não são convertidas
em milhares de objetos Forecast ou arquivos JSON.
"""

import hashlib
from datetime import timedelta
import json
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from services.data_loader import (
    DataBundle, DataLoadError, empty_calendar, empty_observations, empty_weather,
)


HISTORY_MEMBER = "painel/serie_horaria_integrada_2024_2026.csv"
PREDICTION_MEMBERS = {
    24: "painel/previsoes_24h_2025_2026.csv",
    48: "painel/previsoes_48h_2025_2026.csv",
}
SUMMARY_MEMBERS = (
    "painel/resumo_metricas_24h.csv", "painel/resumo_metricas_48h.csv", "painel/resumo_metricas.csv",
)
HISTORY_REQUIRED = {
    "instante", "carga_mwmed", "temp_a621_c", "temp_a601_c", "fonte_temp",
    "fim_de_semana", "eh_feriado", "eh_ponto_facultativo", "esfera_feriado", "nome_evento_calendario",
}
PREDICTION_REQUIRED = {
    "origem_t", "passo_h", "instante_alvo", "y_real", "y_pred", "y_baseline_semanal",
    "y_baseline_persistencia", "modelo", "etapa",
}
SPLIT_MAP = {"teste_2025": "test_2025", "forecast_2026": "forecast_2026", "validacao_2025": "validation_2025", "validacao_1s2025": "validation_2025"}
SCOPE_MAP = {"nacional": "nacional", "estadual_rj": "estadual", "municipal_rio": "municipal", "estadual": "estadual", "municipal": "municipal"}


def _empty_bundle(issue: str, provenance: dict | None = None) -> DataBundle:
    return DataBundle("provided", [], empty_observations(), [issue], provenance=provenance or {})


def _config(config_path: Path) -> tuple[dict, Path]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataLoadError(f"Configuração da base fornecida indisponível: {exc}") from exc
    required = {"archive", "sha256", "region", "unit", "timezone", "region_source", "timezone_source"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise DataLoadError("Configuração fornecida deve declarar archive, sha256, region, unit, timezone, region_source e timezone_source.")
    if any(not isinstance(config[key], str) or not config[key].strip() for key in required):
        raise DataLoadError("Todos os metadados da base fornecida devem ser textos não vazios.")
    if config["unit"] != "MWmed":
        raise DataLoadError("A coluna carga_mwmed exige unit='MWmed'; não há conversão implícita de unidade.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", config["sha256"]):
        raise DataLoadError("sha256 da base fornecida deve conter 64 caracteres hexadecimais.")
    archive_name = Path(config["archive"])
    if archive_name.is_absolute() or archive_name.name != config["archive"] or archive_name.suffix.lower() != ".zip":
        raise DataLoadError("archive deve ser o nome relativo de um ZIP na mesma pasta da configuração.")
    try:
        ZoneInfo(config["timezone"])
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise DataLoadError("Fuso horário inválido na configuração da base fornecida.") from exc
    return config, config_path.parent / archive_name


def _read_member(archive: ZipFile, member: str, columns: set[str] | None = None) -> pd.DataFrame:
    if archive.namelist().count(member) != 1:
        raise DataLoadError(f"O ZIP deve conter exatamente um arquivo {member}.")
    try:
        with archive.open(member) as file:
            frame = pd.read_csv(file, encoding="utf-8-sig", low_memory=False,
                                usecols=(lambda column: column in columns) if columns else None)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataLoadError(f"Não foi possível ler {member}: {exc}") from exc
    if columns and not columns.issubset(frame.columns):
        missing = sorted(columns.difference(frame.columns))
        raise DataLoadError(f"{member}: colunas obrigatórias ausentes: {', '.join(missing)}.")
    if frame.empty:
        raise DataLoadError(f"{member}: o arquivo está vazio.")
    return frame


def _utc_instants(values: pd.Series, timezone: str, label: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="raise")
        if parsed.isna().any() or parsed.dt.tz is not None:
            raise ValueError("datetimes ausentes ou com convenção diferente do CSV ingênuo")
        return parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC").dt.as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataLoadError(f"{label}: instante inválido para o fuso declarado {timezone}.") from exc


def _numeric(values: pd.Series, label: str, *, nullable: bool = False, nonnegative: bool = False) -> pd.Series:
    try:
        numbers = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataLoadError(f"{label}: os valores devem ser numéricos.") from exc
    if (not nullable and numbers.isna().any()) or (~np.isfinite(numbers) & numbers.notna()).any():
        raise DataLoadError(f"{label}: os valores devem ser finitos{' ou ausentes' if nullable else ''}.")
    if nonnegative and numbers.lt(0).any():
        raise DataLoadError(f"{label}: cargas negativas não são aceitas.")
    return numbers


def _history(raw: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    timestamp = _utc_instants(raw["instante"], config["timezone"], HISTORY_MEMBER)
    loads = _numeric(raw["carga_mwmed"], "carga_mwmed", nonnegative=True)
    observations = pd.DataFrame({"timestamp": timestamp, "observed_load": loads, "region": config["region"], "unit": config["unit"]})
    if timestamp.duplicated().any():
        raise DataLoadError("Série integrada: existem instantes duplicados.")
    observations = observations.sort_values("timestamp").reset_index(drop=True)
    differences = observations["timestamp"].diff().dropna().dt.total_seconds()
    if not differences.mod(3600).eq(0).all():
        raise DataLoadError("Série integrada: grade horária irregular; não se interpola o histórico.")
    weather_frames = []
    weather_nulls = {}
    for station, column in (("A621", "temp_a621_c"), ("A601", "temp_a601_c")):
        temperature = _numeric(raw[column], column, nullable=True)
        if not temperature.dropna().between(-60, 60).all():
            raise DataLoadError(f"{column}: temperatura fora de -60 a 60 °C.")
        weather_nulls[station] = int(temperature.isna().sum())
        available = temperature.notna()
        weather_frames.append(pd.DataFrame({
            "timestamp": timestamp[available], "temperature_c": temperature[available],
            "station": station, "region": config["region"],
        }))
    weather = pd.concat(weather_frames, ignore_index=True).sort_values(["station", "timestamp"]).reset_index(drop=True)
    flags = raw[["fim_de_semana", "eh_feriado", "eh_ponto_facultativo"]].apply(pd.to_numeric, errors="coerce")
    if not flags.isin([0, 1]).all().all():
        raise DataLoadError("Calendário integrado: os indicadores devem conter somente 0 ou 1.")
    local_dates = timestamp.dt.tz_convert(config["timezone"]).dt.date
    day_type = np.select(
        [flags["eh_feriado"].eq(1), flags["eh_ponto_facultativo"].eq(1), flags["fim_de_semana"].eq(1)],
        ["feriado", "ponto_facultativo", "fim_de_semana"], default="dia_util",
    )
    original_scope = raw["esfera_feriado"].fillna("").astype(str).str.strip()
    special = flags["eh_feriado"].eq(1) | flags["eh_ponto_facultativo"].eq(1)
    scope = original_scope.map(SCOPE_MAP).where(special, "nenhum").fillna("nao_informado")
    unknown_labels = original_scope.ne("") & ~original_scope.isin(SCOPE_MAP)
    if unknown_labels.any():
        raise DataLoadError("Calendário integrado: esfera declarada desconhecida; preserve o rótulo e revise os metadados.")
    event_name = raw["nome_evento_calendario"].fillna("").astype(str).str.strip()
    if (special & event_name.eq("")).any():
        raise DataLoadError("Calendário integrado: feriado ou ponto facultativo sem nome do evento.")
    calendar = pd.DataFrame({"date": local_dates, "region": config["region"], "day_type": day_type,
                             "holiday_scope": scope, "event_name": event_name})
    distinct = calendar.drop_duplicates()
    if distinct.duplicated(["region", "date"]).any():
        raise DataLoadError("Calendário integrado: categorias inconsistentes dentro do mesmo dia.")
    calendar = distinct.sort_values("date").reset_index(drop=True)
    provenance = {
        "history_source_file": HISTORY_MEMBER,
        "temperature_composition_source_counts": raw["fonte_temp"].fillna("nao_informado").value_counts().astype(int).to_dict(),
        "temperature_composite_used_as_observed_station": False,
        "weather_missing_by_station": weather_nulls,
        "calendar_unknown_scope_days": int(calendar["holiday_scope"].eq("nao_informado").sum()),
        "calendar_precedence": "feriado > ponto_facultativo > fim_de_semana > dia_util",
        "calendar_scope_original_counts": original_scope.replace("", "nao_informado").value_counts().astype(int).to_dict(),
    }
    return observations, weather, calendar, provenance


def _prediction_rows(raw: pd.DataFrame, horizon: int, config: dict, observations: pd.DataFrame,
                     member: str, issues: list[str]) -> pd.DataFrame:
    issued = _utc_instants(raw["origem_t"], config["timezone"], f"{member}.origem_t")
    target = _utc_instants(raw["instante_alvo"], config["timezone"], f"{member}.instante_alvo")
    step = _numeric(raw["passo_h"], f"{member}.passo_h")
    if not step.mod(1).eq(0).all() or not step.between(1, horizon).all():
        raise DataLoadError(f"{member}: passo_h deve ser inteiro entre 1 e {horizon}.")
    if not ((target - issued).dt.total_seconds() / 3600).eq(step).all():
        raise DataLoadError(f"{member}: os alvos não correspondem à emissão original e ao passo_h.")
    models = raw["modelo"].fillna("").astype(str).str.strip()
    if models.eq("").any():
        raise DataLoadError(f"{member}: modelo não pode ser vazio.")
    split = raw["etapa"].map(SPLIT_MAP)
    if split.isna().any():
        raise DataLoadError(f"{member}: etapa deve ser explicitamente reconhecida, sem inferência pelo ano.")
    keys = pd.DataFrame({"issued_at": issued, "model_name": models, "dataset_split": split})
    steps = keys.assign(forecast_step=step.astype(int))
    if steps.duplicated().any():
        raise DataLoadError(f"{member}: passos duplicados para a mesma emissão, modelo e split.")
    counts = steps.groupby(["issued_at", "model_name", "dataset_split"], sort=False)["forecast_step"].agg(["size", "min", "max"])
    if not (counts["size"].eq(horizon) & counts["min"].eq(1) & counts["max"].eq(horizon)).all():
        raise DataLoadError(f"{member}: cada emissão deve conter exatamente {horizon} passos consecutivos.")
    predicted = _numeric(raw["y_pred"], f"{member}.y_pred", nonnegative=True)
    provided_actual = _numeric(raw["y_real"], f"{member}.y_real", nullable=True, nonnegative=True)
    observed_series = observations.set_index("timestamp")["observed_load"]
    actual = target.map(observed_series)
    difference = (provided_actual - actual).abs()
    if (provided_actual.notna() & (actual.isna() | difference.gt(1e-6))).any():
        raise DataLoadError(f"{member}: y_real diverge do histórico observado ou não possui observação correspondente (tolerância 1e-6).")
    baselines = {}
    for source, destination, expected in (
        ("y_baseline_persistencia", "persistence_load", issued.map(observed_series)),
        ("y_baseline_semanal", "seasonal_load", (target - timedelta(hours=168)).map(observed_series)),
    ):
        supplied = _numeric(raw[source], f"{member}.{source}", nullable=True, nonnegative=True)
        if (supplied.notna() & expected.notna() & (supplied - expected).abs().gt(1e-6)).any():
            raise DataLoadError(f"{member}: {source} diverge dos dados disponíveis até a emissão original (tolerância 1e-6).")
        unverifiable = supplied.notna() & expected.isna()
        if unverifiable.any():
            issues.append(f"{member}: {int(unverifiable.sum())} valores de {source} sem histórico anterior verificável foram deixados indisponíveis.")
        baselines[destination] = supplied.where(expected.notna())
    unique_keys = keys.drop_duplicates().copy()
    unique_keys["forecast_id"] = f"archive:{config['sha256'][:12]}:{horizon}:" + pd.util.hash_pandas_object(unique_keys, index=False).astype(str)
    out = keys.merge(unique_keys, on=["issued_at", "model_name", "dataset_split"], how="left", validate="many_to_one")
    out["timestamp"] = target.to_numpy()
    out["issued_at_original"] = raw["origem_t"].astype(str).to_numpy()
    out["forecast_step"] = step.astype(int).to_numpy()
    out["lead_hour"] = step.to_numpy()
    out["predicted_load"] = predicted.to_numpy()
    out["observed_load"] = actual.to_numpy()
    for destination, values in baselines.items():
        out[destination] = values.to_numpy()
    out["model_version"] = "não informada"
    out["model_label"] = f"{horizon} h"
    out["horizon_hours"] = horizon
    out["region"] = config["region"]
    out["unit"] = config["unit"]
    out["source_file"] = member
    return out


def load_archive_bundle(config_path: str | Path) -> DataBundle:
    """Lê o ZIP autenticado por hash e retorna frames canônicos ou issues.

    Erros de configuração/histórico recusam a base. Um CSV de previsões inválido
    é omitido integralmente; não se completam emissões nem se trocam valores.
    O cache pertence à camada que chama este adaptador.
    """
    config_path = Path(config_path)
    try:
        config, archive_path = _config(config_path)
        with archive_path.open("rb") as archive_file:
            digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
        if digest.lower() != config["sha256"].lower():
            raise DataLoadError("O SHA-256 do ZIP difere da configuração; a base não foi carregada.")
        provenance = dict(config, archive_path=str(archive_path), archive_sha256=digest,
                          geographic_scope_status="informado pelo grupo", timezone_status="informado pelo grupo")
        with ZipFile(archive_path) as archive:
            raw_history = _read_member(archive, HISTORY_MEMBER, HISTORY_REQUIRED)
            observations, weather, calendar, details = _history(raw_history, config)
            provenance.update(details)
            issues = []
            frames = []
            for horizon, member in PREDICTION_MEMBERS.items():
                try:
                    raw = _read_member(archive, member, PREDICTION_REQUIRED)
                    frames.append(_prediction_rows(raw, horizon, config, observations, member, issues))
                except DataLoadError as exc:
                    issues.append(str(exc))
            summaries = []
            for member in SUMMARY_MEMBERS:
                if member in archive.namelist():
                    try:
                        summaries.append(_read_member(archive, member).assign(source_file=member))
                    except DataLoadError as exc:
                        issues.append(str(exc))
            rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            reported = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
            provenance["prediction_source_row_counts"] = rows["source_file"].value_counts().astype(int).to_dict() if not rows.empty else {}
            provenance["reported_metrics_source_files"] = [str(frame["source_file"].iloc[0]) for frame in summaries]
            return DataBundle("provided", [], observations, issues, weather=weather, calendar=calendar,
                              prediction_rows=rows, reported_metrics=reported, provenance=provenance)
    except (DataLoadError, OSError, BadZipFile, ValueError, KeyError) as exc:
        return _empty_bundle(f"Base fornecida indisponível: {exc}")
