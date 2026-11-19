"""Catálogo leve de emissões e materialização sob demanda de uma previsão.

Lotes tabulares permanecem em um único DataFrame: não são transformados em
milhares de objetos Forecast para preencher os seletores da interface.
"""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from contracts.forecast_schema import DATASET_SPLITS, SUPPORTED_UNITS, Forecast, validate_forecast
from services.analysis_service import forecast_identifier

if TYPE_CHECKING:
    from services.data_loader import DataBundle


CATALOG_COLUMNS = [
    "forecast_id", "model_name", "model_version", "horizon_hours", "issued_at",
    "region", "unit", "dataset_split", "source_file",
]
UNKNOWN_VERSION = "não informada"


def _empty_catalog() -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in CATALOG_COLUMNS})
    frame["issued_at"] = pd.Series(dtype="datetime64[ns, UTC]")
    frame["horizon_hours"] = pd.Series(dtype="int64")
    return frame


def _batch_rows(bundle: DataBundle) -> pd.DataFrame:
    rows = getattr(bundle, "prediction_rows", None)
    return rows if isinstance(rows, pd.DataFrame) else pd.DataFrame()


def _aware_utc(value: object, field: str) -> pd.Timestamp:
    try:
        instant = pd.Timestamp(value)
        if pd.isna(instant) or instant.tzinfo is None:
            raise ValueError("fuso ausente")
        return instant.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field}: informe um instante válido com fuso horário.") from exc


def _native_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CATALOG_COLUMNS).difference(rows.columns).difference({"model_version"}))
    if missing:
        raise ValueError(f"Lote de previsões sem metadados obrigatórios: {', '.join(missing)}.")
    columns = [column for column in CATALOG_COLUMNS if column in rows]
    # A operação vetorizada reduz centenas de milhares de alvos a suas emissões.
    metadata = rows[columns].drop_duplicates().copy()
    if "model_version" not in metadata:
        metadata["model_version"] = UNKNOWN_VERSION
    else:
        metadata["model_version"] = metadata["model_version"].fillna(UNKNOWN_VERSION)
        metadata.loc[metadata["model_version"].eq(""), "model_version"] = UNKNOWN_VERSION
    for column in ("forecast_id", "model_name", "model_version", "region", "unit", "dataset_split"):
        if not metadata[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Metadado {column} deve conter texto não vazio em todas as emissões.")
    if not metadata["horizon_hours"].map(
        lambda value: isinstance(value, Integral) and not isinstance(value, bool) and value in (24, 48)
    ).all():
        raise ValueError("O horizonte de cada emissão deve ser o inteiro 24 ou 48.")
    if not metadata["unit"].isin(SUPPORTED_UNITS).all():
        raise ValueError("As unidades do catálogo devem ser MW ou MWmed, sem conversão implícita.")
    if not metadata["dataset_split"].isin(DATASET_SPLITS).all():
        raise ValueError("O catálogo contém uma partição dataset_split não reconhecida.")
    metadata["issued_at"] = pd.to_datetime([
        _aware_utc(value, "issued_at") for value in metadata["issued_at"]
    ], utc=True)
    if metadata["source_file"].isna().any():
        raise ValueError("Cada emissão deve identificar seu arquivo de origem.")
    metadata["source_file"] = metadata["source_file"].map(str)
    if metadata["source_file"].eq("").any():
        raise ValueError("Cada emissão deve identificar seu arquivo de origem.")
    metadata = metadata[CATALOG_COLUMNS].drop_duplicates()
    if metadata["forecast_id"].duplicated().any():
        raise ValueError("Há metadados conflitantes para o mesmo forecast_id; as emissões não podem ser combinadas.")
    return metadata


def _sort_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["issued_at", "horizon_hours", "model_name", "model_version", "forecast_id"],
        ascending=[False, True, True, True, True], kind="stable",
    ).reset_index(drop=True)


def forecast_catalog(bundle: DataBundle) -> pd.DataFrame:
    """Retorna uma linha por emissão, sem construir Forecast para um lote nativo."""
    rows = _batch_rows(bundle)
    if not rows.empty:
        return _sort_catalog(_native_metadata(rows))
    records = []
    for forecast in getattr(bundle, "forecasts", []):
        records.append({
            "forecast_id": forecast_identifier(forecast), "model_name": forecast.model_name,
            "model_version": forecast.model_version, "horizon_hours": forecast.horizon_hours,
            "issued_at": _aware_utc(forecast.issued_at, "issued_at"), "region": forecast.region,
            "unit": forecast.unit, "dataset_split": forecast.dataset_split,
            "source_file": str(forecast.source_path) if forecast.source_path else "",
        })
    if not records:
        return _empty_catalog()
    # Cópias idênticas de JSONs possuem ID de conteúdo igual, mesmo após renomear.
    frame = pd.DataFrame(records, columns=CATALOG_COLUMNS)
    frame = frame.sort_values("source_file", kind="stable").drop_duplicates("forecast_id")
    return _sort_catalog(frame)


def materialize_forecast(bundle: DataBundle, forecast_id: str) -> Forecast:
    """Valida e materializa somente os 24/48 alvos da emissão selecionada."""
    if not isinstance(forecast_id, str) or not forecast_id:
        raise ValueError("Selecione um forecast_id válido.")
    rows = _batch_rows(bundle)
    if rows.empty:
        candidates = [
            forecast for forecast in getattr(bundle, "forecasts", [])
            if forecast_identifier(forecast) == forecast_id
        ]
        if not candidates:
            raise ValueError(f"Emissão desconhecida: {forecast_id}.")
        return min(candidates, key=lambda forecast: str(forecast.source_path or ""))
    if "forecast_id" not in rows:
        raise ValueError("O lote de previsões deve conter forecast_id.")
    selected = rows.loc[rows["forecast_id"].eq(forecast_id)]
    if selected.empty:
        raise ValueError(f"Emissão desconhecida: {forecast_id}.")
    metadata = _native_metadata(selected).iloc[0]
    horizon = int(metadata["horizon_hours"])
    if len(selected) != horizon:
        raise ValueError(f"A emissão selecionada exige exatamente {horizon} alvos; encontrados {len(selected)}.")
    missing = sorted({"timestamp", "predicted_load", "forecast_step"}.difference(selected.columns))
    if missing:
        raise ValueError(f"A emissão selecionada não contém: {', '.join(missing)}.")
    if not selected["forecast_step"].map(
        lambda value: isinstance(value, Integral) and not isinstance(value, bool)
    ).all() or sorted(selected["forecast_step"].tolist()) != list(range(1, horizon + 1)):
        raise ValueError(f"Os passos da emissão devem ser únicos e inteiros de 1 a {horizon}.")
    selected = selected.sort_values("forecast_step", kind="stable")
    payload = {
        "model_name": metadata["model_name"], "model_version": metadata["model_version"],
        "horizon_hours": horizon, "issued_at": metadata["issued_at"].isoformat(),
        "region": metadata["region"], "unit": metadata["unit"],
        "dataset_split": metadata["dataset_split"],
        "predictions": [
            {"timestamp": _aware_utc(timestamp, "timestamp").isoformat(), "predicted_load": value}
            for timestamp, value in selected[["timestamp", "predicted_load"]].itertuples(index=False, name=None)
        ],
    }
    forecast = validate_forecast(payload)
    original = forecast.issued_at_original
    if "issued_at_original" in selected:
        originals = selected["issued_at_original"].dropna().unique()
        if len(originals) > 1:
            raise ValueError("A emissão selecionada contém origens temporais textuais conflitantes.")
        if len(originals) == 1 and str(originals[0]):
            original = str(originals[0])
    return replace(forecast, source_path=Path(metadata["source_file"]), issued_at_original=original)
