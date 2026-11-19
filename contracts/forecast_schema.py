"""Validação independente da interface para previsões horárias.

Instantes são normalizados em UTC; a emissão continua representando o instante
original do modelo. Não se ordenam nem se completam previsões inválidas.
"""

from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_UNITS = {"MW", "MWmed"}
DATASET_SPLITS = {"validation_2025", "test_2025", "forecast_2026", "unspecified"}


class ForecastValidationError(ValueError):
    """Uma previsão não atende ao contrato de integração."""


@dataclass(frozen=True)
class Forecast:
    model_name: str
    model_version: str
    horizon_hours: int
    region: str
    unit: str
    issued_at: pd.Timestamp
    predictions: pd.DataFrame
    source_path: Path | None = None
    issued_at_original: str = ""
    dataset_split: str = "unspecified"


def parse_timestamp(value: Any, field_name: str) -> pd.Timestamp:
    """Aceita ISO 8601 com fuso explícito e retorna um instante UTC."""
    if not isinstance(value, str) or not value.strip():
        raise ForecastValidationError(f"{field_name}: informe um datetime ISO 8601 com fuso horário.")
    try:
        # datetime.fromisoformat evita interpretações ambíguas de datas locais.
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("fuso ausente")
        if "T" not in value.upper():
            raise ValueError("separador ISO ausente")
        # O parser datetime verifica ISO/fuso; o texto original preserva a
        # precisão de nanossegundos que datetime converteria a microssegundos.
        return pd.Timestamp(value.strip()).tz_convert("UTC").as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ForecastValidationError(
            f"{field_name}: datetime inválido; use ISO 8601 com fuso horário (ex.: 2026-02-12T00:00:00-03:00)."
        ) from exc


def _required_text(payload: dict[str, Any], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str) or not value.strip():
        raise ForecastValidationError(f"{field_name}: informe um texto não vazio.")
    return value.strip()


def validate_forecast(payload: dict[str, Any]) -> Forecast:
    """Valida metadados, horizonte, valores e sequência temporal de um JSON."""
    if not isinstance(payload, dict):
        raise ForecastValidationError("O arquivo de previsão deve conter um objeto JSON.")
    required = {
        "model_name", "model_version", "horizon_hours", "region", "unit",
        "issued_at", "predictions",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ForecastValidationError(f"Campos obrigatórios ausentes: {', '.join(missing)}.")

    model_name = _required_text(payload, "model_name")
    model_version = _required_text(payload, "model_version")
    region = _required_text(payload, "region")
    if not isinstance(payload["unit"], str) or payload["unit"] not in SUPPORTED_UNITS:
        raise ForecastValidationError("unit: a unidade de carga deve ser exatamente 'MW' ou 'MWmed', sem conversão implícita.")
    dataset_split = payload.get("dataset_split", "unspecified")
    if not isinstance(dataset_split, str) or dataset_split not in DATASET_SPLITS:
        raise ForecastValidationError(
            "dataset_split: use validation_2025, test_2025, forecast_2026 ou unspecified."
        )
    horizon = payload["horizon_hours"]
    if type(horizon) is not int or horizon not in (24, 48):
        raise ForecastValidationError("horizon_hours: o horizonte deve ser o inteiro 24 ou 48.")
    issued_at = parse_timestamp(payload["issued_at"], "issued_at")
    rows = payload["predictions"]
    if not isinstance(rows, list) or len(rows) != horizon:
        raise ForecastValidationError(f"predictions: são necessárias exatamente {horizon} previsões.")

    timestamps: list[pd.Timestamp] = []
    loads: list[float] = []
    for index, row in enumerate(rows):
        label = f"predictions[{index}]"
        if not isinstance(row, dict) or not {"timestamp", "predicted_load"}.issubset(row):
            raise ForecastValidationError(f"{label}: timestamp e predicted_load são obrigatórios.")
        timestamps.append(parse_timestamp(row["timestamp"], f"{label}.timestamp"))
        value = row["predicted_load"]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ForecastValidationError(f"{label}.predicted_load: informe um número finito na unidade declarada.")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ForecastValidationError(f"{label}.predicted_load: informe um número finito na unidade declarada.") from exc
        if not np.isfinite(numeric) or numeric < 0:
            raise ForecastValidationError(f"{label}.predicted_load: a carga deve ser finita e não negativa na unidade declarada.")
        loads.append(numeric)

    time_index = pd.DatetimeIndex(timestamps)
    if time_index.has_duplicates:
        raise ForecastValidationError("predictions: existem timestamps duplicados, inclusive após normalização para UTC.")
    if not time_index.is_monotonic_increasing:
        raise ForecastValidationError("predictions: os timestamps devem estar ordenados em ordem crescente.")
    differences = time_index.to_series().diff().iloc[1:].dt.total_seconds()
    if not differences.eq(3600).all():
        raise ForecastValidationError("predictions: os timestamps devem ter intervalos horários consecutivos de uma hora.")
    if time_index[0] <= issued_at:
        raise ForecastValidationError("predictions: todos os instantes previstos devem ser posteriores à emissão (issued_at).")
    if int(time_index[0].value) - int(issued_at.value) > 3600 * 1_000_000_000:
        raise ForecastValidationError(
            "predictions: o primeiro alvo deve ocorrer até uma hora após a emissão (issued_at); "
            "não se admite uma lacuna antes do horizonte."
        )

    return Forecast(
        model_name=model_name,
        model_version=model_version,
        horizon_hours=horizon,
        region=region,
        unit=payload["unit"],
        issued_at=issued_at,
        predictions=pd.DataFrame({"timestamp": time_index, "predicted_load": loads}),
        issued_at_original=payload["issued_at"],
        dataset_split=dataset_split,
    )
