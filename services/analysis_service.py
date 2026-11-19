"""Retrospective comparisons and contextual joins without temporal leakage.

Weather at a target is retrospective context, never an input claimed to have
been available at issuance. Baselines use observed loads available by issuance.
MW and MWmed, geographic regions and dataset splits remain separate groups.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from contracts.forecast_schema import Forecast
from services.forecast_service import _prediction_frame, _utc_timestamp, normalize_observations
from services.metrics_service import calculate_metrics


LOCAL_TIMEZONE = "America/Sao_Paulo"
NS_PER_HOUR = 3_600_000_000_000
NS_PER_WEEK = 168 * NS_PER_HOUR
ROW_COLUMNS = [
    "forecast_id", "model_name", "model_version", "model_label", "horizon_hours",
    "issued_at", "timestamp", "forecast_step", "lead_hour", "predicted_load", "observed_load",
    "residual", "absolute_error", "dataset_split", "unit", "region",
    "persistence_load", "seasonal_load", "temperature_c", "temperature_station_count",
    "day_type", "holiday_scope", "event_name", "month", "weekday",
]
GROUP_COLUMNS = ["region", "unit", "dataset_split"]
MODEL_COLUMNS = ["model_name", "model_version", "model_label", "horizon_hours"]
COMPARISON_COLUMNS = [
    *GROUP_COLUMNS, *MODEL_COLUMNS, "segment", "method", "total_targets", "n_pairs",
    "coverage", "mae", "rmse", "mape", "mape_valid_pairs", "mape_excluded_zeros",
    "n_issuances", "available_issuances", "comparison_verified", "sample_policy",
]
LEAD_COLUMNS = [*GROUP_COLUMNS, *MODEL_COLUMNS, "forecast_step", "lead_hour_min", "lead_hour_max", "n_pairs", "mape_valid_pairs", "mape_excluded_zeros", "mae", "rmse", "mape"]
DAY_TYPES = {"dia_util", "fim_de_semana", "feriado", "ponto_facultativo"}
HOLIDAY_SCOPES = {"nacional", "estadual", "municipal", "nenhum", "nao_informado"}


def _cutoff(as_of: pd.Timestamp | None) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") if as_of is None else _utc_timestamp(as_of, "as_of")


def _empty_rows() -> pd.DataFrame:
    frame = pd.DataFrame(columns=ROW_COLUMNS)
    for column in ("timestamp", "issued_at"):
        frame[column] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame


def forecast_identifier(forecast: Forecast) -> str:
    """A content-based ID survives renaming or moving a forecast JSON file."""
    predictions = _prediction_frame(forecast)
    payload = {
        "model_name": forecast.model_name,
        "model_version": forecast.model_version,
        "horizon_hours": forecast.horizon_hours,
        "region": forecast.region,
        "unit": forecast.unit,
        "issued_at": _utc_timestamp(forecast.issued_at, "issued_at").isoformat(),
        "dataset_split": getattr(forecast, "dataset_split", "unspecified"),
        "predictions": [
            [timestamp.isoformat(), float(value)]
            for timestamp, value in predictions.itertuples(index=False, name=None)
        ],
    }
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]


def _observed_series(
    observations: pd.DataFrame, region: str, unit: str, cutoff: pd.Timestamp,
) -> pd.Series:
    frame = observations.loc[
        observations["region"].eq(region) & observations["unit"].eq(unit)
        & observations["timestamp"].le(cutoff)
    ]
    return frame.set_index("timestamp")["observed_load"].sort_index()


def _persistence(series: pd.Series, issued_at: pd.Timestamp) -> float:
    available = series.loc[series.index <= issued_at]
    if available.empty:
        return np.nan
    latest = available.index[-1]
    return float(available.iloc[-1]) if issued_at.value - latest.value <= NS_PER_HOUR else np.nan


def _seasonal(
    targets: pd.Series, series: pd.Series, issued_at: pd.Timestamp,
) -> list[float]:
    result = []
    for target in targets:
        reference = pd.Timestamp(target.value - NS_PER_WEEK, tz="UTC")
        result.append(
            float(series.loc[reference])
            if reference <= issued_at and reference in series.index else np.nan
        )
    return result


def _weather_frame(weather: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["timestamp", "temperature_c", "station", "region"]
    if weather is None or weather.empty:
        return pd.DataFrame(columns=columns)
    if not set(columns).issubset(weather.columns):
        raise ValueError("Meteorologia deve conter timestamp, temperature_c, station e region.")
    frame = weather[columns].copy()
    if isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype):
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("UTC")
        if frame["timestamp"].isna().any():
            raise ValueError("Meteorologia contém timestamps inválidos.")
    else:
        frame["timestamp"] = pd.to_datetime(
            [_utc_timestamp(value, "timestamp meteorológico") for value in frame["timestamp"]], utc=True,
        )
    for column in ("station", "region"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Meteorologia: {column} deve ser texto não vazio.")
    if frame["temperature_c"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("Temperaturas devem ser números finitos.")
    try:
        frame["temperature_c"] = pd.to_numeric(frame["temperature_c"], errors="raise").astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Temperaturas devem ser números finitos.") from exc
    if not np.isfinite(frame["temperature_c"]).all():
        raise ValueError("Temperaturas devem ser números finitos.")
    if frame.duplicated(["region", "station", "timestamp"]).any():
        raise ValueError("Meteorologia contém observações duplicadas por estação, região e instante.")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _calendar_frame(calendar: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["date", "region", "day_type", "holiday_scope", "event_name"]
    if calendar is None or calendar.empty:
        return pd.DataFrame(columns=columns)
    if not set(columns).issubset(calendar.columns):
        raise ValueError("Calendário deve conter date, region, day_type, holiday_scope e event_name.")
    frame = calendar[columns].copy()
    try:
        frame["date"] = frame["date"].map(lambda value: date.fromisoformat(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Calendário: date deve ser uma data local YYYY-MM-DD.") from exc
    if not frame["day_type"].isin(DAY_TYPES).all():
        raise ValueError("Calendário contém um day_type não reconhecido.")
    if not frame["holiday_scope"].isin(HOLIDAY_SCOPES).all():
        raise ValueError("Calendário contém um holiday_scope não reconhecido.")
    if frame.duplicated(["region", "date"]).any():
        raise ValueError("Calendário contém datas duplicadas para a mesma região.")
    frame["event_name"] = frame["event_name"].fillna("").astype(str)
    return frame


def calendar_for_targets(forecast: Forecast, calendar: pd.DataFrame | None) -> pd.DataFrame:
    """Use only explicitly supplied local calendar labels; never infer holidays."""
    frame = _prediction_frame(forecast)[["timestamp"]].copy()
    frame["date"] = frame["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date
    supplied = _calendar_frame(calendar)
    supplied = supplied.loc[supplied["region"].eq(forecast.region)].drop(columns="region")
    frame = frame.merge(supplied, on="date", how="left", validate="many_to_one")
    frame["day_type"] = frame["day_type"].fillna("nao_informado")
    frame["holiday_scope"] = frame["holiday_scope"].fillna("nao_informado")
    frame["event_name"] = frame["event_name"].fillna("")
    return frame[["timestamp", "date", "day_type", "holiday_scope", "event_name"]]


def temperature_at_issue(
    weather: pd.DataFrame | None, issued_at: pd.Timestamp,
    region: str, station: str | None = None, as_of: pd.Timestamp | None = None,
) -> dict | None:
    """Last observed temperature per station, at most one hour before issuance.

    No target-time weather or observation newer than issuance is used. When
    multiple stations are selected their last available values are averaged;
    station count and oldest/latest timestamps make that aggregation explicit.
    """
    issue = _utc_timestamp(issued_at, "issued_at")
    cutoff = min(issue, _cutoff(as_of))
    frame = _weather_frame(weather)
    frame = frame.loc[frame["region"].eq(region) & frame["timestamp"].le(cutoff)]
    if station is not None:
        frame = frame.loc[frame["station"].eq(station)]
    if frame.empty:
        return None
    latest = frame.sort_values("timestamp").groupby("station", as_index=False).tail(1)
    latest = latest.loc[latest["timestamp"].map(lambda stamp: issue.value - stamp.value <= NS_PER_HOUR)]
    if latest.empty:
        return None
    values = latest["temperature_c"].to_numpy(dtype=float)
    # Temperatures may be negative. Scaling also avoids an overflowing sum.
    scale = float(np.abs(values).max())
    mean = float(np.mean(values / scale) * scale) if scale else 0.0
    return {
        "temperature_c": mean,
        "station_count": int(latest["station"].nunique()),
        "stations": sorted(latest["station"].unique().tolist()),
        "timestamp": latest["timestamp"].max(),
        "oldest_timestamp": latest["timestamp"].min(),
        "station": station,
        "aggregation": "média aritmética das últimas observações por estação",
    }


def build_evaluation_rows(
    forecasts: Iterable[Forecast], observations: pd.DataFrame,
    weather: pd.DataFrame | None = None, calendar: pd.DataFrame | None = None,
    region: str | None = None, unit: str | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per forecast target, with unavailable actuals left as NaN.

    The persistence and weekly baselines use only observations at/before the
    original issuance. Weather at targets is attached only retrospectively.
    Identical copies of one run are deduplicated by content, while conflicting
    copies with the same model/version/issuance identity raise an error.
    """
    cutoff = _cutoff(as_of)
    observed = normalize_observations(observations)
    temperatures = _weather_frame(weather)
    temperatures = temperatures.loc[temperatures["timestamp"].le(cutoff)]
    if not temperatures.empty:
        temperatures = temperatures.groupby(["region", "timestamp"], as_index=False).agg(
            temperature_c=("temperature_c", "mean"), temperature_station_count=("station", "nunique"),
        )
    else:
        temperatures = pd.DataFrame(columns=["region", "timestamp", "temperature_c", "temperature_station_count"])
    calendars = _calendar_frame(calendar)
    frames, identities = [], {}
    for forecast in forecasts:
        if (region is not None and forecast.region != region) or (unit is not None and forecast.unit != unit):
            continue
        frame = _prediction_frame(forecast)
        issue = _utc_timestamp(forecast.issued_at, "issued_at")
        split = getattr(forecast, "dataset_split", "unspecified")
        forecast_id = forecast_identifier(forecast)
        identity = (forecast.model_name, forecast.model_version, forecast.horizon_hours, forecast.region, forecast.unit, issue, split)
        if identity in identities:
            if identities[identity] != forecast_id:
                raise ValueError("Há previsões conflitantes para o mesmo modelo, versão, emissão e horizonte.")
            continue
        identities[identity] = forecast_id
        series = _observed_series(observed, forecast.region, forecast.unit, cutoff)
        frame["observed_load"] = frame["timestamp"].map(series)
        frame["residual"] = frame["observed_load"] - frame["predicted_load"]
        frame["absolute_error"] = frame["residual"].abs()
        frame["persistence_load"] = _persistence(series, issue)
        frame["seasonal_load"] = _seasonal(frame["timestamp"], series, issue)
        frame["forecast_id"] = forecast_id
        frame["model_name"] = forecast.model_name
        frame["model_version"] = forecast.model_version
        frame["model_label"] = f"{forecast.horizon_hours} h"
        frame["horizon_hours"] = forecast.horizon_hours
        frame["issued_at"] = issue
        frame["forecast_step"] = np.arange(1, len(frame) + 1)
        frame["lead_hour"] = (frame["timestamp"] - issue).dt.total_seconds() / 3600
        frame["dataset_split"] = split
        frame["region"] = forecast.region
        frame["unit"] = forecast.unit
        frame = frame.merge(temperatures, on=["region", "timestamp"], how="left", validate="many_to_one")
        frame["temperature_station_count"] = pd.to_numeric(frame["temperature_station_count"], errors="raise").fillna(0).astype(int)
        labels = calendar_for_targets(forecast, calendars).drop(columns="date")
        frame = frame.merge(labels, on="timestamp", how="left", validate="one_to_one")
        local = frame["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
        frame["month"] = local.dt.month
        frame["weekday"] = local.dt.dayofweek
        frames.append(frame[ROW_COLUMNS])
    if not frames:
        return _empty_rows()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["region", "unit", "dataset_split", "issued_at", "horizon_hours", "model_name", "timestamp"]
    ).reset_index(drop=True)


def _metrics_for(frame: pd.DataFrame, prediction_column: str) -> dict:
    # Grouping guarantees compatible metadata; metrics are intentionally
    # aggregated over the selected issuances rather than pretending one origin.
    pairs = frame[["observed_load", prediction_column]].rename(columns={prediction_column: "predicted_load"})
    return calculate_metrics(pairs)


def _segments(frame: pd.DataFrame, horizon: int) -> list[tuple[str, pd.DataFrame]]:
    first = frame.loc[frame["forecast_step"].le(24)]
    if horizon == 24:
        return [("1–24 h", first)]
    return [
        ("1–24 h", first),
        ("25–48 h", frame.loc[frame["forecast_step"].gt(24)]),
        ("1–48 h", frame),
    ]


def comparison_metrics(rows: pd.DataFrame, *, common_sample: bool = True) -> pd.DataFrame:
    """Compare models and two baselines on explicitly reported samples.

    Region, exact unit and declared split never share a metric. If 24h and 48h
    runs have common issuances, only those origins are selected. For their
    first 24h, common_sample also intersects exact (issuance,target) pairs
    across model/version groups. In each segment, the same observed/baseline
    pairs are used for all three methods. Without a common origin the output
    remains descriptive and the sample_policy explicitly reports that limit.
    Coverage is n_pairs / total_targets after issuance selection; zero pairs
    produce undefined metrics, never fictitious zero errors.
    comparison_verified concerns the paired first 24 steps across horizons;
    it is false for independent samples, absent pairs and aggregate-only data.
    """
    if rows.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    required = set(GROUP_COLUMNS + MODEL_COLUMNS + ["issued_at", "timestamp", "forecast_step", "lead_hour", "predicted_load", "observed_load", "persistence_load", "seasonal_load"])
    if not required.issubset(rows.columns):
        raise ValueError("As linhas de avaliação não contêm as colunas necessárias para comparação.")
    results = []
    for group_key, context in rows.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        context = context.copy()
        origins_by_horizon = {
            horizon: set(context.loc[context["horizon_hours"].eq(horizon), "issued_at"])
            for horizon in (24, 48)
        }
        common_origins = origins_by_horizon[24] & origins_by_horizon[48]
        available = {
            key: group["issued_at"].nunique()
            for key, group in context.groupby(MODEL_COLUMNS, dropna=False)
        }
        if common_origins:
            context = context.loc[context["issued_at"].isin(common_origins)]
            policy = "Emissões comuns entre 24 h e 48 h"
        elif origins_by_horizon[24] and origins_by_horizon[48]:
            policy = "Emissões disponíveis; sem emissões comuns entre 24 h e 48 h"
        else:
            policy = "Emissões disponíveis; somente um horizonte disponível"
        shared_first_pairs = None
        if common_sample and common_origins:
            pair_sets = []
            for _, candidate in context.groupby(MODEL_COLUMNS, dropna=False):
                candidate = candidate.loc[candidate["forecast_step"].le(24)].dropna(
                    subset=["observed_load", "predicted_load", "persistence_load", "seasonal_load"]
                )
                pair_sets.append(set(zip(candidate["issued_at"], candidate["timestamp"], strict=True)))
            shared_first_pairs = set.intersection(*pair_sets) if pair_sets else set()
        comparison_verified = bool(common_sample and common_origins and shared_first_pairs)
        if not comparison_verified:
            policy += "; comparação entre horizontes não verificada"
        for model_key, model_rows in context.groupby(MODEL_COLUMNS, dropna=False, sort=True):
            horizon = int(model_key[-1])
            for segment, targets in _segments(model_rows, horizon):
                total_targets = len(targets)
                shared = targets.dropna(subset=["observed_load", "predicted_load", "persistence_load", "seasonal_load"])
                if shared_first_pairs is not None and segment == "1–24 h":
                    mask = [(issue, timestamp) in shared_first_pairs for issue, timestamp in zip(shared["issued_at"], shared["timestamp"], strict=True)]
                    shared = shared.loc[mask]
                for method, column in (("Modelo", "predicted_load"), ("Persistência", "persistence_load"), ("Sazonal semanal", "seasonal_load")):
                    sample = shared if common_sample else targets.dropna(subset=["observed_load", column])
                    metrics = _metrics_for(sample, column)
                    results.append({
                        **dict(zip(GROUP_COLUMNS, group_key, strict=True)),
                        **dict(zip(MODEL_COLUMNS, model_key, strict=True)),
                        "segment": segment,
                        "method": method,
                        "total_targets": total_targets,
                        **metrics,
                        "coverage": metrics["n_pairs"] / total_targets if total_targets else 0.0,
                        "n_issuances": int(targets["issued_at"].nunique()),
                        "available_issuances": int(available[model_key]),
                        "comparison_verified": comparison_verified,
                        "sample_policy": policy + (
                            "; interseção de observados e ambos os baselines" if common_sample
                            else "; amostras disponíveis independentes por método"
                        ),
                    })
    return pd.DataFrame(results, columns=COMPARISON_COLUMNS)


def error_by_lead(rows: pd.DataFrame) -> pd.DataFrame:
    """Available observed pairs by integer forecast step, without filling.

    Elapsed lead is retained as a min/max range; it is not silently relabeled
    as an integer when an issuance falls between whole hours.
    MAPE sample counts are retained separately from the MAE/RMSE pair count.
    """
    if rows.empty:
        return pd.DataFrame(columns=LEAD_COLUMNS)
    available = rows.dropna(subset=["observed_load", "predicted_load"])
    result = []
    keys = GROUP_COLUMNS + MODEL_COLUMNS + ["forecast_step"]
    for values, group in available.groupby(keys, dropna=False, sort=True):
        result.append({
            **dict(zip(keys, values, strict=True)), **_metrics_for(group, "predicted_load"),
            "lead_hour_min": float(group["lead_hour"].min()), "lead_hour_max": float(group["lead_hour"].max()),
        })
    return pd.DataFrame(result).reindex(columns=LEAD_COLUMNS)


def largest_errors(
    rows: pd.DataFrame, month: int | Iterable[int] | None = None,
    weekday: int | Iterable[int] | None = None, day_types: Iterable[str] | None = None,
    temp_range: tuple[float, float] | None = None, limit: int = 20,
) -> pd.DataFrame:
    """Filter retrospective errors using local month/weekday and explicit labels."""
    if type(limit) is not int or limit < 1:
        raise ValueError("O limite deve ser um inteiro positivo.")
    if rows.empty:
        return rows.copy()
    frame = rows.dropna(subset=["observed_load", "absolute_error"]).copy()
    local = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(LOCAL_TIMEZONE)
    frame["month"], frame["weekday"] = local.dt.month, local.dt.dayofweek
    for selection, column in ((month, "month"), (weekday, "weekday")):
        if selection is not None:
            selected = [selection] if isinstance(selection, (int, np.integer)) else list(selection)
            frame = frame.loc[frame[column].isin(selected)]
    if day_types is not None:
        selected_types = [day_types] if isinstance(day_types, str) else list(day_types)
        frame = frame.loc[frame["day_type"].isin(selected_types)]
    if temp_range is not None:
        minimum, maximum = temp_range
        if not np.isfinite([minimum, maximum]).all() or minimum > maximum:
            raise ValueError("O intervalo de temperatura deve ser finito e crescente.")
        frame = frame.loc[frame["temperature_c"].between(minimum, maximum, inclusive="both")]
    return frame.sort_values(["absolute_error", "timestamp", "forecast_id"], ascending=[False, True, True]).head(limit).reset_index(drop=True)


def explorer_frame(
    forecast: Forecast, observations: pd.DataFrame, history_hours: int = 72,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return observed context and selected targets with a leakage-free baseline."""
    if type(history_hours) is not int or not 48 <= history_hours <= 168:
        raise ValueError("O contexto histórico deve estar entre 48 e 168 horas.")
    cutoff = _cutoff(as_of)
    issue = _utc_timestamp(forecast.issued_at, "issued_at")
    observed = normalize_observations(observations)
    series = _observed_series(observed, forecast.region, forecast.unit, cutoff)
    history_start = pd.Timestamp(issue.value - history_hours * NS_PER_HOUR, tz="UTC")
    history = series.loc[(series.index <= issue) & (series.index > history_start)].rename("observed_load").reset_index()
    history["predicted_load"] = np.nan
    history["seasonal_load"] = np.nan
    history["phase"] = "histórico"
    future = _prediction_frame(forecast)
    future["observed_load"] = future["timestamp"].map(series)
    future["seasonal_load"] = _seasonal(future["timestamp"], series, issue)
    future["phase"] = "previsão"
    columns = ["timestamp", "observed_load", "predicted_load", "seasonal_load", "phase"]
    return pd.concat([history[columns], future[columns]], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
