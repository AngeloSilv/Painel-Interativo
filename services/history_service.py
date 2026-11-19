"""Seleção temporal do histórico, sem interpolação ou preenchimento de lacunas."""

from datetime import date, timedelta

import pandas as pd

from contracts.forecast_schema import Forecast


def _availability_cutoff(as_of: pd.Timestamp | None) -> pd.Timestamp:
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise ValueError("O instante de disponibilidade deve conter fuso horário.")
    return cutoff.tz_convert("UTC")


def future_observation_count(
    observations: pd.DataFrame, region: str | None = None,
    *, unit: str | None = None, as_of: pd.Timestamp | None = None,
) -> int:
    if observations.empty:
        return 0
    selected = observations if region is None else observations.loc[observations["region"] == region]
    if unit is not None:
        selected = selected.loc[selected["unit"] == unit]
    return int(selected["timestamp"].gt(_availability_cutoff(as_of)).sum())


def region_history(
    observations: pd.DataFrame, region: str | None,
    *, unit: str | None = None, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if observations.empty or region is None:
        return observations.iloc[:0].copy()
    selected = observations.loc[
        (observations["region"] == region)
        & observations["timestamp"].le(_availability_cutoff(as_of))
    ].sort_values("timestamp").reset_index(drop=True)
    if unit is not None:
        selected = selected.loc[selected["unit"] == unit].copy()
    elif selected["unit"].nunique() > 1:
        raise ValueError("Selecione uma única unidade para consultar o histórico.")
    return selected


def filter_history(
    observations: pd.DataFrame, start: date, end: date,
    timezone: str = "America/Sao_Paulo",
) -> pd.DataFrame:
    if end < start:
        raise ValueError("A data final deve ser igual ou posterior à inicial.")
    if observations.empty:
        return observations.copy()
    first = pd.Timestamp(start).tz_localize(timezone).tz_convert("UTC")
    stop = pd.Timestamp(end + timedelta(days=1)).tz_localize(timezone).tz_convert("UTC")
    return observations.loc[
        (observations["timestamp"] >= first) & (observations["timestamp"] < stop)
    ].copy()


def forecast_history(forecast: Forecast, observations: pd.DataFrame, hours: int = 72) -> pd.DataFrame:
    history = region_history(observations, forecast.region, unit=forecast.unit)
    if history.empty:
        return history
    start = pd.Timestamp(forecast.issued_at.value - hours * 3_600_000_000_000, tz="UTC")
    return history.loc[
        (history["timestamp"] <= forecast.issued_at)
        & (history["timestamp"] > start)
    ].copy()


def history_summary(observations: pd.DataFrame) -> dict:
    if observations.empty:
        return {"count": 0, "mean_load": None, "peak_load": None, "missing_hours": 0}
    span_seconds = (observations["timestamp"].max() - observations["timestamp"].min()).total_seconds()
    expected = int(span_seconds / 3600) + 1
    scale = float(observations["observed_load"].max())
    mean = float((observations["observed_load"] / scale).mean()) * scale if scale else 0.0
    return {
        "count": len(observations),
        "mean_load": mean,
        "peak_load": float(observations["observed_load"].max()),
        "missing_hours": max(0, expected - len(observations)),
    }
