"""Vectorized analysis of supplied batch forecasts, preserving provenance.

This module consumes normalized archive rows, never executes archive content,
and does not create fictitious Forecast objects for aggregate-only results.
"""

from __future__ import annotations

from datetime import timedelta
import re
import unicodedata

import numpy as np
import pandas as pd

from contracts.forecast_schema import DATASET_SPLITS
from services.analysis_service import (
    COMPARISON_COLUMNS, GROUP_COLUMNS, LEAD_COLUMNS, LOCAL_TIMEZONE, MODEL_COLUMNS,
    NS_PER_HOUR, NS_PER_WEEK, ROW_COLUMNS, _calendar_frame, _cutoff, _empty_rows,
    _weather_frame,
)
from services.forecast_service import _utc_timestamp, normalize_observations
from services.metrics_service import calculate_metrics


RECALCULATED = "Recalculada dos pares"
REPORTED = "Resumo fornecido; sem pares detalhados"
NATIVE_COLUMNS = [
    "forecast_id", "issued_at", "timestamp", "forecast_step", "predicted_load",
    "model_name", "model_version", "model_label", "horizon_hours", "region", "unit", "dataset_split",
]


def _aware_column(values: pd.Series, name: str) -> pd.Series:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        if values.isna().any():
            raise ValueError(f"{name} contém timestamps inválidos.")
        return values.dt.tz_convert("UTC").dt.as_unit("ns")
    return pd.Series(
        pd.to_datetime([_utc_timestamp(value, name) for value in values], utc=True), index=values.index,
    )


def _validate_rows(rows: pd.DataFrame) -> None:
    if not set(NATIVE_COLUMNS).issubset(rows.columns):
        raise ValueError("O lote de previsões não contém todas as colunas canônicas necessárias.")
    if not rows["unit"].isin(("MW", "MWmed")).all():
        raise ValueError("As unidades devem ser MW ou MWmed, sem conversão implícita.")
    if not rows["dataset_split"].isin(DATASET_SPLITS).all():
        raise ValueError("O lote contém uma partição não reconhecida; não inferir a etapa pelas datas.")
    if not rows["horizon_hours"].isin((24, 48)).all():
        raise ValueError("O horizonte deve ser 24 ou 48 horas.")
    try:
        loads = rows["predicted_load"].to_numpy(dtype=float)
        steps = rows["forecast_step"].to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Cargas e passos devem ser números válidos.") from exc
    if not np.isfinite(loads).all() or (loads < 0).any():
        raise ValueError("As cargas previstas devem ser finitas e não negativas.")
    if not np.isfinite(steps).all() or (steps % 1 != 0).any() or (steps < 1).any():
        raise ValueError("Os passos devem ser inteiros positivos.")
    if (steps > rows["horizon_hours"].to_numpy()).any():
        raise ValueError("Um passo excede o horizonte declarado.")


def _lookup_loads(frame: pd.DataFrame, observations: pd.DataFrame, timestamps: pd.Series) -> np.ndarray:
    if observations.empty:
        return np.full(len(frame), np.nan)
    series = observations.set_index(["region", "unit", "timestamp"])["observed_load"]
    keys = pd.MultiIndex.from_arrays([frame["region"], frame["unit"], timestamps])
    return series.reindex(keys).to_numpy(dtype=float)


def _persistence_by_run(frame: pd.DataFrame, observations: pd.DataFrame) -> pd.Series:
    if observations.empty:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    origins = frame[["forecast_id", "issued_at", "region", "unit"]].drop_duplicates()
    if origins["forecast_id"].duplicated().any():
        raise ValueError("Uma execução tem metadados temporais ou unidades conflitantes.")
    right = observations.rename(columns={"timestamp": "reference_timestamp"})
    matched = pd.merge_asof(
        origins.sort_values("issued_at"), right.sort_values("reference_timestamp"),
        left_on="issued_at", right_on="reference_timestamp", by=["region", "unit"],
        tolerance=timedelta(hours=1), direction="backward", allow_exact_matches=True,
    )
    values = matched.set_index("forecast_id")["observed_load"]
    return frame["forecast_id"].map(values)


def prepare_archive_evaluation(
    rows: pd.DataFrame, observations: pd.DataFrame,
    weather: pd.DataFrame | None = None, calendar: pd.DataFrame | None = None,
    region: str | None = None, unit: str | None = None, as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize one long batch with exact, availability-aware contextual joins.

    Observations and both baselines are reconstructed from the canonical
    historical series. Source baseline values remain preserved in raw archive
    rows and are checked by the loader. No per-origin Python objects are made.
    """
    if rows.empty:
        return _empty_rows()
    _validate_rows(rows)
    selected = rows
    if region is not None:
        selected = selected.loc[selected["region"].eq(region)]
    if unit is not None:
        selected = selected.loc[selected["unit"].eq(unit)]
    if selected.empty:
        return _empty_rows()
    frame = selected[NATIVE_COLUMNS].copy().reset_index(drop=True)
    frame["issued_at"] = _aware_column(frame["issued_at"], "issued_at")
    frame["timestamp"] = _aware_column(frame["timestamp"], "timestamp")
    if frame.duplicated(["forecast_id", "timestamp"]).any() or frame.duplicated(["forecast_id", "forecast_step"]).any():
        raise ValueError("O lote contém alvos ou passos duplicados na mesma execução.")
    metadata_columns = [column for column in NATIVE_COLUMNS if column not in {"timestamp", "forecast_step", "predicted_load"}]
    metadata = frame[metadata_columns].drop_duplicates()
    if metadata["forecast_id"].duplicated().any():
        raise ValueError("Uma execução contém metadados conflitantes de modelo, emissão, horizonte, região, unidade ou partição.")
    steps = frame.groupby("forecast_id", observed=True, sort=False)["forecast_step"].agg(["size", "min", "max", "nunique"])
    expected = metadata.set_index("forecast_id")["horizon_hours"].reindex(steps.index)
    if steps["size"].ne(expected).any() or steps["nunique"].ne(expected).any() or steps["min"].ne(1).any() or steps["max"].ne(expected).any():
        raise ValueError("Cada execução deve conter exatamente 24 ou 48 alvos e todos os passos únicos de 1 ao horizonte.")
    if frame["timestamp"].le(frame["issued_at"]).any():
        raise ValueError("Os alvos devem ser posteriores à emissão.")
    elapsed_ns = frame["timestamp"].astype("int64") - frame["issued_at"].astype("int64")
    first_lag = elapsed_ns - (frame["forecast_step"].astype("int64") - 1) * NS_PER_HOUR
    if first_lag.le(0).any() or first_lag.gt(NS_PER_HOUR).any():
        raise ValueError("Passos, cadência horária e emissão do lote são inconsistentes.")
    if first_lag.groupby(frame["forecast_id"], observed=True).nunique().gt(1).any():
        raise ValueError("A cadência horária muda dentro de uma mesma execução.")
    frame["lead_hour"] = elapsed_ns / NS_PER_HOUR
    cutoff = _cutoff(as_of)
    observed = normalize_observations(observations)
    observed = observed.loc[observed["timestamp"].le(cutoff)]
    frame["observed_load"] = _lookup_loads(frame, observed, frame["timestamp"])
    frame["persistence_load"] = _persistence_by_run(frame, observed)
    references = pd.to_datetime(frame["timestamp"].astype("int64") - NS_PER_WEEK, utc=True)
    frame["seasonal_load"] = _lookup_loads(frame, observed, references)
    frame.loc[references.gt(frame["issued_at"]), "seasonal_load"] = np.nan
    frame["residual"] = frame["observed_load"] - frame["predicted_load"]
    frame["absolute_error"] = frame["residual"].abs()

    temperatures = _weather_frame(weather)
    temperatures = temperatures.loc[temperatures["timestamp"].le(cutoff)]
    if not temperatures.empty:
        temperatures = temperatures.groupby(["region", "timestamp"], observed=True, as_index=False).agg(
            temperature_c=("temperature_c", "mean"), temperature_station_count=("station", "nunique"),
        )
        frame = frame.merge(temperatures, on=["region", "timestamp"], how="left", validate="many_to_one")
        frame["temperature_station_count"] = frame["temperature_station_count"].fillna(0).astype(int)
    else:
        frame["temperature_c"] = np.nan
        frame["temperature_station_count"] = 0
    local = frame["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
    frame["date"] = local.dt.date
    frame["month"] = local.dt.month
    frame["weekday"] = local.dt.dayofweek
    labels = _calendar_frame(calendar)
    if not labels.empty:
        frame = frame.merge(labels, on=["region", "date"], how="left", validate="many_to_one")
        frame["day_type"] = frame["day_type"].fillna("nao_informado")
        frame["holiday_scope"] = frame["holiday_scope"].fillna("nao_informado")
        frame["event_name"] = frame["event_name"].fillna("")
    else:
        frame["day_type"], frame["holiday_scope"], frame["event_name"] = "nao_informado", "nao_informado", ""
    return frame[ROW_COLUMNS]


def _fast_metrics(frame: pd.DataFrame, prediction_column: str) -> dict:
    if frame.empty:
        return calculate_metrics(pd.DataFrame())
    actual = frame["observed_load"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all() or (actual < 0).any() or (prediction < 0).any():
        raise ValueError("Os pares devem conter cargas finitas e não negativas.")
    absolute = np.abs(actual - prediction)
    scale = float(absolute.max())
    normalized = absolute / scale if scale else absolute
    mae = float(normalized.mean() * scale) if scale else 0.0
    rmse = float(np.sqrt(np.mean(np.square(normalized))) * scale) if scale else 0.0
    nonzero = actual != 0
    mape = None
    if nonzero.any():
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratios = absolute[nonzero] / actual[nonzero]
        if np.isfinite(ratios).all():
            ratio_scale = float(ratios.max())
            mape = float(np.mean(ratios / ratio_scale)) * ratio_scale * 100 if ratio_scale else 0.0
            if not np.isfinite(mape):
                raise ValueError("O MAPE excede a capacidade numérica disponível.")
        else:
            # Only exceptional numerical ranges need the slower Decimal path.
            pairs = pd.DataFrame({"observed_load": actual, "predicted_load": prediction})
            mape = calculate_metrics(pairs)["mape"]
    return {
        "mae": mae, "rmse": rmse, "mape": mape, "n_pairs": int(len(frame)),
        "mape_valid_pairs": int(nonzero.sum()), "mape_excluded_zeros": int((~nonzero).sum()),
    }


def _context_sample(context: pd.DataFrame, common_sample: bool) -> tuple[pd.DataFrame, str, bool]:
    origins = context[["issued_at", "horizon_hours"]].drop_duplicates()
    counts = origins.groupby("issued_at", observed=True)["horizon_hours"].nunique()
    common = counts.index[counts.eq(2)]
    if len(common):
        selected = context.loc[context["issued_at"].isin(common)].copy()
        policy = "Emissões comuns entre 24 h e 48 h"
    elif context["horizon_hours"].nunique() > 1:
        selected = context.copy()
        policy = "Emissões disponíveis; sem emissões comuns entre 24 h e 48 h"
    else:
        selected = context.copy()
        policy = "Emissões disponíveis; somente um horizonte disponível"
    selected["_common_first"] = True
    comparison_verified = False
    if common_sample and len(common):
        models_count = len(selected[MODEL_COLUMNS].drop_duplicates())
        eligible = selected.loc[selected["forecast_step"].le(24)].dropna(
            subset=["observed_load", "predicted_load", "persistence_load", "seasonal_load"]
        )
        presence = eligible[["issued_at", "timestamp", *MODEL_COLUMNS]].drop_duplicates()
        pair_counts = presence.groupby(["issued_at", "timestamp"], observed=True).size()
        matching = pair_counts.loc[pair_counts.eq(models_count)].rename("_matching").reset_index()
        comparison_verified = not matching.empty
        selected = selected.drop(columns="_common_first").merge(
            matching.assign(_common_first=True).drop(columns="_matching"),
            on=["issued_at", "timestamp"], how="left", validate="many_to_one",
        )
        selected["_common_first"] = selected["_common_first"].eq(True)
    if not comparison_verified:
        policy += "; comparação entre horizontes não verificada"
    return selected, policy, comparison_verified


def fast_comparison_metrics(rows: pd.DataFrame, *, common_sample: bool = True) -> pd.DataFrame:
    """Same sample contract as comparison_metrics, with vectorized arithmetic."""
    if rows.empty:
        return pd.DataFrame(columns=[*COMPARISON_COLUMNS, "metric_origin"])
    if not rows["dataset_split"].isin(DATASET_SPLITS).all():
        raise ValueError("A comparação contém uma partição não reconhecida.")
    if not rows["unit"].isin(("MW", "MWmed")).all():
        raise ValueError("A comparação contém uma unidade não reconhecida.")
    results = []
    columns = [*GROUP_COLUMNS, *MODEL_COLUMNS, "issued_at", "timestamp", "forecast_step", "observed_load", "predicted_load", "persistence_load", "seasonal_load"]
    for context_key, context in rows[columns].groupby(GROUP_COLUMNS, observed=True, dropna=False, sort=True):
        available = context.groupby(MODEL_COLUMNS, observed=True, dropna=False)["issued_at"].nunique()
        selected, policy, comparison_verified = _context_sample(context, common_sample)
        for model_key, model_rows in selected.groupby(MODEL_COLUMNS, observed=True, dropna=False, sort=True):
            first = model_rows.loc[model_rows["forecast_step"].le(24)]
            segments = [("1–24 h", first)]
            if int(model_key[-1]) == 48:
                segments.extend([("25–48 h", model_rows.loc[model_rows["forecast_step"].gt(24)]), ("1–48 h", model_rows)])
            for segment, targets in segments:
                common = targets.dropna(subset=["observed_load", "predicted_load", "persistence_load", "seasonal_load"])
                if common_sample and segment == "1–24 h":
                    common = common.loc[common["_common_first"]]
                for method, column in (("Modelo", "predicted_load"), ("Persistência", "persistence_load"), ("Sazonal semanal", "seasonal_load")):
                    sample = common if common_sample else targets.dropna(subset=["observed_load", column])
                    metrics = _fast_metrics(sample, column)
                    results.append({
                        **dict(zip(GROUP_COLUMNS, context_key, strict=True)),
                        **dict(zip(MODEL_COLUMNS, model_key, strict=True)),
                        "segment": segment, "method": method, "total_targets": len(targets), **metrics,
                        "coverage": metrics["n_pairs"] / len(targets) if len(targets) else 0.0,
                        "n_issuances": int(targets["issued_at"].nunique()),
                        "available_issuances": int(available.loc[model_key]),
                        "comparison_verified": comparison_verified,
                        "sample_policy": policy + (
                            "; interseção de observados e ambos os baselines" if common_sample
                            else "; amostras disponíveis independentes por método"
                        ),
                        "metric_origin": RECALCULATED,
                    })
    return pd.DataFrame(results, columns=[*COMPARISON_COLUMNS, "metric_origin"])


def fast_error_by_lead(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate available pairs by declared split/model/unit and integer step."""
    if rows.empty:
        return pd.DataFrame(columns=LEAD_COLUMNS)
    if not rows["dataset_split"].isin(DATASET_SPLITS).all() or not rows["unit"].isin(("MW", "MWmed")).all():
        raise ValueError("A curva contém partição ou unidade não reconhecida.")
    available = rows.dropna(subset=["observed_load", "predicted_load"])
    keys = [*GROUP_COLUMNS, *MODEL_COLUMNS, "forecast_step"]
    result = []
    for values, frame in available.groupby(keys, observed=True, dropna=False, sort=True):
        result.append({
            **dict(zip(keys, values, strict=True)), **_fast_metrics(frame, "predicted_load"),
            "lead_hour_min": float(frame["lead_hour"].min()), "lead_hour_max": float(frame["lead_hour"].max()),
        })
    return pd.DataFrame(result).reindex(columns=LEAD_COLUMNS)


def _plain(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def _reported_stage(value: object) -> str:
    labels = {
        "validacao1s2025": "validation_2025", "validation2025": "validation_2025",
        "teste2s2025": "test_2025", "teste2025": "test_2025", "test2025": "test_2025",
        "forecast2026": "forecast_2026",
    }
    label = _plain(value)
    if label not in labels:
        raise ValueError(f"Etapa não reconhecida no resumo fornecido: {value}.")
    return labels[label]


def _reported_horizon(record: pd.Series) -> int:
    source = str(record.get("source_file", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()
    if "24h" in source:
        return 24
    if "48h" in source:
        return 48
    label = str(record.get("horizonte_modelo", ""))
    if label.startswith("24"):
        return 24
    if label.startswith("48"):
        return 48
    model = _plain(record.get("modelo", ""))
    if "24h" in model:
        return 24
    if "48h" in model:
        return 48
    raise ValueError("O resumo não identifica o horizonte do modelo.")


def normalize_reported_metrics(
    reported: pd.DataFrame, *, region: str, unit: str = "MWmed",
) -> pd.DataFrame:
    """Normalize producer summaries without inventing detailed target coverage.

    Source column names declare MWmed; summaries are never relabeled as MW.
    Summary counts remain reported counts. Actual target coverage, MAPE zero
    counts and available detailed issuances remain undefined (NaN).
    """
    extra = ["metric_origin", "source_file", "reported_model", "reported_origin_count", "reported_pair_count"]
    if reported.empty or unit != "MWmed":
        return pd.DataFrame(columns=[*COMPARISON_COLUMNS, *extra])
    required = {"etapa", "modelo", "mae_mwmed", "rmse_mwmed", "mape_pct"}
    if not required.issubset(reported.columns):
        raise ValueError("O resumo fornecido não contém o esquema esperado de métricas MWmed.")
    if "unit" in reported and not reported["unit"].dropna().eq("MWmed").all():
        raise ValueError("A unidade do resumo contradiz as colunas de métricas MWmed.")
    if "region" in reported and not reported["region"].dropna().eq(region).all():
        raise ValueError("A região do resumo contradiz o contexto declarado.")
    records = []
    for _, record in reported.iterrows():
        horizon = _reported_horizon(record)
        split = _reported_stage(record["etapa"])
        name = _plain(record["modelo"])
        if "integrado" in name:
            method = "Modelo"
        elif "persist" in name:
            method = "Persistência"
        elif "sazonal" in name:
            method = "Sazonal semanal"
        elif "piloto" in name:
            method = "Piloto"
        else:
            raise ValueError(f"Método não reconhecido no resumo: {record['modelo']}.")
        block = record.get("bloco_horas")
        if pd.isna(block):
            block = record.get("horizonte")
        normalized_block = str(block).replace("–", "-").replace(" ", "").lower()
        blocks = {"1-24h": "1–24 h", "25-48h": "25–48 h", "1-48h": "1–48 h"}
        if normalized_block not in blocks or (horizon == 24 and normalized_block != "1-24h"):
            raise ValueError(f"Bloco de horas não reconhecido no resumo: {block}.")
        numeric = {}
        for key, original in (("mae", "mae_mwmed"), ("rmse", "rmse_mwmed"), ("mape", "mape_pct")):
            value = float(record[original])
            if not np.isfinite(value) or value < 0:
                raise ValueError("O resumo contém métricas inválidas.")
            numeric[key] = value
        origins = record.get("n_origens")
        pairs = record.get("n_pares")
        if pd.isna(origins):
            origins = record.get("origens")
        if pd.isna(pairs):
            pairs = record.get("pares")
        if pd.isna(origins) or pd.isna(pairs) or float(origins) < 0 or float(pairs) < 0 or float(origins) % 1 or float(pairs) % 1:
            raise ValueError("As contagens fornecidas devem ser inteiros não negativos.")
        source = str(record.get("source_file", "Resumo fornecido"))
        model_name = (
            f"piloto_so_carga_{horizon}h" if method == "Piloto"
            else "integrado_24h" if horizon == 24 else "modelo_integrado_48h"
        )
        records.append({
            "region": region, "unit": "MWmed", "dataset_split": split,
            "model_name": model_name, "model_version": "não informada", "model_label": f"{horizon} h",
            "horizon_hours": horizon, "segment": blocks[normalized_block], "method": method,
            "total_targets": np.nan, "n_pairs": int(pairs), "coverage": np.nan, **numeric,
            "mape_valid_pairs": np.nan, "mape_excluded_zeros": np.nan,
            "n_issuances": int(origins), "available_issuances": np.nan,
            "comparison_verified": False,
            "sample_policy": "Resumo agregado do produtor; cobertura dos pares detalhados indisponível",
            "metric_origin": REPORTED, "source_file": source, "reported_model": record["modelo"],
            "reported_origin_count": int(origins), "reported_pair_count": int(pairs),
            "_priority": 2 if (f"{horizon}h" in source.lower()) else 1,
        })
    frame = pd.DataFrame(records)
    keys = [*GROUP_COLUMNS, "model_name", "horizon_hours", "segment", "method"]
    return frame.sort_values("_priority", ascending=False, kind="stable").drop_duplicates(keys).drop(columns="_priority").reset_index(drop=True)


def merge_reported_metrics(
    recomputed: pd.DataFrame, reported: pd.DataFrame, *, region: str | None = None, unit: str | None = None,
) -> pd.DataFrame:
    """Append aggregate-only records; never overwrite a recalculated metric.

    Only one geographic/unit context may be inferred. Callers with no detailed
    rows must explicitly provide their declared region/unit. Pilot summaries
    remain method='Piloto' and cannot create model curves or detailed targets.
    """
    result = recomputed.copy()
    if "metric_origin" not in result:
        result["metric_origin"] = RECALCULATED
    if "comparison_verified" not in result:
        result["comparison_verified"] = False
    if reported.empty:
        return result
    if region is None or unit is None:
        contexts = result[["region", "unit"]].drop_duplicates() if {"region", "unit"}.issubset(result.columns) else pd.DataFrame()
        if region is not None:
            contexts = contexts.loc[contexts["region"].eq(region)]
        if unit is not None:
            contexts = contexts.loc[contexts["unit"].eq(unit)]
        if len(contexts) != 1:
            raise ValueError("Informe explicitamente uma única região e unidade para associar os resumos.")
        region = str(contexts.iloc[0]["region"]) if region is None else region
        unit = str(contexts.iloc[0]["unit"]) if unit is None else unit
    summaries = normalize_reported_metrics(reported, region=region, unit=unit)
    if summaries.empty:
        return result
    keys = [*GROUP_COLUMNS, "model_name", "horizon_hours", "segment", "method"]
    if not result.empty:
        existing = result[keys].drop_duplicates().assign(_has_detailed=True)
        summaries = summaries.merge(existing, on=keys, how="left", validate="many_to_one")
        summaries = summaries.loc[summaries["_has_detailed"].ne(True)].drop(columns="_has_detailed")
    if summaries.empty:
        return result
    return pd.concat([result, summaries], ignore_index=True)
