"""Batch analysis agrees with per-run services and preserves aggregate provenance."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from services.analysis_service import build_evaluation_rows, comparison_metrics, error_by_lead
from services.archive_analysis import (
    RECALCULATED, REPORTED, fast_comparison_metrics, fast_error_by_lead,
    merge_reported_metrics, normalize_reported_metrics, prepare_archive_evaluation,
)
from tests.test_analysis_service import AS_OF, REGION, make_forecast


@pytest.fixture
def batch_sources():
    forecasts = [
        make_forecast(horizon, issued_at=f"2025-01-{day:02d}T00:15:00Z")
        for day in (8, 10, 12) for horizon in (24, 48)
    ]
    timestamps = pd.date_range("2024-12-25T00:00:00Z", "2025-01-31T00:00:00Z", freq="h")
    observations = pd.DataFrame({
        "timestamp": timestamps, "observed_load": 1000 + 200 * np.sin(np.arange(len(timestamps)) / 24),
        "region": REGION, "unit": "MWmed",
    })
    rows = build_evaluation_rows(forecasts, observations, as_of=AS_OF)
    assert len(rows) == 216
    return forecasts, observations, rows


def sort_metrics(frame):
    return frame.sort_values(["dataset_split", "model_name", "horizon_hours", "segment", "method"]).reset_index(drop=True)


def test_vectorized_batch_context_matches_per_forecast_services(batch_sources):
    _, observations, expected = batch_sources
    actual = prepare_archive_evaluation(expected, observations, as_of=AS_OF)
    columns = ["forecast_id", "timestamp", "issued_at", "forecast_step", "lead_hour", "predicted_load", "observed_load", "residual", "persistence_load", "seasonal_load"]
    pd.testing.assert_frame_equal(actual[columns], expected[columns], check_dtype=False)


@pytest.mark.parametrize("common_sample", [True, False])
def test_fast_comparison_equals_existing_method_for_216_targets(batch_sources, common_sample):
    _, _, rows = batch_sources
    expected = sort_metrics(comparison_metrics(rows, common_sample=common_sample))
    actual = sort_metrics(fast_comparison_metrics(rows, common_sample=common_sample))
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    assert actual["metric_origin"].eq(RECALCULATED).all()


def test_fast_step_curves_equal_existing_method_without_fractional_step_alias(batch_sources):
    _, _, rows = batch_sources
    expected = error_by_lead(rows)
    actual = fast_error_by_lead(rows)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    assert actual["forecast_step"].between(1, 48).all()
    assert actual["lead_hour_min"].iloc[0] == 0.75


def test_fast_common_sample_matches_existing_method_with_partial_cross_model_coverage(batch_sources):
    _, _, rows = batch_sources
    rows = rows.copy()
    first_issue = rows["issued_at"].min()
    rows.loc[rows["issued_at"].eq(first_issue) & rows["horizon_hours"].eq(24) & rows["forecast_step"].eq(2), "observed_load"] = np.nan
    rows.loc[rows["issued_at"].eq(first_issue) & rows["horizon_hours"].eq(48) & rows["forecast_step"].eq(3), "seasonal_load"] = np.nan
    expected = sort_metrics(comparison_metrics(rows, common_sample=True))
    actual = sort_metrics(fast_comparison_metrics(rows, common_sample=True))
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    first_day = actual.loc[actual["segment"].eq("1–24 h")]
    assert first_day["n_pairs"].eq(70).all()
    assert first_day["total_targets"].eq(72).all()
    assert first_day["coverage"].eq(70 / 72).all()


def test_future_actuals_and_weather_are_excluded_but_known_baselines_remain(batch_sources):
    _, observations, rows = batch_sources
    rows = rows.loc[rows["issued_at"].eq(pd.Timestamp("2025-01-08T00:15Z"))]
    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-08T01:00Z", "2025-01-08T02:00Z"]),
        "temperature_c": [25.0, 30.0], "station": "A", "region": REGION,
    })
    actual = prepare_archive_evaluation(rows, observations, weather=weather, as_of=pd.Timestamp("2025-01-08T01:00Z"))
    assert actual["observed_load"].notna().sum() == 2  # one target per horizon
    assert actual["temperature_c"].notna().sum() == 2
    assert actual["seasonal_load"].notna().all()
    assert actual["persistence_load"].notna().all()


def test_archive_baselines_are_reconstructed_not_trusted_from_unverified_columns(batch_sources):
    _, observations, rows = batch_sources
    source = rows.copy()
    source["persistence_load"] = 9999999
    source["seasonal_load"] = 9999999
    actual = prepare_archive_evaluation(source, observations, as_of=AS_OF)
    assert not actual["persistence_load"].eq(9999999).any()
    assert not actual["seasonal_load"].eq(9999999).any()
    pd.testing.assert_series_equal(actual["persistence_load"], rows["persistence_load"], check_names=False)


def test_archive_calendar_uses_explicit_unknown_scope_and_local_day(batch_sources):
    _, observations, rows = batch_sources
    calendar = pd.DataFrame([{
        "date": date(2025, 1, 7), "region": REGION, "day_type": "ponto_facultativo",
        "holiday_scope": "nao_informado", "event_name": "Evento fornecido",
    }])
    actual = prepare_archive_evaluation(rows, observations, calendar=calendar, as_of=AS_OF)
    first = actual.loc[actual["timestamp"].eq(pd.Timestamp("2025-01-08T01:00Z"))]
    assert first["day_type"].eq("ponto_facultativo").all()
    assert first["holiday_scope"].eq("nao_informado").all()


def test_archive_analysis_never_mixes_units_or_regions(batch_sources):
    _, observations, rows = batch_sources
    other = rows.copy()
    other["unit"] = "MW"
    other["forecast_id"] = other["forecast_id"] + "_mw"
    other_observations = observations.copy()
    other_observations["unit"] = "MW"
    other_observations["observed_load"] += 500
    actual = prepare_archive_evaluation(pd.concat([rows, other], ignore_index=True), pd.concat([observations, other_observations], ignore_index=True), as_of=AS_OF)
    first = actual.loc[actual["unit"].eq("MWmed"), "residual"].to_numpy()
    second = actual.loc[actual["unit"].eq("MW"), "residual"].to_numpy()
    assert np.allclose(second - first, 500)
    assert set(fast_comparison_metrics(actual)["unit"]) == {"MW", "MWmed"}
    only = prepare_archive_evaluation(pd.concat([rows, other], ignore_index=True), observations, unit="MWmed", as_of=AS_OF)
    assert only["unit"].eq("MWmed").all()


@pytest.mark.parametrize("stage", ["teste", "2025", "validação não especificada"])
def test_unknown_archive_stage_is_rejected(batch_sources, stage):
    _, observations, rows = batch_sources
    rows = rows.copy()
    rows["dataset_split"] = stage
    with pytest.raises(ValueError, match="partição"):
        prepare_archive_evaluation(rows, observations, as_of=AS_OF)
    with pytest.raises(ValueError, match="partição"):
        fast_comparison_metrics(rows)


def test_fast_metrics_zero_load_policy_and_large_finite_values(batch_sources):
    _, _, rows = batch_sources
    selected = rows.iloc[:2].copy()
    selected["observed_load"] = [0.0, 100.0]
    selected["predicted_load"] = [10.0, 90.0]
    selected["persistence_load"] = [10.0, 90.0]
    selected["seasonal_load"] = [10.0, 90.0]
    metrics = fast_comparison_metrics(selected)
    assert metrics["mae"].eq(10).all()
    assert metrics["mape"].eq(10).all()
    assert metrics["mape_valid_pairs"].eq(1).all()
    assert metrics["mape_excluded_zeros"].eq(1).all()
    selected["observed_load"] = 1e308
    selected[["predicted_load", "persistence_load", "seasonal_load"]] = 0.0
    with np.errstate(over="raise", invalid="raise"):
        large = fast_comparison_metrics(selected)
    assert large["mae"].eq(1e308).all()
    assert large["rmse"].eq(1e308).all()
    assert large["mape"].eq(100).all()


def test_fast_metrics_mape_undefined_when_all_actuals_zero(batch_sources):
    _, _, rows = batch_sources
    selected = rows.iloc[:2].copy()
    selected["observed_load"] = 0.0
    metrics = fast_comparison_metrics(selected)
    assert metrics["mape"].isna().all()
    assert metrics["mape_excluded_zeros"].eq(2).all()


def supplied_summary(stage="Validação 1S2025", model="Modelo integrado 24 h", **extra):
    return pd.DataFrame([{
        "etapa": stage, "modelo": model, "bloco_horas": "1-24 h", "n_origens": 100,
        "n_pares": 2400, "mae_mwmed": 11.11, "rmse_mwmed": 22.22, "mape_pct": 3.33,
        "horizonte_modelo": "24 h (Tales)", "source_file": "painel/resumo_metricas.csv", **extra,
    }])


def test_reported_validation_has_metrics_and_provenance_but_no_fake_coverage(batch_sources):
    _, _, rows = batch_sources
    recomputed = fast_comparison_metrics(rows)
    merged = merge_reported_metrics(recomputed, supplied_summary())
    provided = merged.loc[merged["dataset_split"].eq("validation_2025")]
    assert len(provided) == 1
    assert provided["metric_origin"].eq(REPORTED).all()
    assert provided["mae"].iloc[0] == 11.11
    assert provided["n_pairs"].iloc[0] == 2400
    assert provided["coverage"].isna().all()
    assert provided["total_targets"].isna().all()
    assert provided["mape_valid_pairs"].isna().all()
    assert not rows["dataset_split"].eq("validation_2025").any()


def test_reported_test_never_overwrites_recalculated_model_metrics(batch_sources):
    _, _, rows = batch_sources
    # Match the archive's actual source model identifier.
    rows = rows.copy()
    rows.loc[rows["horizon_hours"].eq(24), "model_name"] = "integrado_24h"
    recomputed = fast_comparison_metrics(rows)
    merged = merge_reported_metrics(recomputed, supplied_summary(stage="Teste 2S2025", mae_mwmed=99999))
    assert len(merged) == len(recomputed)
    pd.testing.assert_frame_equal(merged, recomputed)


def test_pilot_summaries_remain_separate_without_fake_detailed_rows(batch_sources):
    _, _, rows = batch_sources
    merged = merge_reported_metrics(fast_comparison_metrics(rows), supplied_summary(model="Piloto só-carga 24 h"))
    pilot = merged.loc[merged["method"].eq("Piloto")]
    assert len(pilot) == 1
    assert pilot["metric_origin"].eq(REPORTED).all()
    assert "piloto" in pilot["model_name"].iloc[0]
    assert not rows["model_name"].str.contains("piloto").any()


def test_precision_specific_summary_wins_over_global_copy():
    global_summary = supplied_summary()
    precise = supplied_summary(mae_mwmed=11.1111, source_file="painel/resumo_metricas_24h.csv")
    normalized = normalize_reported_metrics(pd.concat([global_summary, precise], ignore_index=True), region=REGION)
    assert len(normalized) == 1
    assert normalized["mae"].iloc[0] == 11.1111
    assert normalized["source_file"].iloc[0].endswith("_24h.csv")


def test_reported_mwmed_is_not_relabelled_as_mw(batch_sources):
    _, _, rows = batch_sources
    actual = fast_comparison_metrics(rows)
    actual["unit"] = "MW"
    merged = merge_reported_metrics(actual, supplied_summary())
    pd.testing.assert_frame_equal(merged, actual)
    assert normalize_reported_metrics(supplied_summary(), region=REGION, unit="MW").empty


def test_explicit_reported_metadata_cannot_be_silently_replaced():
    with pytest.raises(ValueError, match="unidade"):
        normalize_reported_metrics(supplied_summary(unit="MW"), region=REGION)
    with pytest.raises(ValueError, match="região"):
        normalize_reported_metrics(supplied_summary(region="Outra região"), region=REGION)


def test_reported_summary_requires_explicit_context_when_no_details_exist():
    with pytest.raises(ValueError, match="região e unidade"):
        merge_reported_metrics(pd.DataFrame(), supplied_summary())
    merged = merge_reported_metrics(pd.DataFrame(), supplied_summary(), region=REGION, unit="MWmed")
    assert len(merged) == 1
    assert merged["region"].iloc[0] == REGION
    assert merged["metric_origin"].iloc[0] == REPORTED


def test_unknown_reported_stage_is_not_inferred_from_year():
    with pytest.raises(ValueError, match="Etapa não reconhecida"):
        normalize_reported_metrics(supplied_summary(stage="qualquer2025"), region=REGION)


def test_archive_invalid_naive_time_or_duplicate_target_is_refused(batch_sources):
    _, observations, rows = batch_sources
    naive = rows.copy()
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="fuso"):
        prepare_archive_evaluation(naive, observations, as_of=AS_OF)
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicados"):
        prepare_archive_evaluation(duplicate, observations, as_of=AS_OF)


def test_partial_23_target_run_is_rejected_before_it_can_receive_full_coverage(batch_sources):
    _, observations, rows = batch_sources
    run_id = rows["forecast_id"].iloc[0]
    partial = rows.loc[~(rows["forecast_id"].eq(run_id) & rows["forecast_step"].eq(24))]
    with pytest.raises(ValueError, match="exatamente 24 ou 48"):
        prepare_archive_evaluation(partial, observations, as_of=AS_OF)


@pytest.mark.parametrize("column,value", [
    ("model_name", "Outro modelo"), ("model_version", "outra versão"),
    ("model_label", "Outro rótulo"), ("dataset_split", "forecast_2026"),
    ("horizon_hours", 48), ("region", "Outra região"), ("unit", "MW"),
    ("issued_at", pd.Timestamp("2025-01-08T00:16:00Z")),
])
def test_batch_analysis_refuses_conflicting_metadata_within_one_run(batch_sources, column, value):
    _, observations, rows = batch_sources
    invalid = rows.copy()
    invalid.loc[0, column] = value
    with pytest.raises(ValueError, match="metadados conflitantes"):
        prepare_archive_evaluation(invalid, observations, as_of=AS_OF)


def test_fast_comparison_without_common_origins_is_explicitly_unverified(batch_sources):
    _, _, rows = batch_sources
    first_issue = rows["issued_at"].min()
    rows = rows.loc[(rows["horizon_hours"].eq(24) & rows["issued_at"].eq(first_issue)) | (rows["horizon_hours"].eq(48) & rows["issued_at"].ne(first_issue))]
    summary = fast_comparison_metrics(rows)
    assert not summary["comparison_verified"].any()
    assert summary["sample_policy"].str.contains("comparação entre horizontes não verificada").all()
    assert summary["n_pairs"].gt(0).all()


def test_fast_mape_keeps_near_zero_positive_values_and_curve_sample_counts(batch_sources):
    _, _, rows = batch_sources
    selected = rows.iloc[:2].copy()
    selected["observed_load"] = [1e-12, 2.0]
    selected[["predicted_load", "persistence_load", "seasonal_load"]] = [[2e-12] * 3, [1.0] * 3]
    summary = fast_comparison_metrics(selected)
    assert summary["mape"].eq(75.0).all()
    assert summary["mape_valid_pairs"].eq(2).all()
    selected.loc[selected.index[0], "observed_load"] = 0.0
    curve = fast_error_by_lead(selected)
    zero = curve.loc[curve["forecast_step"].eq(selected["forecast_step"].iloc[0])].iloc[0]
    assert zero["n_pairs"] == 1
    assert zero["mape_valid_pairs"] == 0
    assert zero["mape_excluded_zeros"] == 1
    assert pd.isna(zero["mape"])


def test_archive_empty_batches_do_not_invent_metrics():
    assert prepare_archive_evaluation(pd.DataFrame(), pd.DataFrame()).empty
    assert fast_comparison_metrics(pd.DataFrame()).empty
    assert fast_error_by_lead(pd.DataFrame()).empty
