"""Cross-model comparisons use matching origins, targets and available inputs."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.forecast_schema import Forecast
from services.analysis_service import (
    build_evaluation_rows,
    calendar_for_targets,
    comparison_metrics,
    error_by_lead,
    explorer_frame,
    forecast_identifier,
    largest_errors,
    temperature_at_issue,
)


REGION = "Município do Rio de Janeiro (teste)"
AS_OF = pd.Timestamp("2025-03-01T00:00:00Z")


def make_forecast(horizon=24, issued_at="2025-01-08T00:00:00Z", unit="MWmed", split="test_2025"):
    issue = pd.Timestamp(issued_at)
    first = issue.ceil("h") if issue.minute else pd.Timestamp(issue.to_pydatetime() + timedelta(hours=1))
    return Forecast(
        model_name=f"Modelo {horizon}", model_version="1.0", horizon_hours=horizon,
        region=REGION, unit=unit, issued_at=issue, dataset_split=split,
        predictions=pd.DataFrame({
            "timestamp": pd.date_range(first, periods=horizon, freq="h"),
            "predicted_load": np.arange(1, horizon + 1, dtype=float) + 100,
        }),
    )


def make_observations(forecast):
    rows = {forecast.issued_at: 100.0}
    for index, target in enumerate(forecast.predictions["timestamp"], start=1):
        rows[target] = 110.0 + index
        rows[pd.Timestamp(target.to_pydatetime() - timedelta(days=7))] = 70.0 + index
    return pd.DataFrame([
        {"timestamp": timestamp, "observed_load": value, "region": forecast.region, "unit": forecast.unit}
        for timestamp, value in sorted(rows.items())
    ])


def weather_frame(rows):
    return pd.DataFrame([
        {"timestamp": pd.Timestamp(timestamp), "temperature_c": temperature, "station": station, "region": region}
        for timestamp, temperature, station, region in rows
    ])


def test_rows_preserve_all_targets_and_original_origin():
    run = make_forecast()
    observations = make_observations(run)
    observations = observations.loc[observations["timestamp"] != run.predictions["timestamp"].iloc[5]]
    rows = build_evaluation_rows([run], observations, as_of=AS_OF)
    assert len(rows) == 24
    assert rows["observed_load"].notna().sum() == 23
    assert rows["forecast_step"].tolist() == list(range(1, 25))
    assert rows["lead_hour"].tolist() == list(range(1, 25))
    assert rows["issued_at"].eq(run.issued_at).all()
    assert rows["dataset_split"].eq("test_2025").all()
    assert rows["unit"].eq("MWmed").all()
    assert rows["residual"].dropna().eq(10).all()
    assert rows["persistence_load"].eq(100).all()
    assert rows["seasonal_load"].tolist() == list(range(71, 95))


def test_actuals_and_target_weather_after_cutoff_stay_unavailable():
    run = make_forecast()
    weather = weather_frame([
        ("2025-01-08T01:00Z", 25, "A", REGION),
        ("2025-01-08T02:00Z", 30, "A", REGION),
    ])
    rows = build_evaluation_rows([run], make_observations(run), weather=weather, as_of=pd.Timestamp("2025-01-08T01:00Z"))
    assert rows["observed_load"].notna().sum() == 1
    assert rows["temperature_c"].notna().sum() == 1
    assert rows["temperature_c"].iloc[0] == 25
    assert rows["temperature_station_count"].iloc[1] == 0


def test_baselines_never_use_post_issuance_loads():
    run = make_forecast(48)
    observations = make_observations(run)
    observations.loc[observations["timestamp"] > run.issued_at, "observed_load"] = 99999
    rows = build_evaluation_rows([run], observations, as_of=AS_OF)
    assert rows["persistence_load"].eq(100).all()
    assert rows["seasonal_load"].tolist() == list(range(71, 119))
    assert rows["observed_load"].eq(99999).all()


def test_persistence_accepts_last_observation_within_one_hour_only():
    run = make_forecast()
    observations = make_observations(run)
    observations = observations.loc[observations["timestamp"] != run.issued_at].copy()
    row = {"timestamp": pd.Timestamp("2025-01-07T23:00:00Z"), "observed_load": 90.0, "region": REGION, "unit": "MWmed"}
    observations = pd.concat([observations, pd.DataFrame([row])], ignore_index=True)
    rows = build_evaluation_rows([run], observations, as_of=AS_OF)
    assert rows["persistence_load"].eq(90).all()
    observations.loc[observations["timestamp"] == row["timestamp"], "timestamp"] = pd.Timestamp("2025-01-07T22:59:59Z")
    assert build_evaluation_rows([run], observations, as_of=AS_OF)["persistence_load"].isna().all()


def test_weekly_baseline_is_exact_and_does_not_fill_missing_reference():
    run = make_forecast()
    observations = make_observations(run)
    reference = pd.Timestamp("2025-01-01T01:00:00Z")
    observations = observations.loc[observations["timestamp"] != reference]
    rows = build_evaluation_rows([run], observations, as_of=AS_OF)
    assert pd.isna(rows["seasonal_load"].iloc[0])
    assert rows["seasonal_load"].iloc[1] == 72


def test_weekly_baseline_preserves_nanosecond_identity():
    run = make_forecast()
    targets = run.predictions.copy()
    targets["timestamp"] = pd.date_range("2025-01-08T01:00:00.000000001Z", periods=24, freq="h")
    run = replace(run, issued_at=pd.Timestamp("2025-01-08T00:00:00.000000001Z"), predictions=targets)
    observations = pd.DataFrame([
        {"timestamp": pd.Timestamp("2025-01-01T01:00:00.000000001Z"), "observed_load": 70.0, "region": REGION, "unit": "MWmed"},
        {"timestamp": pd.Timestamp("2025-01-01T01:00:00Z"), "observed_load": 999.0, "region": REGION, "unit": "MWmed"},
    ])
    rows = build_evaluation_rows([run], observations, as_of=AS_OF)
    assert rows["seasonal_load"].iloc[0] == 70
    assert rows["seasonal_load"].iloc[1:].isna().all()


def test_baseline_and_weather_tolerance_reject_one_nanosecond_beyond_hour():
    run = make_forecast()
    timestamp = pd.Timestamp("2025-01-07T22:59:59.999999999Z")
    observations = pd.DataFrame([{"timestamp": timestamp, "observed_load": 70.0, "region": REGION, "unit": "MWmed"}])
    rows = build_evaluation_rows([run], observations, as_of=AS_OF)
    assert rows["persistence_load"].isna().all()
    weather = weather_frame([(timestamp, 25, "A", REGION)])
    assert temperature_at_issue(weather, run.issued_at, REGION, as_of=AS_OF) is None


def test_forecast_id_is_stable_across_file_names_and_duplicate_copies():
    run = make_forecast()
    copy = replace(run, source_path=Path("outro_nome.json"))
    assert forecast_identifier(run) == forecast_identifier(copy)
    rows = build_evaluation_rows([run, copy], make_observations(run), as_of=AS_OF)
    assert len(rows) == 24


def test_conflicting_duplicate_run_identity_is_refused():
    run = make_forecast()
    changed = run.predictions.copy()
    changed["predicted_load"] += 1
    conflict = replace(run, predictions=changed)
    with pytest.raises(ValueError, match="conflitantes"):
        build_evaluation_rows([run, conflict], make_observations(run), as_of=AS_OF)


def test_units_are_exact_and_never_converted_or_merged():
    mwmed, mw = make_forecast(unit="MWmed"), make_forecast(unit="MW")
    observations = pd.concat([make_observations(mwmed), make_observations(mw)], ignore_index=True)
    observations.loc[observations["unit"].eq("MW"), "observed_load"] += 1000
    rows = build_evaluation_rows([mwmed, mw], observations, as_of=AS_OF)
    assert rows.loc[rows["unit"].eq("MWmed"), "residual"].eq(10).all()
    assert rows.loc[rows["unit"].eq("MW"), "residual"].eq(1010).all()
    summary = comparison_metrics(rows)
    assert set(summary["unit"]) == {"MW", "MWmed"}
    selected = build_evaluation_rows([mwmed, mw], observations, unit="MWmed", as_of=AS_OF)
    assert len(selected) == 24
    assert selected["unit"].eq("MWmed").all()


def test_comparison_segments_and_methods_have_matching_sample_sizes():
    runs = [make_forecast(24), make_forecast(48)]
    observations = make_observations(runs[1])
    rows = build_evaluation_rows(runs, observations, as_of=AS_OF)
    metrics = comparison_metrics(rows)
    assert len(metrics) == 12
    assert set(metrics["method"]) == {"Modelo", "Persistência", "Sazonal semanal"}
    for (horizon, segment), group in metrics.groupby(["horizon_hours", "segment"]):
        count = 48 if horizon == 48 and segment == "1–48 h" else 24
        assert group["n_pairs"].eq(count).all()
        assert group["total_targets"].eq(count).all()
        assert group["coverage"].eq(1).all()
    assert metrics["sample_policy"].str.contains("Emissões comuns").all()
    model = metrics.loc[metrics["method"].eq("Modelo")]
    assert model["mae"].eq(10).all()
    assert model["rmse"].eq(10).all()


def test_common_sample_intersects_observed_and_both_baselines():
    runs = [make_forecast(24), make_forecast(48)]
    observations = make_observations(runs[1])
    excluded = [pd.Timestamp("2025-01-08T02:00Z"), pd.Timestamp("2025-01-01T03:00Z")]
    observations = observations.loc[~observations["timestamp"].isin(excluded)]
    rows = build_evaluation_rows(runs, observations, as_of=AS_OF)
    first = comparison_metrics(rows).query("segment == '1–24 h'")
    assert first["n_pairs"].eq(22).all()
    assert first["total_targets"].eq(24).all()
    assert first["coverage"].eq(22 / 24).all()
    independent = comparison_metrics(rows, common_sample=False).query("segment == '1–24 h'")
    assert independent.loc[independent["method"].eq("Modelo"), "n_pairs"].eq(23).all()
    assert independent.loc[independent["method"].eq("Sazonal semanal"), "n_pairs"].eq(22).all()


def test_comparison_uses_only_common_issuances_when_present():
    first24, first48 = make_forecast(24), make_forecast(48)
    unpaired = make_forecast(24, issued_at="2025-01-09T00:00Z")
    rows = build_evaluation_rows([first24, first48, unpaired], make_observations(first48), as_of=AS_OF)
    metrics = comparison_metrics(rows)
    h24 = metrics.loc[metrics["horizon_hours"].eq(24)]
    assert h24["available_issuances"].eq(2).all()
    assert h24["n_issuances"].eq(1).all()
    assert h24["total_targets"].eq(24).all()


def test_nonoverlapping_issuances_are_explicit_and_splits_do_not_merge():
    runs = [make_forecast(24), make_forecast(48, issued_at="2025-01-09T00:00Z")]
    observations = make_observations(runs[1])
    rows = build_evaluation_rows(runs, observations, as_of=AS_OF)
    metrics = comparison_metrics(rows)
    assert metrics["sample_policy"].str.contains("sem emissões comuns").all()
    assert not metrics["comparison_verified"].any()
    validation = replace(runs[0], dataset_split="validation_2025")
    rows = build_evaluation_rows([*runs, validation], observations, as_of=AS_OF)
    assert set(comparison_metrics(rows)["dataset_split"]) == {"validation_2025", "test_2025"}


def test_comparison_verification_requires_common_valid_pairs_and_common_mode():
    runs = [make_forecast(24), make_forecast(48)]
    rows = build_evaluation_rows(runs, make_observations(runs[1]), as_of=AS_OF)
    assert comparison_metrics(rows)["comparison_verified"].all()
    assert not comparison_metrics(rows, common_sample=False)["comparison_verified"].any()
    rows["seasonal_load"] = np.nan
    assert not comparison_metrics(rows)["comparison_verified"].any()


def test_common_sample_without_baselines_has_undefined_metrics_and_zero_coverage():
    run = make_forecast()
    observations = make_observations(run)
    observations = observations.loc[observations["timestamp"] > run.issued_at]
    metrics = comparison_metrics(build_evaluation_rows([run], observations, as_of=AS_OF))
    assert metrics["n_pairs"].eq(0).all()
    assert metrics["coverage"].eq(0).all()
    assert metrics["mae"].isna().all()


def test_fractional_elapsed_lead_keeps_integer_steps_and_correct_segments():
    run = make_forecast(48, issued_at="2025-01-08T00:15:00Z")
    rows = build_evaluation_rows([run], make_observations(run), as_of=AS_OF)
    assert rows["forecast_step"].iloc[24] == 25
    assert rows["lead_hour"].iloc[24] == 24.75
    metrics = comparison_metrics(rows)
    second_day = metrics.loc[metrics["segment"].eq("25–48 h")]
    assert second_day["n_pairs"].eq(24).all()
    errors = error_by_lead(rows)
    assert len(errors) == 48
    assert errors["forecast_step"].tolist() == list(range(1, 49))
    assert errors["lead_hour_min"].iloc[24] == 24.75


def test_error_by_step_aggregates_multiple_origins_without_mixing_split_or_unit():
    run = make_forecast()
    rows = build_evaluation_rows([run], make_observations(run), as_of=AS_OF)
    extra = rows.copy()
    extra["issued_at"] = extra["issued_at"].map(lambda stamp: pd.Timestamp(stamp.to_pydatetime() + timedelta(days=1)))
    extra["forecast_id"] = "another"
    errors = error_by_lead(pd.concat([rows, extra], ignore_index=True))
    assert errors["n_pairs"].eq(2).all()
    assert errors["mae"].eq(10).all()


def test_error_curve_retains_actual_mape_sample_count_with_zero_observations():
    run = make_forecast()
    rows = build_evaluation_rows([run], make_observations(run), as_of=AS_OF)
    first = rows.iloc[[0]].copy()
    extra = first.copy()
    extra["forecast_id"] = "another"
    extra["issued_at"] = pd.Timestamp("2025-01-09T00:00Z")
    extra["observed_load"] = 0.0
    errors = error_by_lead(pd.concat([first, extra], ignore_index=True))
    assert errors["n_pairs"].iloc[0] == 2
    assert errors["mape_valid_pairs"].iloc[0] == 1
    assert errors["mape_excluded_zeros"].iloc[0] == 1


def test_calendar_joins_local_dates_and_never_infers_labels():
    run = make_forecast()
    calendar = pd.DataFrame([{
        "date": date(2025, 1, 7), "region": REGION, "day_type": "feriado",
        "holiday_scope": "municipal", "event_name": "Evento sintético informado",
    }])
    labels = calendar_for_targets(run, calendar)
    assert labels["day_type"].iloc[0] == "feriado"
    assert labels["holiday_scope"].iloc[0] == "municipal"
    assert labels["event_name"].iloc[0] == "Evento sintético informado"
    assert labels["day_type"].iloc[2] == "nao_informado"
    assert calendar_for_targets(run, None)["day_type"].eq("nao_informado").all()


def test_target_weather_aggregates_exact_timestamp_and_reports_stations():
    run = make_forecast()
    weather = weather_frame([
        ("2025-01-08T01:00Z", 20, "A", REGION),
        ("2025-01-08T01:00Z", 30, "B", REGION),
        ("2025-01-08T01:00Z", 80, "C", "Outra região"),
        ("2025-01-08T02:30Z", 100, "A", REGION),
    ])
    rows = build_evaluation_rows([run], make_observations(run), weather=weather, as_of=AS_OF)
    assert rows["temperature_c"].iloc[0] == 25
    assert rows["temperature_station_count"].iloc[0] == 2
    assert pd.isna(rows["temperature_c"].iloc[1])


def test_issue_temperature_only_uses_last_available_station_values_within_one_hour():
    weather = weather_frame([
        ("2025-01-07T23:30Z", 20, "A", REGION),
        ("2025-01-07T23:50Z", 30, "B", REGION),
        ("2025-01-07T22:00Z", 0, "C", REGION),
        ("2025-01-08T00:01Z", 100, "A", REGION),
    ])
    result = temperature_at_issue(weather, pd.Timestamp("2025-01-08T00:00Z"), REGION, as_of=AS_OF)
    assert result["temperature_c"] == pytest.approx(25)
    assert result["station_count"] == 2
    assert result["stations"] == ["A", "B"]
    assert result["timestamp"] == pd.Timestamp("2025-01-07T23:50Z")
    assert result["oldest_timestamp"] == pd.Timestamp("2025-01-07T23:30Z")
    selected = temperature_at_issue(weather, pd.Timestamp("2025-01-08T00:00Z"), REGION, station="A", as_of=AS_OF)
    assert selected["temperature_c"] == 20
    assert temperature_at_issue(weather, pd.Timestamp("2025-01-08T02:00Z"), REGION, as_of=AS_OF) is None


def test_issue_temperature_cannot_show_a_weather_observation_later_than_as_of():
    weather = weather_frame([
        ("2025-01-07T23:30Z", 20, "A", REGION),
        ("2025-01-07T23:50Z", 30, "B", REGION),
    ])
    result = temperature_at_issue(weather, pd.Timestamp("2025-01-08T00:00Z"), REGION, as_of=pd.Timestamp("2025-01-07T23:40Z"))
    assert result["station_count"] == 1
    assert result["temperature_c"] == 20


def test_error_filters_use_local_month_weekday_calendar_and_temperature():
    run = make_forecast(24, issued_at="2024-12-31T23:00Z")
    weather = weather_frame([(target, 25, "A", REGION) for target in run.predictions["timestamp"]])
    calendar = pd.DataFrame([{
        "date": date(2024, 12, 31), "region": REGION, "day_type": "dia_util",
        "holiday_scope": "nenhum", "event_name": "",
    }])
    rows = build_evaluation_rows([run], make_observations(run), weather, calendar, as_of=AS_OF)
    filtered = largest_errors(rows, month=12, weekday=1, day_types=["dia_util"], temp_range=(24, 26))
    assert len(filtered) == 3
    assert filtered["month"].eq(12).all()
    assert filtered["weekday"].eq(1).all()
    assert len(largest_errors(rows, limit=2)) == 2
    assert largest_errors(rows, temp_range=(30, 40)).empty


def test_explorer_observed_future_targets_remain_nan_until_cutoff():
    run = make_forecast()
    frame = explorer_frame(run, make_observations(run), as_of=run.issued_at)
    targets = frame.loc[frame["phase"].eq("previsão")]
    history = frame.loc[frame["phase"].eq("histórico")]
    assert len(targets) == 24
    assert targets["observed_load"].isna().all()
    assert targets["seasonal_load"].tolist() == list(range(71, 95))
    assert history["timestamp"].le(run.issued_at).all()
    assert history["predicted_load"].isna().all()


@pytest.mark.parametrize("hours", [0, 24, 47, 169])
def test_explorer_rejects_context_outside_48_to_168_hours(hours):
    run = make_forecast()
    with pytest.raises(ValueError, match="48 e 168"):
        explorer_frame(run, make_observations(run), history_hours=hours)


def test_duplicate_weather_and_calendar_are_refused():
    run = make_forecast()
    weather = weather_frame([("2025-01-08T01:00Z", 20, "A", REGION)] * 2)
    with pytest.raises(ValueError, match="duplicadas"):
        build_evaluation_rows([run], make_observations(run), weather=weather, as_of=AS_OF)
    calendar = pd.DataFrame([{
        "date": date(2025, 1, 8), "region": REGION, "day_type": "dia_util", "holiday_scope": "nenhum", "event_name": "",
    }] * 2)
    with pytest.raises(ValueError, match="duplicadas"):
        calendar_for_targets(run, calendar)


def test_missing_sources_have_honest_empty_outputs():
    rows = build_evaluation_rows([], pd.DataFrame())
    assert rows.empty
    assert {"forecast_step", "observed_load", "day_type"}.issubset(rows.columns)
    assert comparison_metrics(rows).empty
    assert error_by_lead(rows).empty
    assert largest_errors(rows).empty
    assert temperature_at_issue(None, pd.Timestamp("2025-01-08T00:00Z"), REGION) is None
