"""Guarda territorial não depende de rede nem da presença do ZIP particular."""

from copy import deepcopy

import pandas as pd
import pytest

from services.data_loader import DataBundle
from contracts.forecast_schema import Forecast
from services.geography_service import (
    KNOWN_REGIONAL_ARCHIVE_SHA256, TARGET_REGION, geographic_status, municipal_bundle,
)


def bundle(region=TARGET_REGION, *, mode="provided", archive_hash=None):
    timestamp = pd.Timestamp("2024-01-01T03:00:00Z")
    provenance = {"region": region, "region_source": "informado pelo grupo"} if region else {}
    if archive_hash:
        provenance["archive_sha256"] = archive_hash
    return DataBundle(
        mode, [], pd.DataFrame({"timestamp": [timestamp], "observed_load": [35000.0], "region": [region], "unit": ["MWmed"]}), [],
        weather=pd.DataFrame({"timestamp": [timestamp] * 3, "temperature_c": [25, 24, 23], "station": ["A621", "A601", "Outra"], "region": [region] * 3}),
        calendar=pd.DataFrame({"date": [timestamp.date()], "region": [region], "day_type": ["dia_util"]}),
        prediction_rows=pd.DataFrame({"forecast_id": ["origem"], "region": [region], "predicted_load": [36000.0]}),
        reported_metrics=pd.DataFrame({"mae": [100.0]}), provenance=provenance,
    )


def test_known_regional_archive_overrides_incorrect_municipal_label():
    status = geographic_status(bundle(archive_hash=KNOWN_REGIONAL_ARCHIVE_SHA256))
    assert status["status"] == "incompatible"
    assert status["target_region"] == TARGET_REGION
    assert status["source_region"] == "Subsistema Sudeste/Centro-Oeste"
    assert status["evidence"]["matched_hours"] == 8784
    assert status["evidence"]["checked_period_start"].startswith("2024")
    assert status["evidence"]["archive_sha256"] == KNOWN_REGIONAL_ARCHIVE_SHA256


def test_known_regional_hash_cannot_be_bypassed_by_switching_mode_to_demo():
    assert geographic_status(bundle(mode="demo", archive_hash=KNOWN_REGIONAL_ARCHIVE_SHA256))["status"] == "incompatible"


def test_incompatible_view_removes_load_predictions_metrics_and_preserves_raw():
    raw = bundle(archive_hash=KNOWN_REGIONAL_ARCHIVE_SHA256)
    original_observations = raw.observations.copy(deep=True)
    original_predictions = raw.prediction_rows.copy(deep=True)
    original_provenance = deepcopy(raw.provenance)
    view = municipal_bundle(raw)
    assert view.observations.empty and view.prediction_rows.empty and view.reported_metrics.empty
    assert view.forecasts == []
    assert view.weather["station"].tolist() == ["A621"]
    assert len(view.calendar) == 1
    assert view.provenance["geographic_status"] == "incompatible"
    assert len(raw.weather) == 3 and raw.issues == []
    pd.testing.assert_frame_equal(raw.observations, original_observations)
    pd.testing.assert_frame_equal(raw.prediction_rows, original_predictions)
    assert raw.provenance == original_provenance


@pytest.mark.parametrize("region", [TARGET_REGION, "Rio de Janeiro (município)"])
def test_explicit_municipal_declaration_is_not_called_verified(region):
    status = geographic_status(bundle(region))
    assert status["status"] == "declared"
    assert "não constitui comprovação" in status["reason"]
    assert status["evidence"] == {}


@pytest.mark.parametrize("region", ["Estado do Rio de Janeiro", "Sudeste/Centro-Oeste", "Região Metropolitana do Rio de Janeiro", "Rio de Janeiro"])
def test_other_or_ambiguous_region_is_not_relabelled_as_target(region):
    assert geographic_status(bundle(region))["status"] == "incompatible"


def test_mixed_declared_regions_are_blocked():
    raw = bundle()
    raw.prediction_rows["region"] = "Estado do Rio de Janeiro"
    assert geographic_status(raw)["status"] == "incompatible"


def test_empty_metadata_is_unverified_without_inventing_municipality():
    raw = DataBundle("real", [], pd.DataFrame(), [])
    status = geographic_status(raw)
    assert status["status"] == "unverified" and status["source_region"] is None


def test_weather_station_identity_is_official_and_never_inferred_from_region_label():
    status = geographic_status(bundle())
    assert status["municipality_weather_stations"] == ["A621"]
    assert status["outside_municipality_weather_stations"] == ["A601"]
    assert status["unverified_weather_stations"] == ["Outra"]
    assert status["weather_station_metadata"]["A601"]["municipality_code"] == "3305554"
    assert status["weather_station_metadata"]["A621"]["municipality_code"] == "3304557"
    view = municipal_bundle(bundle())
    assert view.weather["station"].tolist() == ["A621"]
    assert len(view.observations) == 1


def test_synthetic_mode_remains_identified_and_does_not_filter_artificial_stations():
    raw = bundle("Município do Rio de Janeiro (demonstração)", mode="demo")
    view = municipal_bundle(raw)
    assert geographic_status(raw)["status"] == "synthetic"
    assert len(view.observations) == 1 and len(view.weather) == 3
    assert view.provenance["geographic_status"] == "synthetic"


def test_returned_evidence_cannot_mutate_next_audit():
    raw = bundle(archive_hash=KNOWN_REGIONAL_ARCHIVE_SHA256)
    first = geographic_status(raw)
    first["evidence"]["matched_hours"] = 0
    assert geographic_status(raw)["evidence"]["matched_hours"] == 8784


def test_regional_json_without_provenance_remains_blocked_on_second_application():
    forecast = Forecast("Modelo", "1", 24, "Sudeste/Centro-Oeste", "MWmed", pd.Timestamp("2025-01-01T00:00:00Z"), pd.DataFrame())
    raw = DataBundle("real", [forecast], pd.DataFrame(), [])
    first = municipal_bundle(raw)
    second = municipal_bundle(first)
    assert geographic_status(first)["status"] == geographic_status(second)["status"] == "incompatible"
    assert second.provenance["geographic_audit"] == first.provenance["geographic_audit"]
    assert second.provenance["geographic_quarantine"]["source_regions"] == ["Sudeste/Centro-Oeste"]
    assert second.provenance["geographic_quarantine"]["source_counts"]["forecasts"] == 1
    assert first.issues == second.issues and len(raw.forecasts) == 1 and raw.provenance == {}


def test_filtered_station_audit_retains_original_external_station():
    raw = bundle(archive_hash=KNOWN_REGIONAL_ARCHIVE_SHA256)
    first = municipal_bundle(raw)
    second = municipal_bundle(first)
    status = geographic_status(second)
    assert first.weather["station"].tolist() == second.weather["station"].tolist() == ["A621"]
    assert status["outside_municipality_weather_stations"] == ["A601"]
    assert status["unverified_weather_stations"] == ["Outra"]
    assert status["weather_station_metadata"]["A601"]["municipality"] == "Seropédica"
    assert len(raw.weather) == 3


def test_prior_declaration_cannot_override_known_hash_evidence():
    raw = bundle(archive_hash=KNOWN_REGIONAL_ARCHIVE_SHA256)
    raw.provenance["geographic_audit"] = {"status": "declared", "source_region": TARGET_REGION}
    raw.provenance["geographic_quarantine"] = {"mode": "provided", "archive_sha256": KNOWN_REGIONAL_ARCHIVE_SHA256}
    assert geographic_status(raw)["status"] == "incompatible"
    assert municipal_bundle(raw).observations.empty


def test_prior_quarantine_is_not_reused_for_a_different_archive_or_new_data():
    first = municipal_bundle(bundle("Estado do Rio de Janeiro", archive_hash="a" * 64))
    first.provenance["archive_sha256"] = "b" * 64
    first.provenance["region"] = TARGET_REGION
    first.calendar = pd.DataFrame()
    first.weather = pd.DataFrame()
    assert geographic_status(first)["status"] == "declared"


def test_malformed_prior_audit_cannot_hide_known_hash_evidence():
    raw = bundle(archive_hash=KNOWN_REGIONAL_ARCHIVE_SHA256)
    raw.forecasts = []
    raw.observations = raw.prediction_rows = raw.reported_metrics = pd.DataFrame()
    raw.provenance["geographic_audit"] = {"status": "incompatible", "source_regions": None}
    raw.provenance["geographic_quarantine"] = {
        "mode": "provided", "archive_sha256": KNOWN_REGIONAL_ARCHIVE_SHA256, "source_regions": None,
    }
    assert geographic_status(raw)["status"] == "incompatible"
    assert geographic_status(raw)["evidence"]["matched_hours"] == 8784
