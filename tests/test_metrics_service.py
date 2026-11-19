"""Known numerical examples, zero-load policy and undefined evaluation."""

import numpy as np
import pandas as pd
import pytest

from services.metrics_service import calculate_metrics


def test_known_metrics_are_computed_without_rounding():
    aligned = pd.DataFrame({"observed_load": [100, 200, 300], "predicted_load": [110, 180, 330]})
    result = calculate_metrics(aligned)
    assert result["mae"] == pytest.approx(20)
    assert result["rmse"] == pytest.approx(np.sqrt(1400 / 3))
    assert result["mape"] == pytest.approx(10)
    assert result["n_pairs"] == 3
    assert result["mape_valid_pairs"] == 3
    assert result["mape_excluded_zeros"] == 0


def test_large_finite_errors_do_not_overflow_mae_or_rmse():
    aligned = pd.DataFrame({"observed_load": [1e308, 1e308, 0.0], "predicted_load": [0.0, 0.0, 0.0]})
    with np.errstate(over="raise", invalid="raise"):
        result = calculate_metrics(aligned)
    assert result["mae"] == pytest.approx(1e308 * (2 / 3))
    assert result["rmse"] == pytest.approx(1e308 * np.sqrt(2 / 3))
    assert result["mape"] == 100.0


def test_mape_refuses_result_beyond_float_range_instead_of_infinity():
    aligned = pd.DataFrame({"observed_load": [1e-300], "predicted_load": [1e300]})
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="MAPE excede"):
            calculate_metrics(aligned)


def test_zero_observations_remain_in_mae_rmse_but_not_mape():
    aligned = pd.DataFrame({"observed_load": [0, 100], "predicted_load": [10, 90]})
    result = calculate_metrics(aligned)
    assert result["mae"] == 10
    assert result["rmse"] == 10
    assert result["mape"] == pytest.approx(10)
    assert result["mape_valid_pairs"] == 1
    assert result["mape_excluded_zeros"] == 1
    assert result["n_pairs"] == 2


def test_positive_near_zero_observations_remain_in_mape_without_epsilon():
    # Positive pairs have 100% and 50% errors, so their MAPE is exactly 75%.
    # The zero pair participates only in the two metrics expressed in load units.
    aligned = pd.DataFrame({"observed_load": [0.0, 1e-12, 2.0], "predicted_load": [7.0, 2e-12, 1.0]})
    result = calculate_metrics(aligned)
    assert result["mape"] == pytest.approx(75.0)
    assert result["mape_valid_pairs"] == 2
    assert result["mape_excluded_zeros"] == 1
    assert result["mae"] == pytest.approx((8 + 1e-12) / 3)
    assert result["rmse"] == pytest.approx(np.sqrt((50 + 1e-24) / 3))


def test_mape_undefined_when_all_observed_loads_are_zero():
    result = calculate_metrics(pd.DataFrame({"observed_load": [0, 0], "predicted_load": [0, 3]}))
    assert result["mae"] == 1.5
    assert result["rmse"] == pytest.approx(np.sqrt(4.5))
    assert result["mape"] is None
    assert result["mape_valid_pairs"] == 0
    assert result["mape_excluded_zeros"] == 2


def test_empty_sample_is_undefined_instead_of_perfect():
    result = calculate_metrics(pd.DataFrame())
    assert result == {
        "mae": None,
        "rmse": None,
        "mape": None,
        "n_pairs": 0,
        "mape_valid_pairs": 0,
        "mape_excluded_zeros": 0,
    }


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, True, "inválido", -1])
@pytest.mark.parametrize("column", ["observed_load", "predicted_load"])
def test_nonfinite_boolean_negative_or_invalid_loads_are_refused(invalid, column):
    aligned = pd.DataFrame({"observed_load": [100], "predicted_load": [100]})
    aligned[column] = [invalid]
    with pytest.raises(ValueError):
        calculate_metrics(aligned)


def test_nonempty_sample_requires_load_columns():
    with pytest.raises(ValueError, match="observed_load"):
        calculate_metrics(pd.DataFrame({"unrelated": [1]}))


@pytest.mark.parametrize("metadata,values", [
    ("horizon_hours", [24, 48]),
    ("issued_at", ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"]),
    ("unit", ["MW", "MWmed"]),
    ("dataset_split", ["validation_2025", "test_2025"]),
    ("region", ["Rio", "Outra região"]),
])
def test_metrics_refuse_mixed_issuances_or_horizons(metadata, values):
    aligned = pd.DataFrame({"observed_load": [100, 200], "predicted_load": [90, 190], metadata: values})
    with pytest.raises(ValueError, match="única emissão"):
        calculate_metrics(aligned)
