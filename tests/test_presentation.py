"""Formatação de ausência e exportação sem arredondar os dados científicos."""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from components.metrics_cards import format_number
from components.tables import csv_bytes


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, np.inf, -np.inf])
def test_unavailable_values_are_explicit_without_zero(value):
    assert format_number(value, 2) == "N/D"


def test_brazilian_number_keeps_value_unit_separate():
    assert format_number(62150.1234, 2) == "62.150,12"
    assert format_number(0, 2) == "0,00"


def test_csv_keeps_original_precision_and_geographic_metadata():
    frame = pd.DataFrame({"predicted_load": [1234.56789], "unit": ["MWmed"], "region": ["Município do Rio de Janeiro"]})
    original = frame.copy(deep=True)
    assert format_number(frame.predicted_load.iloc[0], 2) == "1.234,57"
    restored = pd.read_csv(BytesIO(csv_bytes(frame)), encoding="utf-8-sig")
    assert restored.predicted_load.iloc[0] == pytest.approx(1234.56789)
    assert restored.region.iloc[0] == "Município do Rio de Janeiro"
    assert restored.unit.iloc[0] == "MWmed"
    pd.testing.assert_frame_equal(frame, original)
