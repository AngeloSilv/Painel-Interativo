"""Error metrics for an already aligned, single-issuance evaluation sample."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd


def calculate_metrics(aligned: pd.DataFrame) -> dict[str, float | int | None]:
    """Compute MAE/RMSE in the declared load unit and MAPE in percent.

    MAPE excludes pairs whose observed load equals zero. Such pairs continue to
    contribute to MAE and RMSE. If all observations are zero, MAPE is undefined
    (``None``). Empty samples do not imply zero error: all three metrics are
    ``None``. Values are returned without display rounding. MAE and RMSE use
    scaled arithmetic to avoid overflow when loads are large but finite.
    Positive observations close to zero remain included, without an arbitrary
    epsilon; their denominators can make MAPE very large and unstable.
    A MAPE beyond float64's representable range raises an explicit ValueError.
    """
    if aligned.empty:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "n_pairs": 0,
            "mape_valid_pairs": 0,
            "mape_excluded_zeros": 0,
        }
    if not {"observed_load", "predicted_load"}.issubset(aligned.columns):
        raise ValueError("A avaliação deve conter observed_load e predicted_load.")
    for metadata in ("issued_at", "horizon_hours", "unit", "region", "dataset_split"):
        if metadata in aligned and aligned[metadata].nunique(dropna=False) != 1:
            raise ValueError("Calcule métricas de uma única emissão, horizonte, unidade, região e partição por vez.")
    if "unit" in aligned and not aligned["unit"].isin(("MW", "MWmed")).all():
        raise ValueError("A unidade deve ser MW ou MWmed, sem mistura nem conversão implícita.")
    for column in ("observed_load", "predicted_load"):
        if aligned[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError("Cargas booleanas não são válidas para métricas.")
    try:
        observed = pd.to_numeric(aligned["observed_load"], errors="raise").to_numpy(dtype=float)
        predicted = pd.to_numeric(aligned["predicted_load"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("As cargas da avaliação devem ser números finitos.") from exc
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("As cargas da avaliação devem ser números finitos.")
    if (observed < 0).any() or (predicted < 0).any():
        raise ValueError("As cargas da avaliação devem ser não negativas.")
    errors = observed - predicted
    nonzero = observed != 0
    absolute_errors = np.abs(errors)
    scale = float(absolute_errors.max())
    if scale:
        normalized = absolute_errors / scale
        mae = float(normalized.mean() * scale)
        rmse = float(np.sqrt(np.mean(np.square(normalized))) * scale)
    else:
        mae = rmse = 0.0
    mape = None
    if nonzero.any():
        # Decimal keeps tiny observed values from overflowing intermediate
        # divisions. Only the final result must fit the public float API.
        with localcontext() as context:
            context.prec = 40
            ratios = (
                Decimal(float(error)) / Decimal(float(actual))
                for error, actual in zip(absolute_errors[nonzero], observed[nonzero], strict=True)
            )
            mape = float(sum(ratios, Decimal(0)) / int(nonzero.sum()) * 100)
        if not np.isfinite(mape):
            raise ValueError(
                "O MAPE excede a capacidade numérica disponível; verifique a escala das cargas observadas."
            )
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n_pairs": int(len(observed)),
        "mape_valid_pairs": int(nonzero.sum()),
        "mape_excluded_zeros": int((~nonzero).sum()),
    }
