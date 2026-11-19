"""Gera uma simulação fixa, reproduzível e sem fontes externas.

Não representa ONS, uma cidade ou resultados de modelos treinados. A função
que gera previsões só utiliza os timestamps, sem consultar cargas observadas.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEMO_REGION = "Município do Rio de Janeiro (demonstração)"
DEMO_UNIT = "MWmed"
DEMO_TIMEZONE = timezone(timedelta(hours=-3))
HISTORY_START = datetime(2024, 1, 1, tzinfo=DEMO_TIMEZONE)
HISTORY_END = datetime(2026, 11, 14, 23, tzinfo=DEMO_TIMEZONE)
HISTORY_HOURS = int((HISTORY_END - HISTORY_START).total_seconds() // 3600) + 1
BACKTEST_ISSUED_AT = datetime(2025, 11, 9, 23, tzinfo=DEMO_TIMEZONE)
FUTURE_ISSUED_AT = HISTORY_END
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"
DEMO_RUNS = [
    ("backtest", BACKTEST_ISSUED_AT, "test_2025"),
    ("future", FUTURE_ISSUED_AT, "forecast_2026"),
    ("validation_2025_06", datetime(2025, 6, 8, 23, tzinfo=DEMO_TIMEZONE), "validation_2025"),
    ("test_2025_09", datetime(2025, 9, 12, 23, tzinfo=DEMO_TIMEZONE), "test_2025"),
    ("test_2025_10", datetime(2025, 10, 19, 23, tzinfo=DEMO_TIMEZONE), "test_2025"),
    ("test_2025_12", datetime(2025, 12, 7, 23, tzinfo=DEMO_TIMEZONE), "test_2025"),
    ("forecast_2026_02", datetime(2026, 2, 11, 23, tzinfo=DEMO_TIMEZONE), "forecast_2026"),
    ("forecast_2026_04", datetime(2026, 4, 12, 23, tzinfo=DEMO_TIMEZONE), "forecast_2026"),
    ("forecast_2026_06", datetime(2026, 6, 14, 23, tzinfo=DEMO_TIMEZONE), "forecast_2026"),
    ("forecast_2026_09", datetime(2026, 9, 13, 23, tzinfo=DEMO_TIMEZONE), "forecast_2026"),
]


def _calendar_category(day: date) -> tuple[str, str, str]:
    """Eventos artificiais, não correspondem a calendário oficial de feriados."""
    artificial_events = {
        (2, 12): ("feriado", "nacional", "Evento sintético — feriado nacional"),
        (3, 18): ("feriado", "municipal", "Evento sintético — feriado municipal"),
        (8, 7): ("feriado", "estadual", "Evento sintético — feriado estadual"),
        (10, 20): ("ponto_facultativo", "municipal", "Evento sintético — ponto facultativo municipal"),
    }
    return artificial_events.get(
        (day.month, day.day),
        ("fim_de_semana" if day.weekday() >= 5 else "dia_util", "nenhum", ""),
    )


def _synthetic_temperature(timestamps: pd.DatetimeIndex) -> np.ndarray:
    annual = 4.5 * np.cos(2 * np.pi * (timestamps.dayofyear.to_numpy() - 22) / 365.25)
    daily = 3.2 * np.cos(2 * np.pi * (timestamps.hour.to_numpy() - 14) / 24)
    return 25 + annual + daily


def _seasonal_load(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Curva artificial conhecida; nenhum parâmetro é ajustado ao histórico."""
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    daytime = 125 * np.cos(2 * np.pi * (hour - 15) / 24)
    evening = 42 * np.cos(4 * np.pi * (hour - 19) / 24)
    weekend = np.where(weekday >= 5, -105, 0)
    temperature_effect = 12 * (_synthetic_temperature(timestamps) - 25)
    annual = 35 * np.cos(2 * np.pi * (timestamps.dayofyear.to_numpy() - 20) / 365.25)
    synthetic_event = np.array([
        -95 if _calendar_category(day)[0] == "feriado"
        else -50 if _calendar_category(day)[0] == "ponto_facultativo" else 0
        for day in timestamps.date
    ])
    return 1180 + daytime + evening + weekend + temperature_effect + annual + synthetic_event


def _forecast_payload(issued_at: datetime, horizon: int, dataset_split: str = "unspecified") -> dict:
    targets = pd.date_range(issued_at + timedelta(hours=1), periods=horizon, freq="h")
    # Erro ilustrativo determinístico, sem acesso às observações dos alvos.
    lead = np.arange(1, horizon + 1)
    values = _seasonal_load(targets) + 18 * np.sin(lead / 4) + 0.35 * lead
    return {
        "model_name": "Gerador sintético demonstrativo",
        "model_version": "demo-2.0",
        "horizon_hours": horizon,
        "region": DEMO_REGION,
        "unit": DEMO_UNIT,
        "issued_at": issued_at.isoformat(),
        "dataset_split": dataset_split,
        "predictions": [
            {"timestamp": timestamp.isoformat(), "predicted_load": round(float(value), 3)}
            for timestamp, value in zip(targets, values, strict=True)
        ],
    }


def build_demo_data(output_dir: Path = DEFAULT_OUTPUT_DIR, overwrite: bool = False) -> list[Path]:
    """Cria dados sintéticos 2024–2026; preserva arquivos existentes por padrão."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in ("historical.csv", "weather.csv", "calendar.csv")] + [
        output_dir / f"forecast_{kind}_{horizon}h.json"
        for kind, _, _ in DEMO_RUNS for horizon in (24, 48)
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Arquivos de demonstração já existem: {names}. Use --overwrite para regenerar.")

    timestamps = pd.date_range(HISTORY_START, periods=HISTORY_HOURS, freq="h")
    noise = np.random.default_rng(2026).normal(0, 13, size=len(timestamps))
    observations = pd.DataFrame({
        "timestamp": [timestamp.isoformat() for timestamp in timestamps],
        "observed_load": np.round(_seasonal_load(timestamps) + noise, 3),
        "region": DEMO_REGION,
        "unit": DEMO_UNIT,
    })
    observations.to_csv(targets[0], index=False, encoding="utf-8", lineterminator="\n")
    weather_frames = []
    for station, offset, seed in (("Estação sintética A", 0.0, 202401), ("Estação sintética B", -1.4, 202402)):
        noise = np.random.default_rng(seed).normal(0, 0.45, size=len(timestamps))
        weather_frames.append(pd.DataFrame({
            "timestamp": [timestamp.isoformat() for timestamp in timestamps],
            "temperature_c": np.round(_synthetic_temperature(timestamps) + offset + noise, 3),
            "station": station,
            "region": DEMO_REGION,
        }))
    pd.concat(weather_frames, ignore_index=True).to_csv(
        output_dir / "weather.csv", index=False, encoding="utf-8", lineterminator="\n",
    )
    calendar_rows = []
    for day in pd.date_range("2024-01-01", "2026-12-31", freq="D").date:
        day_type, scope, event_name = _calendar_category(day)
        calendar_rows.append({
            "date": day.isoformat(), "region": DEMO_REGION,
            "day_type": day_type, "holiday_scope": scope, "event_name": event_name,
        })
    pd.DataFrame(calendar_rows).to_csv(
        output_dir / "calendar.csv", index=False, encoding="utf-8", lineterminator="\n",
    )
    for kind, issued_at, dataset_split in DEMO_RUNS:
        for horizon in (24, 48):
            path = output_dir / f"forecast_{kind}_{horizon}h.json"
            path.write_text(
                json.dumps(_forecast_payload(issued_at, horizon, dataset_split), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerar dados sintéticos fixos do painel (não são dados reais).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Sobrescrever somente os arquivos conhecidos de demonstração.")
    args = parser.parse_args()
    for path in build_demo_data(args.output_dir, overwrite=args.overwrite):
        print(path)


if __name__ == "__main__":
    main()
