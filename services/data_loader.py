"""Carregamento local de dados com isolamento entre demonstração e dados reais."""

from dataclasses import dataclass, field, replace
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.forecast_schema import (
    Forecast, ForecastValidationError, SUPPORTED_UNITS, parse_timestamp, validate_forecast,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OBSERVATION_COLUMNS = ["timestamp", "observed_load", "region", "unit"]
WEATHER_COLUMNS = ["timestamp", "temperature_c", "station", "region"]
CALENDAR_COLUMNS = ["date", "region", "day_type", "holiday_scope", "event_name"]
DAY_TYPES = {"dia_util", "fim_de_semana", "feriado", "ponto_facultativo"}
HOLIDAY_SCOPES = {"nacional", "estadual", "municipal", "nenhum", "nao_informado"}


class DataLoadError(ValueError):
    """Um arquivo não pôde ser carregado ou contém dados inválidos."""


@dataclass
class DataBundle:
    mode: str
    forecasts: list[Forecast]
    observations: pd.DataFrame
    issues: list[str]
    weather: pd.DataFrame = field(default_factory=lambda: empty_weather())
    calendar: pd.DataFrame = field(default_factory=lambda: empty_calendar())
    prediction_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    reported_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    provenance: dict = field(default_factory=dict)


def empty_observations() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        "observed_load": pd.Series(dtype="float64"),
        "region": pd.Series(dtype="str"),
        "unit": pd.Series(dtype="str"),
    })


def empty_weather() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        "temperature_c": pd.Series(dtype="float64"),
        "station": pd.Series(dtype="str"),
        "region": pd.Series(dtype="str"),
    })


def empty_calendar() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in CALENDAR_COLUMNS})


def _read_csv(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise DataLoadError(f"Arquivo de {label.lower()} não encontrado: {path.name}.") from exc
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataLoadError(f"Não foi possível ler o CSV {path.name}: {exc}") from exc
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataLoadError(f"{label} {path.name}: colunas obrigatórias ausentes: {', '.join(missing)}.")
    return frame[columns].copy()


def _parse_timestamps(frame: pd.DataFrame, source: str) -> None:
    try:
        frame["timestamp"] = pd.DatetimeIndex([
            parse_timestamp(value, f"linha {index + 2}.timestamp")
            for index, value in enumerate(frame["timestamp"])
        ])
    except ForecastValidationError as exc:
        raise DataLoadError(f"{source}: {exc}") from exc


def _require_region_units(frame: pd.DataFrame, source: str) -> None:
    if not frame["unit"].isin(SUPPORTED_UNITS).all():
        raise DataLoadError(f"{source}: a unidade deve ser exatamente 'MW' ou 'MWmed', sem conversão implícita.")


def _reject_duplicate_json_keys(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataLoadError(f"Chave JSON duplicada: {key}.")
        result[key] = value
    return result


def _validate_hourly_grid(frame: pd.DataFrame, source: str) -> None:
    """Cada par região/unidade tem uma grade própria; lacunas inteiras são válidas."""
    hour_ns = 3600 * 1_000_000_000
    for (region, unit), rows in frame.groupby(["region", "unit"], sort=False):
        instants = [int(timestamp.value) for timestamp in rows["timestamp"].sort_values()]
        for previous, current in zip(instants, instants[1:]):
            difference = current - previous
            if difference <= 0 or difference % hour_ns != 0:
                raise DataLoadError(
                    f"{source}: grade horária irregular na região '{region}', unidade '{unit}'. "
                    "As diferenças entre observações devem ser múltiplos inteiros positivos de uma hora; "
                    "lacunas de horas inteiras são permitidas, sem interpolação."
                )


def load_forecast(path: str | Path) -> Forecast:
    """Lê um JSON completo sem substituir ou reordenar valores do produtor."""
    path = Path(path)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        return replace(validate_forecast(payload), source_path=path)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Arquivo de previsão não encontrado: {path.name}.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataLoadError(f"Não foi possível ler o JSON {path.name}: {exc}") from exc
    except ForecastValidationError as exc:
        raise DataLoadError(f"Previsão inválida em {path.name}: {exc}") from exc


def load_observations(path: str | Path) -> pd.DataFrame:
    """Lê CSV: timestamp, observed_load, region, unit (MW ou MWmed).

    Lacunas de horas inteiras são preservadas. Cada par região/unidade possui
    sua própria grade e só pode ter uma observação por instante. Não se interpola
    nem se preenche carga faltante.
    """
    path = Path(path)
    frame = _read_csv(path, OBSERVATION_COLUMNS, "Histórico")
    if frame.empty:
        return empty_observations()
    _parse_timestamps(frame, f"Histórico {path.name}")
    try:
        frame["observed_load"] = pd.to_numeric(frame["observed_load"], errors="raise").astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataLoadError(f"Histórico {path.name}: observed_load deve conter números finitos na unidade declarada.") from exc
    if not np.isfinite(frame["observed_load"]).all() or (frame["observed_load"] < 0).any():
        raise DataLoadError(f"Histórico {path.name}: observed_load deve ser finita e não negativa na unidade declarada.")
    frame["region"] = frame["region"].str.strip()
    if (frame["region"] == "").any():
        raise DataLoadError(f"Histórico {path.name}: region não pode ser vazia.")
    _require_region_units(frame, f"Histórico {path.name}")
    if frame.duplicated(["region", "unit", "timestamp"]).any():
        raise DataLoadError(f"Histórico {path.name}: existem observações duplicadas por região, unidade e instante.")
    frame = frame.sort_values(["region", "unit", "timestamp"]).reset_index(drop=True)
    _validate_hourly_grid(frame, f"Histórico {path.name}")
    return frame


def load_weather(path: str | Path) -> pd.DataFrame:
    """Lê temperaturas horárias por estação, preservando estações distintas."""
    path = Path(path)
    source = f"Meteorologia {path.name}"
    frame = _read_csv(path, WEATHER_COLUMNS, "Meteorologia")
    if frame.empty:
        return empty_weather()
    _parse_timestamps(frame, source)
    for column in ("station", "region"):
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise DataLoadError(f"{source}: {column} deve conter texto não vazio.")
    try:
        frame["temperature_c"] = pd.to_numeric(frame["temperature_c"], errors="raise").astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataLoadError(f"{source}: temperature_c deve ser numérica e finita, em °C.") from exc
    if not np.isfinite(frame["temperature_c"]).all() or not frame["temperature_c"].between(-60, 60).all():
        raise DataLoadError(f"{source}: temperature_c deve ser finita e estar entre -60 e 60 °C.")
    if frame.duplicated(["station", "region", "timestamp"]).any():
        raise DataLoadError(f"{source}: observações meteorológicas duplicadas por estação, região e instante.")
    return frame.sort_values(["region", "station", "timestamp"]).reset_index(drop=True)


def _validate_calendar(frame: pd.DataFrame, source: str) -> None:
    if frame.duplicated(["region", "date"]).any():
        raise DataLoadError(f"{source}: datas duplicadas por região.")
    for region, rows in frame.groupby("region", sort=False):
        dates = sorted(rows["date"])
        if dates and (dates[-1] - dates[0]).days + 1 != len(dates):
            raise DataLoadError(
                f"{source}: calendário incompleto na região '{region}'; informe todos os dias do intervalo disponível."
            )


def load_calendar(path: str | Path) -> pd.DataFrame:
    """Lê categorias diárias declaradas pelo produtor, sem inferir feriados."""
    path = Path(path)
    source = f"Calendário {path.name}"
    frame = _read_csv(path, CALENDAR_COLUMNS, "Calendário")
    if frame.empty:
        return empty_calendar()
    try:
        if not frame["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError("formato inválido")
        frame["date"] = [date.fromisoformat(value) for value in frame["date"]]
    except (TypeError, ValueError) as exc:
        raise DataLoadError(f"{source}: date deve ser uma data local válida no formato YYYY-MM-DD.") from exc
    for column in ("region", "day_type", "holiday_scope", "event_name"):
        frame[column] = frame[column].str.strip()
    if frame["region"].eq("").any():
        raise DataLoadError(f"{source}: region deve conter texto não vazio.")
    if not frame["day_type"].isin(DAY_TYPES).all():
        raise DataLoadError(f"{source}: day_type deve ser dia_util, fim_de_semana, feriado ou ponto_facultativo.")
    if not frame["holiday_scope"].isin(HOLIDAY_SCOPES).all():
        raise DataLoadError(f"{source}: holiday_scope deve ser nacional, estadual, municipal, nenhum ou nao_informado.")
    special = frame["day_type"].isin({"feriado", "ponto_facultativo"})
    if (special & frame["event_name"].eq("")).any():
        raise DataLoadError(f"{source}: event_name é obrigatório para feriados e pontos facultativos.")
    if (special & frame["holiday_scope"].eq("nenhum")).any() or ((~special) & frame["holiday_scope"].ne("nenhum")).any():
        raise DataLoadError(f"{source}: holiday_scope deve corresponder ao tipo de dia declarado.")
    _validate_calendar(frame, source)
    return frame.sort_values(["region", "date"]).reset_index(drop=True)


def _load_csv_files(paths: list[Path], loader, empty_factory, issues: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(loader(path))
        except DataLoadError as exc:
            issues.append(str(exc))
    return pd.concat(frames, ignore_index=True) if frames else empty_factory()


def _discover_files(directory: Path, suffix: str, issues: list[str]) -> list[Path]:
    """Distingue uma pasta ainda ausente de uma pasta que não pôde ser lida."""
    try:
        return sorted(
            path for path in directory.iterdir()
            if path.suffix.lower() == suffix and path.is_file()
        )
    except FileNotFoundError:
        return []
    except OSError as exc:
        issues.append(f"Não foi possível listar a pasta {directory.name}: {exc}")
        return []


def load_data(mode: str, data_dir: Path = DATA_DIR) -> DataBundle:
    """Carrega uma única origem; arquivos ausentes produzem estados vazios.

    Arquivos inválidos não interrompem a interface: são omitidos e reportados.
    Havendo colisão ou grade horária irregular entre CSVs, todo o histórico é
    recusado para impedir uma seleção arbitrária durante a avaliação.
    """
    if mode == "provided":
        from services.archive_loader import load_archive_bundle

        return load_archive_bundle(Path(data_dir) / "imported" / "config.json")
    if mode not in ("demo", "real"):
        raise DataLoadError("Modo de dados inválido; use 'demo', 'real' ou 'provided'.")
    data_dir = Path(data_dir)
    forecast_dir = data_dir / ("demo" if mode == "demo" else "predictions")
    historical_dir = data_dir / ("demo" if mode == "demo" else "historical")
    issues = []
    forecast_paths = _discover_files(forecast_dir, ".json", issues)
    historical_paths = _discover_files(historical_dir, ".csv", issues)
    if mode == "demo":
        weather_paths = [path for path in historical_paths if path.name == "weather.csv"]
        calendar_paths = [path for path in historical_paths if path.name == "calendar.csv"]
        historical_paths = [path for path in historical_paths if path.name == "historical.csv"]
    else:
        weather_paths = _discover_files(data_dir / "weather", ".csv", issues)
        calendar_paths = _discover_files(data_dir / "calendar", ".csv", issues)
    forecasts = []
    for path in forecast_paths:
        try:
            forecasts.append(load_forecast(path))
        except DataLoadError as exc:
            issues.append(str(exc))
    observations = _load_csv_files(historical_paths, load_observations, empty_observations, issues)
    if observations.duplicated(["region", "unit", "timestamp"]).any():
        issues.append("Histórico indisponível: CSVs contêm observações duplicadas por região, unidade e instante. Remova as sobreposições.")
        observations = empty_observations()
    observations = observations.sort_values(["region", "unit", "timestamp"]).reset_index(drop=True)
    try:
        _require_region_units(observations, "Histórico combinado indisponível")
        _validate_hourly_grid(observations, "Histórico combinado indisponível")
    except DataLoadError as exc:
        issues.append(str(exc))
        observations = empty_observations()
    weather = _load_csv_files(weather_paths, load_weather, empty_weather, issues)
    if weather.duplicated(["station", "region", "timestamp"]).any():
        issues.append("Meteorologia indisponível: CSVs contêm observações duplicadas por estação, região e instante.")
        weather = empty_weather()
    weather = weather.sort_values(["region", "station", "timestamp"]).reset_index(drop=True)
    calendar = _load_csv_files(calendar_paths, load_calendar, empty_calendar, issues)
    try:
        _validate_calendar(calendar, "Calendário combinado indisponível")
    except DataLoadError as exc:
        issues.append(str(exc))
        calendar = empty_calendar()
    calendar = calendar.sort_values(["region", "date"]).reset_index(drop=True)
    forecasts.sort(key=lambda forecast: (
        forecast.region, forecast.horizon_hours, forecast.issued_at,
        forecast.model_name, forecast.model_version,
    ))
    return DataBundle(mode=mode, forecasts=forecasts, observations=observations, issues=issues, weather=weather, calendar=calendar)
