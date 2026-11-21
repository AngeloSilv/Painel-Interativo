
"""Leitores V1–V5. Métricas compactas e uma partição de detalhe por consulta."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st

from services import data_loader
from services.archive_loader import load_archive_bundle

TARGET = "ONS_SECO_piloto"
REGION = "ONS Sudeste/Centro-Oeste"
TZ = "America/Sao_Paulo"
V5 = "entrega_painel_v5"
OLD = "metricas_versoes_anteriores"
VERSIONS = ("v1", "v2", "v3", "v4", "v5")
SCENARIOS = {
    "hipotese_UTC": "Hipótese UTC",
    "hipotese_Sao_Paulo": "Hipótese São Paulo",
    "sem_cenario_bruto": "Carimbo original V1",
}
STAGES = {
    "teste_2025": "Teste 2025",
    "validacao_1s2025": "Validação 1S2025",
    "retrospectiva_2026": "Retrospectiva 2026",
}
LEGACY_SHA = "b10830eaae445d958d42f05f00551b66a500126af7cbfdfd7d981f34f7e7f9d7"
METRICS = {"mae_mwmed": "MAE (MWmed)", "rmse_mwmed": "RMSE (MWmed)", "mape_pct": "MAPE (%)"}
BASE_KEY = ["alvo_id", "unidade", "versao", "cenario", "etapa", "horizonte_h", "modelo"]


def root():
    return data_loader.DATA_DIR / "versions"


def enabled():
    return (root() / V5 / "MANIFESTO_v5.json").is_file()


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def check_identity(frame, versions):
    if not (frame["alvo_id"].eq(TARGET).all() and frame["unidade"].eq("MWmed").all()
            and frame["versao"].isin(versions).all()
            and frame["cenario"].isin(SCENARIOS).all()
            and frame["escopo_avaliacao"].eq("retrospectiva_nao_cego").all()):
        raise ValueError("Alvo, unidade, versão, cenário ou escopo incompatível.")


@st.cache_data(show_spinner=False)
def metric_tables(directory):
    directory = Path(directory)
    result = {}
    for name, dimensions in (
        ("globais", ["bloco_horas"]), ("por_passo", ["passo_h"]),
        ("por_hora", ["bloco_horas", "hora_alvo_local"]), ("por_mes", ["bloco_horas", "mes_alvo_local"])
    ):
        old = read_csv(directory / OLD / f"metricas_{name}.csv")
        new = read_csv(directory / V5 / f"metricas_{name}_v5.csv")
        check_identity(old, (*VERSIONS[:4], "piloto"))
        check_identity(new, ("v5",))
        frame = pd.concat([old, new], ignore_index=True)
        if frame.duplicated(BASE_KEY + dimensions).any():
            raise ValueError(f"Chave de métrica duplicada: {name}.")
        if not (frame["horizonte_h"].isin([24, 48]).all()
                and frame["etapa"].isin(STAGES).all()
                and frame["eh_selecionado"].isin([0, 1]).all()):
            raise ValueError(f"Dimensões inválidas: {name}.")
        frame["papel"] = np.select(
            [frame["modelo"].isin(["persistencia", "sazonal_semanal"]),
             frame["modelo"].eq("piloto_so_carga") | frame["versao"].eq("piloto"),
             frame["eh_selecionado"].eq(1)],
            ["Referência", "Piloto", "Selecionado"], default="Alternativa",
        )
        result[name] = frame
    paired = read_csv(directory / V5 / "comparacao_v4_v5_pares_comuns.csv")
    check_identity(paired, ("v5",))
    if (paired.duplicated(BASE_KEY + ["bloco_horas", "versao_ref", "modelo_ref", "tipo_comparacao"]).any()
            or not paired["versao_ref"].eq("v4").all() or not paired["n_pares_comuns"].gt(0).all()
            or not paired["n_mape_validos_comuns"].between(0, paired["n_pares_comuns"]).all()
            or any(not paired[f"n_pares_{v}"].eq(paired["n_pares_comuns"] + paired[f"n_excluidos_{v}"]).all()
                   for v in ("v4", "v5"))):
        raise ValueError("Comparação de pares comuns inválida.")
    result["pares"] = paired
    result["disponibilidade"] = read_csv(directory / OLD / "disponibilidade_metricas.csv")
    return result


@dataclass(frozen=True)
class ScenarioData:
    observations: pd.DataFrame
    weather: pd.DataFrame
    calendar: pd.DataFrame
    events: dict
    pending: pd.DataFrame


def classify(row, names):
    holiday, optional = bool(row.eh_feriado), bool(row.eh_ponto_facultativo)
    kind = ("feriado" if holiday else "ponto_facultativo" if optional else
            "fim_de_semana" if row.fim_de_semana else "dia_util")
    ids = json.loads(row.evento_versao_ids)
    matched = [names[i] for i in ids]
    scopes = {"federal_administracao": "federal_administracao", "nacional": "nacional",
              "estadual_rj": "estadual", "municipal_rio": "municipal"}
    event = next((e for e in matched if (holiday and e['eh_feriado']) or (not holiday and optional and e['eh_ponto_facultativo'])), None)
    scope = scopes.get(event['esfera'], "nao_informado") if event else ("nao_informado" if holiday or optional else "nenhum")
    return kind, scope, "; ".join(e['nome_evento'] for e in matched)


@st.cache_data(show_spinner=False)
def load_scenario(directory, scenario):
    base = Path(directory) / V5
    history = read_csv(base / "serie_horaria_auditada_v5.csv", keep_default_na=False, na_values=[""])
    events = read_csv(base / "eventos_calendario_v5.csv")
    check_identity(history, ("v5",))
    check_identity(events, ("v5",))
    history = history.loc[history["cenario"].eq(scenario)].copy()
    events = events.loc[events["cenario"].eq(scenario)]
    if history.empty or events["evento_versao_id"].duplicated().any():
        raise ValueError("Série ou inventário de eventos V5 inválido.")
    names = events.set_index("evento_versao_id", drop=False).to_dict(orient="index")
    timestamp = pd.to_datetime(history["instante"], utc=True)
    if timestamp.duplicated().any() or not np.isfinite(history["carga_mwmed"]).all():
        raise ValueError("Histórico V5 com duplicatas ou cargas inválidas.")
    obs = pd.DataFrame({"timestamp": timestamp, "observed_load": history["carga_mwmed"], "region": REGION, "unit": "MWmed"})
    weather = pd.concat([
        pd.DataFrame({"timestamp": timestamp, "temperature_c": history[column], "station": station, "region": REGION})
        for column, station in (("temp_a621_c", "A621"), ("temp_a601_c", "A601"))
    ]).dropna(subset=["temperature_c"])
    calendar = pd.DataFrame(
        [classify(row, names) for row in history.itertuples(index=False)],
        columns=["day_type", "holiday_scope", "event_name"], index=history.index,
    ).assign(timestamp=timestamp, region=REGION)
    return ScenarioData(obs, weather, calendar, names, read_csv(base / "eventos_pendentes_v5.csv"))


def setting(name, default=""):
    local = os.getenv(name)
    if local is not None:
        return local
    try:
        return str(st.secrets.get(name, default))
    except FileNotFoundError:
        return default


def _details_dir():
    return Path(setting("ENERGY_FORECAST_DETAIL_DIR", str(root() / "detail")))


@st.cache_resource(show_spinner=False)
def _download(base, filename, expected, size_limit):
    if not base.startswith("https://") or Path(filename).name != filename:
        raise ValueError("Endereço de dados inválido.")
    directory = Path(tempfile.gettempdir()) / "energy-forecast-versions" / expected
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.is_file():
        with destination.open("rb") as saved:
            if hashlib.file_digest(saved, "sha256").hexdigest() == expected:
                return destination
    temp = None
    try:
        with urlopen(f"{base.rstrip('/')}/{filename}", timeout=30) as source, tempfile.NamedTemporaryFile(dir=directory, delete=False) as target:
            temp = Path(target.name)
            result = hashlib.sha256()
            size = 0
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > size_limit:
                    raise ValueError("Arquivo remoto excede o tamanho permitido.")
                result.update(chunk)
                target.write(chunk)
        if result.hexdigest() != expected:
            raise ValueError(f"SHA-256 divergente: {filename}.")
        temp.replace(destination)
    finally:
        if temp and temp.exists():
            temp.unlink()
    return destination


def detail_index():
    path = _details_dir() / "index.json"
    if not path.is_file():
        url = setting("ENERGY_FORECAST_DETAIL_URL")
        sha = setting("ENERGY_FORECAST_DETAIL_SHA256")
        if url:
            if not re.fullmatch("[a-f0-9]{64}", sha):
                raise ValueError("Informe o SHA-256 do índice detalhado.")
            path = _download(url, "index.json", sha, 1024 * 1024)
        else:
            path = root() / "preview" / "index.json"
            if not path.is_file():
                return []
    manifest = json.loads((root() / V5 / "MANIFESTO_v5.json").read_text(encoding="utf-8"))
    index = json.loads(path.read_text(encoding="utf-8"))
    expected = {f"previsoes_{h}h_v5.csv": manifest["arquivos"][f"previsoes_{h}h_v5.csv"]["sha256"] for h in (24, 48)}
    if index["schema"] != 1 or index["source"] != expected:
        raise ValueError("Índice incompatível com o manifesto V5.")
    indexed = set()
    for e in index["partitions"]:
        key = (e["horizon"], e["scenario"], e["stage"], e["month"])
        if (key in indexed or e["horizon"] not in (24, 48) or e["scenario"] not in tuple(SCENARIOS)[:2]
                or e["stage"] not in STAGES or not re.fullmatch(r"\d{4}-\d\d", e["month"])
                or e["file"] != "__".join(map(str, key)) + ".parquet"
                or not re.fullmatch("[a-f0-9]{64}", e["sha256"]) or not 0 < e["bytes"] <= 128 * 1024 * 1024):
            raise ValueError("Partição declarada inválida.")
        indexed.add(key)
    preview = path == root() / "preview" / "index.json"
    return [dict(entry, preview=preview) for entry in index["partitions"]]


@st.cache_resource(show_spinner=False)
def verified_path(path, digest, size, mtime):
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != digest:
            raise ValueError(f"SHA-256 divergente: {path.name}.")
    return path


def partition_path(entry):
    directory = root() / "preview" if entry["preview"] else _details_dir()
    local = directory / entry["file"]
    if local.is_file():
        stamp = local.stat()
        return verified_path(local, entry["sha256"], stamp.st_size, stamp.st_mtime_ns)
    url = setting("ENERGY_FORECAST_DETAIL_URL")
    if not url or entry["preview"]:
        raise FileNotFoundError(entry["file"])
    return _download(url, entry["file"], entry["sha256"], entry["bytes"])


@st.cache_data(show_spinner=False, max_entries=24)
def issues_for_models(path, digest):
    return pd.read_parquet(path, columns=["modelo", "origem_t", "eh_selecionado"]).drop_duplicates()


@st.cache_data(show_spinner=False, max_entries=64)
def forecast_detail(path, digest, horizon, model, issue):
    data = pd.read_parquet(path, filters=[("modelo", "=", model), ("origem_t", "=", issue)]).sort_values("passo_h")
    if data["passo_h"].tolist() != list(range(1, horizon + 1)):
        raise ValueError("A emissão não contém o horizonte completo.")
    times = pd.to_datetime(data["instante_alvo"], utc=True)
    if not (times - pd.Timestamp(issue)).eq(pd.to_timedelta(data["passo_h"], unit="h")).all():
        raise ValueError("Alvos incompatíveis com a emissão.")
    return data


@st.cache_resource(show_spinner=False)
def legacy_bundle(path, digest):
    with tempfile.TemporaryDirectory() as temp:
        config = Path(temp) / "config.json"
        # O caminho do ZIP é relativo à configuração no adaptador legado.
        # Use uma cópia temporária de somente 23 MB e preserve os originais.
        shutil.copyfile(path, config.parent / "painel.zip")
        config.write_text(json.dumps(dict(
            archive="painel.zip", sha256=digest, region=REGION, unit="MWmed", timezone=TZ,
            region_source="ONS SE/CO", timezone_source="Hipótese de exibição do leitor histórico",
        )), encoding="utf-8")
        bundle = load_archive_bundle(config)
        if not bundle.prediction_rows.empty:
            bundle.prediction_rows["model_version"] = "v1"
        return bundle


def legacy_available():
    return (_details_dir() / "painel.zip").is_file() or bool(setting("ENERGY_FORECAST_DETAIL_URL"))


def legacy_path():
    local = _details_dir() / "painel.zip"
    if local.is_file():
        return local
    url = setting("ENERGY_FORECAST_DETAIL_URL")
    return _download(url, local.name, LEGACY_SHA, 24_000_000) if url else None
