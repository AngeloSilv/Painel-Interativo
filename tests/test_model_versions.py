"""Integração com o catálogo publicado e partições mínimas descartáveis."""

from datetime import timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from services import model_versions as versions

ROOT = Path(__file__).resolve().parents[1]


def prepare_index(tmp_path):
    manifest = json.loads((versions.root() / versions.V5 / "MANIFESTO_v5.json").read_text(encoding="utf-8"))
    issue = pd.Timestamp("2025-07-01T03:00:00Z")
    entries = []
    source = {}
    for horizon in (24, 48):
        name = f"previsoes_{horizon}h_v5.csv"
        source[name] = manifest["arquivos"][name]["sha256"]
        path = tmp_path / f"{horizon}__hipotese_UTC__teste_2025__2025-07.parquet"
        pd.DataFrame({
            "modelo": "ols_estabilizado", "origem_t": issue.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "passo_h": range(1, horizon + 1), "eh_selecionado": 1,
            "instante_alvo": pd.date_range(issue + timedelta(hours=1), periods=horizon, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "y_real": 40_000.0, "y_pred": 40_010.0, "y_baseline_semanal": 39_800.0,
            "y_baseline_persistencia": 39_900.0, "fim_de_semana": 0, "eh_feriado": 0,
            "eh_ponto_facultativo": 0, "evento_versao_ids": "[]",
        }).to_parquet(path, index=False)
        entries.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            bytes=path.stat().st_size, horizon=horizon, scenario="hipotese_UTC",
                            stage="teste_2025", month="2025-07", rows=horizon))
    (tmp_path / "index.json").write_text(json.dumps({"schema": 1, "source": source, "partitions": entries}))


def test_v5_starts_with_exact_horizons_and_older_aggregate_only(monkeypatch, tmp_path):
    prepare_index(tmp_path)
    monkeypatch.setenv("ENERGY_FORECAST_DETAIL_DIR", str(tmp_path))
    page = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
    assert not page.exception
    assert page.selectbox(key="catalog_explorer_version").value == "V5"
    assert len(page.dataframe[0].value) == 24
    page.radio(key="v5_explorer_horizon").set_value(48).run()
    assert not page.exception and len(page.dataframe[0].value) == 48
    assert page.selectbox(key="catalog_explorer_version").options == ["V5"]
    assert page.selectbox(key="source_mode").options == ["Catálogo V1–V5"]
    assert any("V2–V4 têm apenas métricas agregadas" in item.value for item in page.caption)


def test_metrics_use_real_row_keys_and_paired_source():
    data = versions.metric_tables(str(versions.root()))
    assert set(data["globais"]["versao"]) == {*versions.VERSIONS, "piloto"}
    assert len(data["globais"]) == 540
    assert len(data["pares"]) == 118
    app = AppTest.from_string(
        "from components.model_versions_views import render_evolution\nrender_evolution()",
        default_timeout=60,
    ).run()
    assert not app.exception
    global_table, paired = (app.dataframe[j].value for j in (0, 1))
    assert pd.isna(global_table.loc[global_table["Versão"].eq("V1"), "MAE (MWmed)"].iloc[0])
    assert paired["n_pares_comuns"].tolist() == [104760]
    assert paired["delta_mae_mwmed"].iloc[0] == data["pares"].loc[
        data["pares"]["cenario"].eq("hipotese_UTC") & data["pares"]["horizonte_h"].eq(24)
        & data["pares"]["etapa"].eq("teste_2025") & data["pares"]["modelo"].eq("ols_estabilizado"),
        "delta_mae_mwmed"].iloc[0]


def test_audited_calendar_uses_versioned_ids_and_separate_pending_rows():
    app = AppTest.from_string(
        "from app_pages.sazonalidade import _render_profiles\n"
        "from components.model_versions_views import render_seasonality\n"
        "render_seasonality(_render_profiles)",
        default_timeout=60,
    ).run()
    assert not app.exception and len(app.get("plotly_chart")) >= 6
    assert any("evento_versao_id" in df.value for df in app.dataframe)
    assert any("motivo_pendencia" in df.value for df in app.dataframe)


def test_clean_checkout_ships_real_v5_sample(monkeypatch, tmp_path):
    monkeypatch.setenv("ENERGY_FORECAST_DETAIL_DIR", str(tmp_path))
    monkeypatch.setenv("ENERGY_FORECAST_DETAIL_URL", "")
    index = versions.detail_index()
    assert len(index) == 12 and all(item["preview"] for item in index)
    assert sum(item["rows"] for item in index) == 432
    page = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
    assert not page.exception and len(page.dataframe[0].value) == 24
    assert any("12 emissões reais" in item.value for item in page.caption)
    assert page.selectbox(key="catalog_explorer_version").options == ["V5"]
