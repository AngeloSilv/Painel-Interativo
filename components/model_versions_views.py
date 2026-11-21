"""Controles e visões do catálogo experimental V1–V5."""

from datetime import timedelta
from urllib.error import URLError

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import explorer_chart
from components.metrics_cards import format_number, metric_cards
from components.page_frame import DAY_LABELS, SCOPE_LABELS
from components.tables import csv_bytes, data_table
from services import model_versions as versions
from services.analysis_service import temperature_at_issue
from services.forecast_catalog import forecast_catalog


def choice(label, options, key, labels=None):
    return st.selectbox(label, options, key=key, format_func=(labels or {}).get if labels else str)


def header(title):
    st.title(title)
    st.caption("ONS Sudeste/Centro-Oeste · ONS_SECO_piloto · MWmed")


@st.cache_resource(show_spinner=False)
def _legacy(path, sha):
    bundle = versions.legacy_bundle(path, sha)
    return bundle, forecast_catalog(bundle)


def render_explorer(legacy_render):
    header("Explorador de Previsões")
    available = ["V5", "V1"] if versions.legacy_available() else ["V5"]
    version = choice("Versão", available, "catalog_explorer_version")
    caption = "V2–V4 têm apenas métricas agregadas, em Desempenho e Erros."
    if "V1" not in available:
        caption += " O detalhe V1 está disponível em instalações com o ZIP legado."
    st.caption(caption)
    try:
        if version == "V1":
            from components.sidebar import SidebarSettings

            path = versions.legacy_path()
            if path is None:
                st.info("Previsões V1 indisponíveis nesta instalação. Configure o pacote legado descrito no guia de integração.")
                return
            bundle, catalog = _legacy(path, versions.LEGACY_SHA)
            if catalog.empty:
                st.warning("O pacote legado V1 não pôde ser carregado.")
                return
            st.caption("V1 · 2025/2026 · carimbo original interpretado em São Paulo para exibição")
            with st.expander("Limitações da V1"):
                st.write("O fuso original não está documentado. O horário exibido usa uma interpretação do leitor legado. A emissão e o alvo não devem ser pareados com V2–V5 por esses carimbos.")
            settings = SidebarSettings("v1", 24, versions.REGION)
            legacy_render((bundle, settings), catalog)
            return
        _render_v5()
    except (OSError, ValueError, KeyError, URLError) as error:
        st.warning(f"Fonte detalhada indisponível: {error}")


def _render_v5():
    st.caption("V5 · retrospectiva sobre snapshots fechados")
    index = versions.detail_index()
    if not index:
        st.info("O índice das previsões detalhadas V5 está indisponível. As métricas e a série auditada continuam disponíveis nas outras abas.")
        return
    if index[0]["preview"]:
        st.caption("Amostra V5: 12 emissões reais do manifesto, uma por hipótese, etapa e horizonte. As métricas usam as amostras completas declaradas.")
    columns = st.columns([1.3, 1.2, 0.8, 0.9])
    with columns[0]:
        scenario = choice("Hipótese de horário", list(versions.SCENARIOS)[:2], "v5_explorer_scenario", versions.SCENARIOS)
    with columns[1]:
        stage = choice("Conjunto", list(versions.STAGES), "v5_explorer_stage", versions.STAGES)
    with columns[2]:
        horizon = st.radio("Horizonte", [24, 48], horizontal=True, key="v5_explorer_horizon")
    with columns[3]:
        history_hours = st.selectbox("Histórico anterior", [72, 48, 120, 168], format_func=lambda v: f"{v} h", key="v5_explorer_history")
    entries = {e["month"]: e for e in index if (e["scenario"], e["stage"], e["horizon"]) == (scenario, stage, horizon)}
    with st.expander("Detalhes metodológicos"):
        st.write("As duas hipóteses têm o mesmo estatuto. O carimbo oficial e sua borda seguem indeterminados. Emissões e alvos são armazenados em UTC e exibidos em America/Sao_Paulo. Eventos versionados refletem a informação disponível na origem segundo a hipótese do pacote.")
    if not entries:
        st.info("Nenhuma emissão no recorte selecionado.")
        return
    month = choice("Mês da emissão", sorted(entries, reverse=True), "v5_month")
    entry = entries[month]
    path = versions.partition_path(entry)
    catalog = versions.issues_for_models(path, entry["sha256"])
    models = sorted(catalog["modelo"].unique(), key=lambda name: (not catalog.loc[catalog["modelo"].eq(name), "eh_selecionado"].eq(1).any(), name))
    model = choice("Estimador", models, f"v5_model_{scenario}_{stage}_{horizon}_{month}")
    options = catalog.loc[catalog["modelo"].eq(model), "origem_t"]
    times = pd.to_datetime(options, utc=True).dt.tz_convert(versions.TZ)
    dates = sorted(times.dt.date.unique(), reverse=True)
    cols = st.columns(2)
    with cols[0]:
        day = st.selectbox("Data da emissão", dates, format_func=lambda value: value.strftime("%d/%m/%Y"), key="v5_date")
    with cols[1]:
        issue = st.selectbox("Hora / execução", sorted(options.loc[times.dt.date.eq(day)], reverse=True),
                             format_func=lambda value: pd.Timestamp(value).tz_convert(versions.TZ).strftime("%H:%M %z"), key="v5_issue")
    data = versions.forecast_detail(path, entry["sha256"], horizon, model, issue)
    observed = versions.load_scenario(str(versions.root()), scenario)
    t0 = pd.Timestamp(issue)
    selected_model = bool(data["eh_selecionado"].iloc[0])
    st.caption(f"{'Selecionado' if selected_model else ('Piloto' if model == 'piloto_so_carga' else 'Alternativa')} · emissão {t0.tz_convert(versions.TZ):%d/%m/%Y %H:%M %z}")
    _render_forecast(data, observed, t0, horizon, scenario, stage, history_hours)


def _render_forecast(data, observed, t0, horizon, scenario, stage, history_hours):
    future = data.rename(columns={"instante_alvo": "timestamp", "y_real": "observed_load",
                                  "y_pred": "predicted_load", "y_baseline_semanal": "seasonal_load"})
    future = future.assign(timestamp=pd.to_datetime(future["timestamp"], utc=True), phase="previsão")
    previous = observed.observations
    previous = previous.loc[(previous["timestamp"] <= t0) & (previous["timestamp"] > t0 - timedelta(hours=history_hours))]
    columns = ["timestamp", "observed_load", "predicted_load", "seasonal_load", "phase"]
    frame = pd.concat([previous.assign(predicted_load=float("nan"), seasonal_load=float("nan"), phase="histórico")[columns], future[columns]], ignore_index=True)
    peak = future.loc[future["predicted_load"].idxmax()]
    temperature = temperature_at_issue(observed.weather, t0, versions.REGION)
    metric_cards([
        {"label": "Pico previsto", "value": f"{format_number(peak.predicted_load)} MWmed"},
        {"label": "Horário do pico", "value": peak.timestamp.tz_convert(versions.TZ).strftime("%H:%M · %d/%m")},
        {"label": "Temperatura em t₀", "value": f"{format_number(temperature['temperature_c'])} °C" if temperature else "N/D"},
        {"label": "MAE da emissão", "value": f"{format_number((future.predicted_load - future.observed_load).abs().mean(), 2)} MWmed"},
    ])
    figure = explorer_chart(frame, t0)
    figure.add_trace(go.Scatter(x=future["timestamp"].dt.tz_convert(versions.TZ), y=data["y_baseline_persistencia"],
                                name="Persistência", mode="lines", line={"dash": "dot"}))
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    with st.expander("Valores horários e calendário"):
        labels = [versions.classify(row, observed.events) for row in data.itertuples(index=False)]
        exported = data.assign(versao="v5", alvo_id=versions.TARGET, horizonte_h=horizon, cenario=scenario,
                               etapa=stage, unidade="MWmed")
        display = exported.assign(calendario=[DAY_LABELS[x[0]] for x in labels],
                                  ambito=[SCOPE_LABELS[x[1]] for x in labels],
                                  eventos=[x[2] for x in labels])
        display["instante_local"] = future["timestamp"].dt.tz_convert(versions.TZ).dt.strftime("%d/%m/%Y %H:%M %z")
        data_table(display[["passo_h", "instante_local", "y_real", "y_pred", "y_baseline_semanal",
                            "y_baseline_persistencia", "calendario", "ambito", "eventos", "evento_versao_ids"]])
        st.download_button("Exportar emissão V5", csv_bytes(display), file_name=f"v5_{horizon}h.csv", mime="text/csv")


def _filter(frame, horizon, stage, scenario):
    return frame.loc[frame["horizonte_h"].eq(horizon) & frame["etapa"].eq(stage) & frame["cenario"].eq(scenario)]


def render_evolution():
    header("Desempenho e Erros")
    st.subheader("Evolução dos Modelos")
    try:
        tables = versions.metric_tables(str(versions.root()))
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"Métricas indisponíveis: {error}")
        return
    cols = st.columns([1, 1.4, 1.4, 1])
    with cols[0]:
        horizon = st.radio("Horizonte", [24, 48], horizontal=True, key="evolution_horizon")
    with cols[1]:
        scenario = choice("Hipótese de horário", list(versions.SCENARIOS), "evolution_scenario", versions.SCENARIOS)
    with cols[2]:
        stage = choice("Etapa", list(versions.STAGES), "evolution_stage", versions.STAGES)
    with cols[3]:
        block = choice("Bloco", ["1–24 h"] if horizon == 24 else ["1–48 h", "1–24 h", "25–48 h"], "evolution_block")
    roles = ["Selecionado", "Alternativa", "Piloto", "Referência"]
    left, right = st.columns(2)
    with left:
        role = choice("Grupo de estimadores", roles, "evolution_role")
    with right:
        metric = choice("Indicador dos gráficos", list(versions.METRICS), "evolution_metric", versions.METRICS)
    population = _filter(tables["globais"], horizon, stage, scenario)
    population = population.loc[population["bloco_horas"].eq(block) & population["papel"].eq(role)]
    model = None
    if role != "Selecionado":
        models = sorted(population["modelo"].unique())
        if models:
            model = choice("Estimador", models, "evolution_model")
            population = population.loc[population["modelo"].eq(model)]
    order = list(versions.VERSIONS)
    if role == "Piloto" and scenario == "sem_cenario_bruto":
        order = ["piloto", *order]
    if population["versao"].duplicated().any():
        st.warning("Há mais de uma linha para a mesma versão no recorte. Selecione um estimador único.")
        return
    display = population.set_index("versao").reindex(order).reset_index()
    display["versao"] = display["versao"].str.upper()
    st.caption("Métricas publicadas por versão. O recorte V4 × V5 usa a interseção de pares fornecida na seção abaixo.")
    if display[metric].notna().any():
        fig = px.scatter(display, x="versao", y=metric, hover_data=["modelo", "n_pares"],
                         labels={"versao": "Versão", metric: versions.METRICS[metric]})
        fig.update_traces(marker_size=10)
        fig.update_layout(height=300, margin={"t": 15})
        fig.update_xaxes(categoryorder="array", categoryarray=[v.upper() for v in order])
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    data_table(display[["versao", "modelo", "n_pares", *versions.METRICS]].rename(columns={
        "versao": "Versão", "modelo": "Estimador", "n_pares": "Pares", **versions.METRICS
    }), decimals=4, integer_columns=("Pares",))
    _step_chart(tables["por_passo"], horizon, stage, scenario, block, role, model, metric)
    _paired_section(tables["pares"], horizon, stage, scenario, block, role, model, population)
    with st.expander("Métricas por hora ou mês do alvo"):
        granularity = choice("Agrupamento civil", ["por_hora", "por_mes"], "evolution_civil",
                             {"por_hora": "Hora do dia", "por_mes": "Mês"})
        civil_key = "hora_alvo_local" if granularity == "por_hora" else "mes_alvo_local"
        civil = _filter(tables[granularity], horizon, stage, scenario)
        civil = civil.loc[civil["bloco_horas"].eq(block) & civil["papel"].eq(role)]
        if model is not None:
            civil = civil.loc[civil["modelo"].eq(model)]
        if civil.empty:
            st.caption("N/D neste recorte.")
        else:
            data_table(civil[["versao", "modelo", civil_key, "n_pares", *versions.METRICS]], height=320)
    with st.expander("Disponibilidade e detalhes metodológicos"):
        st.write("V1 tem carimbo bruto sem fuso documentado e fica em recorte próprio. V2–V4 incluem agregados; V1 e V5 têm detalhe quando a fonte correspondente está configurada. As duas hipóteses de V2–V5 seguem abertas.")
        st.write("N/D representa ausência no recorte. MAE e RMSE em MWmed; MAPE em %. Os blocos 1–24 e 25–48 pertencem ao modelo de 48 h. Validação, teste e retrospectiva de 2026 são exibidos separadamente.")
        availability = tables["disponibilidade"]
        subset = _filter(availability, horizon, stage, scenario)
        if not subset.empty:
            data_table(subset[["versao", "modelo", "tabela", "campo", "status", "motivo"]], height=320)


def _step_chart(frame, horizon, stage, scenario, block, role, model, metric):
    st.markdown("#### Erro por passo")
    lo, hi = (25, 48) if block == "25–48 h" else (1, 48) if block == "1–48 h" else (1, 24)
    steps = _filter(frame, horizon, stage, scenario)
    steps = steps.loc[steps["papel"].eq(role) & steps["passo_h"].between(lo, hi)]
    if model is not None:
        steps = steps.loc[steps["modelo"].eq(model)]
    if steps.empty or not steps[metric].notna().any():
        st.caption("N/D neste recorte.")
        return
    fig = go.Figure()
    for (v, name), run in steps.groupby(["versao", "modelo"], sort=True):
        grid = run.set_index("passo_h").reindex(range(lo, hi + 1))
        fig.add_trace(go.Scatter(x=grid.index, y=grid[metric], name=f"{v.upper()} · {name}", mode="lines+markers",
                                 connectgaps=False, customdata=grid["n_pares"],
                                 hovertemplate="%{x} h · %{y:.4f}<br>Pares: %{customdata}<extra>%{fullData.name}</extra>"))
    fig.update_layout(xaxis_title="Passo (h)", yaxis_title=versions.METRICS[metric], height=340, margin={"t": 15})
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _paired_section(frame, horizon, stage, scenario, block, role, model, population):
    st.markdown("#### V4 × V5 · pares comuns")
    pairs = _filter(frame, horizon, stage, scenario)
    pairs = pairs.loc[pairs["bloco_horas"].eq(block)]
    if role == "Selecionado":
        v4 = population.loc[population["versao"].eq("v4"), "modelo"]
        v5 = population.loc[population["versao"].eq("v5"), "modelo"]
        pairs = pairs.loc[pairs["modelo"].isin(v5) & pairs["modelo_ref"].isin(v4)]
    elif model is not None:
        pairs = pairs.loc[pairs["modelo"].eq(model)]
    else:
        pairs = pairs.iloc[:0]
    if pairs.empty:
        st.caption("N/D para a seleção atual.")
        return
    st.caption("Valores do arquivo de pares comuns. Deltas são V5 − V4. Δ MAPE em pontos percentuais.")
    fields = ["tipo_comparacao", "modelo_ref", "modelo", "n_pares_comuns", "n_mape_validos_comuns",
              "n_excluidos_v4", "n_excluidos_v5", "mae_mwmed_v4", "mae_mwmed_v5", "delta_mae_mwmed",
              "rmse_mwmed_v4", "rmse_mwmed_v5", "delta_rmse_mwmed", "mape_pct_v4", "mape_pct_v5", "delta_mape_pp"]
    data_table(pairs[fields], decimals=4)


def render_seasonality(render_profiles):
    from components.sidebar import SidebarSettings
    from services.data_loader import DataBundle

    header("Sazonalidade, Clima e Calendário")
    st.caption("Série horária auditada V5")
    scenario = choice("Hipótese de horário", list(versions.SCENARIOS)[:2], "v5_patterns_scenario", versions.SCENARIOS)
    station = choice("Estação", ["Média disponível", "A621", "A601"], "v5_patterns_station",
                     {"Média disponível": "Média disponível", "A621": "A621 · Vila Militar", "A601": "A601 · Seropédica"})
    try:
        value = versions.load_scenario(str(versions.root()), scenario)
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"Série auditada indisponível: {error}")
        return
    settings = SidebarSettings(scenario, 24, versions.REGION,
                               station=None if station == "Média disponível" else station)
    bundle = DataBundle("v5", [], value.observations, [], weather=value.weather, calendar=value.calendar)
    render_profiles((bundle, settings))
    with st.expander("Inventário versionado e pendências"):
        st.write("A série usa os IDs ativos na própria hora. O alcance de cada ato consta no inventário. As pendências abaixo não entram nas classificações.")
        data_table(pd.DataFrame(list(value.events.values()))[["evento_versao_id", "data_local", "nome_evento", "esfera",
                                                      "alcance", "conhecido_desde", "conhecido_ate", "fonte_url"]], height=300)
        data_table(value.pending, height=220)
