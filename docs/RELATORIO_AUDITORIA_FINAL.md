# Auditoria final do Energy Forecast

## Escopo verificado

A [matriz de requisitos](MATRIZ_REQUISITOS.md) tem **134 requisitos**. São **129 dentro do escopo**, sendo 105 obrigatórios, 16 complementares e 8 condicionados aos dados. Dos cinco restantes, dois estão fora do escopo do painel e três são incompatíveis ou redundantes. A checklist original R001–R157 não estava disponível. A contagem usa a decomposição de todos os grupos do pedido de auditoria.

A revisão consultou a monografia de 21 páginas, o inventário `RELATORIO_VERSOES_MODELOS.md`, os dois ZIPs disponíveis, seus manifestos, o código, a documentação e os testes. A monografia (§§1.2–1.6) pede delimitação territorial pela série efetiva, análise temporal e meteorológica, avaliação posterior ao treino com referências e discussão dos limites. Os capítulos de desenvolvimento, testes e resultados ainda aparecem como títulos. A entrega independente V5 tem o mesmo SHA-256 do membro V5 no pacote consolidado. A leitura não alterou os ZIPs.

## Estado das três abas

| Área | Implementação anterior | O que a auditoria entregou |
| --- | --- | --- |
| Previsões | Filtros de t₀, horizonte, cenário, etapa e estimador na V5. Detalhe V1 disponível com o legado. Histórico, previsto, observado e referências. | Identificação das unidades na tabela, observado e sazonal semanal no leitor legado, contagem de calendário e MAE com denominador. A V5 oculta observações posteriores ao relógio da execução. A validação aparece no leitor quando existem emissões dessa etapa. |
| Desempenho e Erros | Agregados V1–V5, curva por passo, papéis dos estimadores e V4 × V5 por pares comuns. | Filtro de versão com cenários compatíveis, denominador do MAPE, transição 24/25 h, horizontes e referências lado a lado. Deltas relativos V4 × V5 usam o valor V4 nos pares comuns. Episódios consultáveis V1/V5, com recorte mensal, filtros no alvo e exportação. Resumo das mudanças técnicas recolhível. |
| Sazonalidade, Clima e Calendário | Perfis horário, semanal e mensal, carga × temperatura, resolução climática e inventário versionado. | Médias anuais com contagem de horas, extremos do perfil horário e seleção de categorias de calendário por tipo e âmbito. |
| Integridade e interface | Parquet mensal, cache, chaves de métrica e forma de distribuição compacta. | Validação de SHA-256 e contagem dos CSVs compactos durante o carregamento, mais domínio de passos/blocos. Os ajustes locais de navegação no topo, título **Previsões** e textos recolhidos foram preservados. |

A comparação entre horizontes é descritiva, com os denominadores de cada modelo e bloco. O teste dos dois modelos de 24 h e 48 h não prova equivalência entre o modelo de 24 h e o primeiro bloco de 48 h. O seletor de versão destaca um recorte, enquanto a visão de evolução reúne as versões elegíveis no cenário escolhido.

## Evidência de dados e desempenho

- A fonte consolidada histórica e os CSVs da V5 retornaram 540 linhas globais, 8.792 por passo, 8.832 por hora civil, 2.680 por mês civil e 778 registros de disponibilidade. Há 118 linhas no arquivo V4 × V5. As chaves de cada granularidade são únicas. A aba preserva a precisão dos agregados originais.
- A camada local tem 88 partições Parquet, 44 por horizonte. Foram conferidos SHA-256 e contagens. A leitura por partição confirmou 1.938.816 linhas de 24 h e 4.580.064 de 48 h com grade emissão–passo–alvo, horizonte completo e chave de passo sem duplicatas.
- Nos 6.518.880 registros, observado do alvo, persistência em t₀ e referência semanal no alvo menos 168 h coincidiram com a série auditada na tolerância de 10⁻⁸ MWmed. Os IDs da previsão resolveram no inventário do cenário e estavam no intervalo de vigência registrado para a emissão. Essa checagem usa o calendário entregue.
- Medição pontual local em partições grandes, após resolver o caminho, o hash e o estimador: 24 passos em 0,010 s e 48 passos em 0,007 s. Consultas dos episódios nos mesmos recortes retornaram 17.856 pares em 0,391 s e 35.712 pares em 0,063 s. A segunda consulta reutilizou cache de cenário. Essas são latências do serviço local, sem interface nem transferência remota.
- O leitor legado abriu as previsões V1. No teste de dezembro de 2025, o recorte mensal validado retornou 16.704 pares de 24 h e 33.408 de 48 h. A V5 é o padrão. A amostra pública contém 12 emissões de referência e 432 passos reais, com ambas as hipóteses, horizontes e as três etapas.

## Limites científicos e de produto

- O alvo é **ONS_SECO_piloto**, em MWmed, para o subsistema Sudeste/Centro-Oeste. Clima e calendário do Rio contextualizam a série regional. A linha de pesquisa municipal continua condicionada a medições municipais documentadas.
- A V1 tem carimbo original sem fuso confirmado. A V2–V5 conservam UTC e São Paulo como hipóteses. As previsões individuais V2–V4 e uma equivalência temporal documentada com V1 faltam para comparar curvas de previsão entre versões. A retrospectiva de 2026 usa snapshots fechados.
- A ausência de vazamento no treinamento exige dados, código e protocolo de treino e validação reproduzíveis. A verificação presente cobre os arquivos, a relação emissão–alvo, as referências de carga e os IDs conhecidos em t₀ na visão entregue. A validade jurídica dos atos e conclusões causais sobre clima/calendário ficam fora da verificação do aplicativo.
- Treinar o modelo dentro do Streamlit ou oferecer emissão operacional não faz parte da aplicação avaliada. Atribuir carga regional ao município, duplicar as métricas históricas ou gerar pares ausentes produziria resultados incompatíveis com a fonte.
- Alguns nomes técnicos preservam os identificadores da entrega. Elementos nativos do Streamlit podem usar o idioma do próprio framework.

## Execução e distribuição

O Windows executou Python 3.13.14. A suíte aprovou **466 testes** no ambiente local e **466** em uma cópia limpa da seleção do Git, com avisos de depreciação e sintaxe tratados como erros. `pip check` passou e o smoke HTTP registrou retorno 200 nas três rotas também na cópia limpa. No navegador, conferi as três abas e baixei o CSV de pares comuns. O CSV do teste de 2025, hipótese UTC, 24 h, continha 104.760 pares comuns e os deltas do estrato selecionado.

O verificador da seleção pública aprovou os 107 arquivos previstos para o Git. A busca por padrões usuais de credenciais na seleção textual não encontrou correspondências. Os ZIPs originais, os Parquets integrais e as bases privadas ficam fora da distribuição. O workflow executa os testes de portabilidade em Ubuntu e Windows. A execução Linux não foi feita neste computador. [Histórico de CI da branch](https://github.com/tccilda/Painel/actions?query=branch%3Afeat%2Fpainel-interativo)

A aplicação está preparada para Streamlit Community Cloud com `streamlit_app.py`, Python 3.13 e os dados compactos versionados. A oferta integral de emissões depende da configuração de armazenamento aprovada. A hospedagem não foi alterada nesta auditoria. Consulte o [guia de preparo e hospedagem](INTEGRACAO_VERSOES.md).

## Arquivos da auditoria

- **Aplicação:** `app_pages/explorador.py`, `app_pages/sazonalidade.py`, `components/charts.py`, `components/model_versions_views.py`, `services/model_versions.py`, `services/pattern_service.py`.
- **Navegação e seleção de fonte:** `streamlit_app.py` e `components/sidebar.py` já continham ajustes locais no início da auditoria.
- **Testes:** `tests/test_app.py`, `tests/test_model_versions.py`, `tests/test_provided_app.py`.
- **Documentação:** `README.md`, `docs/INTEGRACAO_VERSOES.md`, `docs/MATRIZ_REQUISITOS.md`, `docs/RELATORIO_AUDITORIA_FINAL.md`.
