# Integração dos modelos e dados reais

1. Exporte um JSON por emissão/modelo/horizonte em `data/predictions/`, com nomes distintos para preservar execuções anteriores.
2. Coloque a carga correspondente em `data/historical/`, a meteorologia em `data/weather/` e o calendário municipal verificado em `data/calendar/`.
3. Selecione **Arquivos dos modelos** e a região/unidade/estação na barra lateral. Atualize a página após incluir ou corrigir arquivos.

O modo real nunca recorre aos dados sintéticos. Sem meteorologia, carga e métricas continuam disponíveis; os campos climáticos ficam vazios. Sem calendário, os dias ficam **não informados**. Não se presumem dias comuns nem feriados a partir da ausência de registros.

O grupo deve verificar a procedência municipal da carga e das estações, e as fontes oficiais e datas aplicáveis dos feriados e pontos facultativos. Esta versão não baixa calendários ou meteorologia e não certifica essas fontes.

### JSON das previsões

| Campo | Regra |
| --- | --- |
| `model_name`, `model_version`, `region` | Texto não vazio |
| `horizon_hours` | Inteiro 24 ou 48 |
| `unit` | `MWmed` para o alvo definido; `MW` aceito como série distinta, sem conversão |
| `issued_at` | Emissão original ISO 8601 com fuso explícito |
| `dataset_split` | `validation_2025`, `test_2025`, `forecast_2026` ou `unspecified` |
| `predictions` | Exatamente 24 ou 48 objetos ordenados |
| `predictions[].timestamp` | Alvo ISO 8601 com fuso explícito |
| `predictions[].predicted_load` | Número finito, não negativo, na unidade declarada |

`dataset_split` é opcional para compatibilidade: quando ausente fica `unspecified`, sem inferência pelo ano. Esse rótulo é declarado pelo produtor; não prova separação de treino/teste.

Os alvos devem ser únicos após normalização UTC, consecutivos a cada hora, posteriores a t₀ e com o primeiro alvo no intervalo `(t₀, t₀ + 1h]`. Emissão fora da hora cheia é aceita. Valores booleanos, NaN, infinitos, chaves JSON repetidas e campos obrigatórios ausentes são recusados. Previsões inválidas não são reordenadas, interpoladas ou truncadas.

### Exemplos completos de 24h e 48h

Em uma cópia de desenvolvimento sem fixtures, execute:

```powershell
.\.venv\Scripts\python.exe -m services.demo_data
.\.venv\Scripts\python.exe -c "from services.data_loader import load_forecast; from pathlib import Path; paths=sorted(Path('data/demo').glob('forecast_test_2025_09_*.json')); assert len(paths)==2; print([(p.name, load_forecast(p).horizon_hours) for p in paths])"
```

Os arquivos `forecast_test_2025_09_24h.json` e `forecast_test_2025_09_48h.json` são exemplos sintéticos completos com emissões independentes por horizonte. O gerador preserva arquivos existentes por padrão.

Para validar um arquivo exportado, na raiz:

```powershell
.\.venv\Scripts\python.exe -c "from services.data_loader import load_forecast; print(load_forecast('data/predictions/seu_arquivo.json'))"
```

O CSV baixado pelo painel inclui todos os alvos, emissão UTC e texto original, nome/versão do modelo, horizonte, região, unidade, partição e nome do arquivo de origem.

### CSVs de carga e meteorologia

Use UTF-8, separador vírgula e ponto decimal. Cabeçalhos obrigatórios:

```csv
timestamp,observed_load,region,unit
```

```csv
timestamp,temperature_c,station,region
```

`timestamp` deve conter data/hora ISO 8601 com fuso. `region` deve corresponder exatamente ao JSON. Carga deve ser finita e não negativa; temperatura, finita entre −60°C e 60°C. Estação e região não podem ser vazias.

Cada série de carga é identificada por região **e unidade**, com no máximo um registro por instante. A grade deve ser horária; lacunas de horas inteiras são aceitas. MW e MWmed coexistem como séries distintas, sem serem combinados. Meteorologia permite no máximo uma leitura por região/estação/instante. Sobreposições entre arquivos invalidam a origem ambígua para impedir seleção arbitrária de valores.

### CSV de calendário

```csv
date,region,day_type,holiday_scope,event_name
```

- `date`: data local `YYYY-MM-DD`, sem horário, única por região.
- `day_type`: `dia_util`, `fim_de_semana`, `feriado` ou `ponto_facultativo`.
- `holiday_scope`: `nacional`, `estadual`, `municipal`, `nenhum` ou `nao_informado`.
- `event_name`: obrigatório em feriados/pontos facultativos; pode ficar vazio nos dias comuns.

Forneça uma sequência diária completa no intervalo coberto por região, incluindo dias comuns. Feriados exigem âmbito nacional/estadual/municipal; pontos facultativos podem indicar o âmbito aplicável. Dias comuns usam `nenhum`. Classificações contraditórias ou dias duplicados são recusados. Datas fora da cobertura permanecem não informadas. Os eventos da pasta de demonstração não devem ser copiados como calendário oficial.

## Entrega pelos responsáveis dos modelos

Entregue um JSON por emissão, horizonte e versão. Registre treino, validação, teste e previsão em um manifesto de proveniência do experimento, incluindo período, método de geração, fontes, unidade, fuso e disponibilidade das entradas em t₀. O campo `dataset_split` classifica os resultados aceitos no painel e não tem valor para treino. Preserve a série original e seu recorte municipal comprovado. A previsão de 24h deve ser uma execução própria do modelo de 24h.

Para publicação, inclua somente arquivos cuja redistribuição foi aprovada. Registre a evidência de abrangência municipal, a licença ou permissão e a cobertura de carga, estações e calendário. Dependências de inferência são necessárias apenas se o grupo decidir executar futuramente os modelos no painel. O fluxo atual consome previsões prontas.
