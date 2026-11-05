# SolarDesk

Aplicativo desktop em Python para importar dados fotovoltaicos, acompanhar potência e irradiância, estimar a geração por ensemble de XGBoost e CatBoost combinado por regressão Ridge e sinalizar possíveis desvios.

## Instalação no Windows

1. Instale Python 3.12 de 64 bits, incluindo o Python Launcher.
2. Baixe e extraia o código deste repositório.
3. Execute `Instalar.cmd` com acesso à internet para instalar as dependências.
4. Execute `Iniciar.cmd` para abrir o aplicativo.

Após a instalação, o processamento é local. Esta versão exige Python: não é um executável independente.

## Dados e uso

Importe CSVs com as colunas obrigatórias `Datetime`, `POA` (W/m²) e `Pdc1` (W), em intervalos de 15 minutos. Datas sem fuso são interpretadas como UTC−03. As colunas `Vdc1`, `Idc1`, `Validacao` e `Tar med` são opcionais. O importador não converte automaticamente outros formatos de usina ou unidades.

Novos registros são incorporados ao histórico SQLite. Repetições idênticas não duplicam as medições; conflitos no mesmo horário recusam o arquivo. A importação atualiza as previsões quando há modelo ativo. O retreinamento é iniciado pelo usuário em **Nowcasting → Treinar nova versão** e a ativação depende da avaliação da candidata.

As medições, configurações e versões dos modelos ficam na pasta local `data/`. Esta distribuição inclui o histórico inicial e os modelos treinados da instalação de pesquisa, permitindo abrir o aplicativo com dados e previsões disponíveis após instalar as dependências.

### Base e modelos incluídos

- `examples/Base_PVlib_v6.csv`: base sintética com dados originais de irradiância no plano dos módulos de Campinas, ano 2020, utilizada no desenvolvimento do ensemble. Contém 35.136 registros de 15 minutos. Não deve ser confundida com a base própria da UNIVASF ou a base operacional de múltiplos inversores da UNICAMP.
- `data/historico.sqlite3`: cópia consistente do histórico importado e das configurações da instalação inicial.
- `data/models/`: modelos nativos XGBoost e CatBoost, coeficientes do combinador, metadados de avaliação e previsões de teste.

A base original está incluída para permitir reimportação e reprodução. Dados novos importados pelo usuário alteram o banco local. Logs e arquivos temporários continuam excluídos do Git. Não é necessário copiar o ambiente `.venv`: `Instalar.cmd` o recria.

## Escopo e limitações

- O ensemble utiliza irradiância contemporânea e histórico de potência: é uma estimativa de nowcasting, não uma previsão futura sem essas entradas.
- O treinamento separa cronologicamente ajuste dos modelos, parada antecipada, combinação Ridge, calibração e teste.
- O PR depende da confirmação da potência nominal correspondente aos dados importados.
- O cálculo de FF ainda não está disponível nesta versão.
- Não há importador genérico nem análise individual de múltiplos inversores nesta interface. O formato atual usa uma coluna de potência.
- Alertas sinalizam condições candidatas à investigação; não confirmam sujidade ou falhas.

## Código e testes

- `app.py`: interface Qt e tarefas em segundo plano.
- `solar_core.py`: importação, banco local, indicadores e alertas.
- `solar_ml.py`: treinamento, avaliação e gerenciamento dos modelos.
- `tests/test_core.py`: testes automatizados.

Execute os testes no ambiente instalado:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O documento [Histórico da versão](docs/HISTORICO_DA_VERSAO.md) registra a instalação original e suas avaliações. A base e os modelos dessa instalação estão incluídos; o ambiente Python deve ser instalado em cada computador. A procedência da base é esclarecida acima, atualizando a dúvida registrada no documento histórico.

## Origem e distribuição

Software desenvolvido no contexto da pesquisa de iniciação em desenvolvimento tecnológico de Ítalo Gustavo Vieira Souza, na UNIVASF, com apoio PIBITI/CNPq. Houve assistência de inteligência artificial na implementação e revisão de código e documentação.

Uma licença de distribuição ainda não foi definida. A disponibilização do código não substitui a definição dos direitos de uso ou das permissões de compartilhamento das bases utilizadas na pesquisa.
