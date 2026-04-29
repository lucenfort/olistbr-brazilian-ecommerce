# Análise de E-commerce Brasileiro Olist

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Polars](https://img.shields.io/badge/Polars-1.0+-orange?style=for-the-badge&logo=polars)
![DuckDB](https://img.shields.io/badge/DuckDB-1.0+-yellow?style=for-the-badge&logo=duckdb)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-XGBoost_%7C_LightGBM-green?style=for-the-badge)

## 📌 Visão Geral

Este projeto consiste em uma análise de dados end-to-end do dataset público **"Brazilian E-Commerce Public Dataset by Olist"**, que abrange informações sobre aproximadamente 100.000 pedidos realizados no Brasil entre 2016 e 2018.

A solução implementa um pipeline completo contemplando:
- **Extração, Transformação e Carga (ETL)**: Download automatizado, ingestão em banco de dados analítico e processamento eficiente de grandes volumes de dados.
- **Análise Exploratória de Dados (EDA)**: Estatísticas descritivas e descoberta de padrões de negócio.
- **Análise Estratégica**: Segmentação RFM, análise de retenção (cohort) e correlação de satisfação.
- **Machine Learning**: Treinamento, otimização hiperparamétrica e avaliação de múltiplos algoritmos preditivos (Logistic Regression, Random Forest, XGBoost e LightGBM) para previsão de atrasos em entregas.

## 🚀 Arquitetura e Stack Tecnológica

O projeto foi refatorado para utilizar as melhores práticas e ferramentas modernas de processamento de dados em Python, substituindo stacks tradicionais (Pandas/SQLite) por soluções vetorizadas e orientadas a colunas:

- **Processamento e ETL**: `Polars` (manipulação de DataFrames multithreaded) e `DuckDB` (banco de dados analítico embutido).
- **Machine Learning**: `Scikit-Learn`, `XGBoost`, `LightGBM`, `Imbalanced-Learn` (SMOTE).
- **Otimização Bayesiana**: `Optuna` (busca inteligente de hiperparâmetros).
- **Visualização**: `Plotly` e `Kaleido` (gráficos interativos e exportação profissional).

## 📊 Principais Resultados

### Análise de Negócio
- **Volume**: 96.470 pedidos válidos analisados, totalizando um faturamento de R$ 13.278.587,41.
- **Retenção**: A taxa de recorrência de clientes é de **3,0%**, evidenciando um modelo de negócio altamente transacional (baixa retenção).
- **Segmentação RFM**: Os clientes foram divididos em **3 clusters ótimos** (método Silhouette):
  - *Cluster 0*: Clientes padrão (1 compra, recência média de 133 dias, ticket médio ~R$80).
  - *Cluster 1*: Clientes inativos (1 compra, recência alta de 362 dias, ticket médio ~R$80).
  - *Cluster 2*: Clientes de alto valor (ticket médio ~R$250).

### Modelagem Preditiva (Previsão de Atraso)
O objetivo do modelo é prever se um pedido sofrerá atraso na entrega, utilizando apenas informações disponíveis no momento da compra (evitando rigorosamente o *data leakage* comum nesse dataset).

O **XGBoost** foi selecionado como o melhor algoritmo, superando os demais nas métricas de teste:
- **ROC-AUC**: 0.7908
- **F1-Score**: 0.2444 (devido ao alto desbalanceamento, tratável no threshold de decisão do negócio)
- **PR-AUC**: 0.3596

## 📂 Estrutura do Projeto

```text
olistbr-brazilian-ecommerce/
├── src/
│   ├── config.py              # Constantes e configurações globais
│   ├── etl/
│   │   ├── download.py        # Coleta automatizada do Kaggle
│   │   ├── ingest.py          # Ingestão vetorizada DuckDB
│   │   └── transform.py       # Limpeza e Feature Engineering (sem leakage)
│   ├── analysis/
│   │   ├── eda.py             # Análise exploratória
│   │   └── business.py        # Lógica de retenção e segmentação RFM
│   ├── models/
│   │   ├── preprocessing.py   # Encoding, Scaling e SMOTE (somente no treino)
│   │   ├── training.py        # Otimização Bayesiana (Optuna) e Treinamento
│   │   └── evaluation.py      # Métricas, matrizes e learning curves
│   ├── visualization/
│   │   ├── charts.py          # Geração de +15 gráficos profissionais
│   │   └── dashboards.py      # Geração de relatórios em Markdown
│   └── main.py                # Orquestrador do pipeline (executar este arquivo)
├── data/                      # Diretório de dados brutos (baixados do Kaggle)
├── resultado/                 # Relatórios e artefatos gerados
│   ├── graficos/              # Gráficos estáticos e HTML de EDA e Business
│   ├── modelos/               # Curvas ROC, PR, Matrizes de Confusão e Feature Importance
│   └── relatorios/            # Relatório técnico Markdown final
└── requirements.txt           # Dependências do projeto
```

## ⚙️ Como Executar

O projeto requer o **Python 3.10+**.

1. **Clone o repositório:**
```bash
git clone https://github.com/lucenfort/olistbr-brazilian-ecommerce.git
cd olistbr-brazilian-ecommerce
```

2. **Crie um ambiente virtual e instale as dependências:**
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

3. **Execute o pipeline principal:**
```bash
python src/main.py
```

O script irá baixar o dataset via API do Kaggle, instanciar o DuckDB, processar as views com Polars, treinar os algoritmos e exportar gráficos/relatórios completos na pasta `resultado/`.
