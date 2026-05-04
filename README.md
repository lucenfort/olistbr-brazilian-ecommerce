# Olist E-commerce Analytics

<p align="center">
  <img src="./assets/banner.svg" alt="Project Banner" width="100%" />
</p>

<p align="left">
	<img src="https://img.shields.io/badge/Python-3.12-FFD700?style=for-the-badge&logo=python&logoColor=111111&labelColor=0B0B0B" alt="Python" />
	<img src="https://img.shields.io/badge/Polars-1.0+-00FFF7?style=for-the-badge&logo=polars&logoColor=111111&labelColor=0B0B0B" alt="Polars" />
	<img src="https://img.shields.io/badge/DuckDB-1.0+-FF00FF?style=for-the-badge&logo=duckdb&logoColor=111111&labelColor=0B0B0B" alt="DuckDB" />
	<img src="https://img.shields.io/badge/Status-Estável-9F00FF?style=for-the-badge&logoColor=111111&labelColor=0B0B0B" alt="Status" />
</p>

Análise de dados end-to-end do dataset público "Brazilian E-Commerce Public Dataset by Olist". Este projeto implementa um pipeline de dados robusto, desde o ETL até a modelagem preditiva de atrasos em entregas, utilizando as ferramentas mais modernas do ecossistema Python.

## [>] SYS.NAVEGAÇÃO

[Visão Geral](#-visão-geral) • [Stack](#-arquitetura-e-stack-tecnológica) • [Estrutura](#-estrutura-do-projeto) • [Execução](#-como-executar) • [Resultados](#-principais-resultados)

---

## [~] VISÃO_GERAL

A solução abrange informações sobre aproximadamente 100.000 pedidos realizados no Brasil entre 2016 e 2018.
- **ETL Avançado**: Download automatizado e ingestão em DuckDB com processamento Polars.
- **Análise Estratégica**: Segmentação RFM, análise de retenção (cohort) e satisfação.
- **Machine Learning**: Treinamento de XGBoost e LightGBM com otimização Optuna.

## [@] ARQUITETURA_E_STACK

- **Processamento**: `Polars` & `DuckDB` (Vetorização orientada a colunas).
- **Machine Learning**: `Scikit-Learn`, `XGBoost`, `LightGBM`.
- **Otimização**: `Optuna` (Busca Bayesiana de hiperparâmetros).
- **Visualização**: `Plotly` (Gráficos interativos de alta fidelidade).

## [=] ESTRUTURA_PROJETO

```
olistbr-brazilian-ecommerce/
├── assets/               # HUDs e Banner Cyberpunk
├── data/                 # Diretório de dados brutos
├── resultado/            # Gráficos, Modelos e Relatórios
├── src/                  # Código fonte modular
│   ├── analysis/         # EDA e Business Logic
│   ├── etl/              # Download, Ingestão e Transformação
│   ├── models/           # Treinamento e Avaliação ML
│   ├── visualization/    # Geração de Gráficos e Dashboards
│   ├── config.py         # Configurações globais
│   └── main.py           # Orquestrador do pipeline
├── requirements.txt      # Dependências do sistema
└── README.md             # Documentação técnica
```

---

## [*] INSTALAÇÃO_E_EXECUÇÃO

### 1. Clonar o sistema
```bash
git clone https://github.com/lucenfort/olistbr-brazilian-ecommerce.git
cd olistbr-brazilian-ecommerce
```

### 2. Configurar Ambiente
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 3. Executar Pipeline
```bash
python src/main.py
```

O script automatiza o download via API do Kaggle, processa os dados e gera relatórios completos em `resultado/`.

---

## [#] PRINCIPAIS_RESULTADOS

### Insights de Negócio
- **Faturamento**: R$ 13.278.587,41 analisados.
- **Recorrência**: 3,0% (Modelo altamente transacional).
- **Clusters**: Segmentação RFM identificou 3 perfis distintos de clientes através do método Silhouette.

### Performance Preditiva (Atrasos)
- **Melhor Modelo**: XGBoost
- **ROC-AUC**: 0.7908
- **PR-AUC**: 0.3596

---

