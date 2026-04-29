#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline principal de análise do e-commerce brasileiro Olist.

Orquestra todas as etapas do projeto de ciência de dados:
1. ETL — Download, ingestão (DuckDB) e transformação (Polars)
2. Análise exploratória de dados
3. Análise de negócio (retenção, segmentação RFM, satisfação)
4. Modelagem preditiva com IA (4 modelos otimizados)
5. Geração de gráficos profissionais e relatório técnico
"""

import os
import sys
import time
import warnings

# Adiciona o diretório raiz ao path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

from src.config import RESULTS_DIR, CHARTS_DIR, MODELS_DIR, REPORTS_DIR
from src.etl.download import download_dataset
from src.etl.ingest import ingest_to_duckdb
from src.etl.transform import build_analytical_dataset, build_eda_dataset
from src.analysis.eda import run_full_eda
from src.analysis.business import run_business_analysis
from src.models.preprocessing import prepare_features
from src.models.training import train_all_models
from src.models.evaluation import evaluate_all_models
from src.visualization.charts import generate_all_charts
from src.visualization.dashboards import generate_report


def _setup_directories() -> None:
    """Cria os diretórios de saída."""
    for d in [RESULTS_DIR, CHARTS_DIR, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    """Executa o pipeline completo de análise."""
    start_total = time.time()

    print("=" * 70)
    print("  PIPELINE DE ANÁLISE — E-COMMERCE BRASILEIRO OLIST")
    print("=" * 70)
    print()

    _setup_directories()

    # ------------------------------------------------------------------
    # ETAPA 1: ETL
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  ETAPA 1: ETL (Extract, Transform, Load)")
    print("=" * 70)

    t0 = time.time()
    download_dataset()
    conn = ingest_to_duckdb()
    df_eda = build_eda_dataset(conn)
    df_model = build_analytical_dataset(conn)
    print(f"\n  ETL concluído em {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # ETAPA 2: Análise Exploratória
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  ETAPA 2: Análise Exploratória de Dados")
    print("=" * 70)

    t0 = time.time()
    eda_results = run_full_eda(df_eda)
    print(f"\n  EDA concluída em {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # ETAPA 3: Análise de Negócio
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  ETAPA 3: Análise de Negócio")
    print("=" * 70)

    t0 = time.time()
    business_results = run_business_analysis(df_eda)
    print(f"\n  Análise de negócio concluída em {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # ETAPA 4: Modelagem Preditiva
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  ETAPA 4: Modelagem Preditiva com IA")
    print("=" * 70)

    t0 = time.time()
    preprocessed = prepare_features(df_model)
    models = train_all_models(
        preprocessed["X_train"],
        preprocessed["y_train"],
    )
    eval_results = evaluate_all_models(
        models=models,
        X_test=preprocessed["X_test"],
        y_test=preprocessed["y_test"],
        X_train=preprocessed["X_train"],
        y_train=preprocessed["y_train"],
        feature_names=preprocessed["feature_names"],
    )
    print(f"\n  Modelagem concluída em {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # ETAPA 5: Visualização e Relatório
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  ETAPA 5: Geração de Gráficos e Relatório")
    print("=" * 70)

    t0 = time.time()
    generate_all_charts(eda_results, business_results, eval_results)
    report_path = generate_report(eda_results, business_results, eval_results)
    print(f"\n  Visualização concluída em {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------
    conn.close()

    total_time = time.time() - start_total
    print("=" * 70)
    print("  PIPELINE CONCLUÍDO")
    print("=" * 70)
    print(f"\n  Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Gráficos:    {CHARTS_DIR}")
    print(f"  Modelos:     {MODELS_DIR}")
    print(f"  Relatório:   {report_path}")
    print()

    # Sumário dos resultados
    best = eval_results["best_model"]
    print("  --- Resultados Principais ---")
    print(f"  Melhor modelo: {best['name']}")
    print(f"  ROC-AUC:       {best['metrics']['roc_auc']:.4f}")
    print(f"  F1-Score:      {best['metrics']['f1']:.4f}")

    ov = eda_results.get("overview", {})
    print(f"  Total pedidos: {ov.get('total_orders', 'N/A'):,}")
    print(f"  Faturamento:   R$ {ov.get('total_revenue', 0):,.2f}")

    ret = business_results.get("retention", {})
    print(f"  Retenção:      {ret.get('retention_rate', 0):.1%}")
    print()


if __name__ == "__main__":
    main()
