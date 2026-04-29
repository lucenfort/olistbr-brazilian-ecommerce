"""
Módulo de geração de relatórios.

Gera relatórios em Markdown com resultados consolidados
das análises e modelagem.
"""

import os
from typing import Any

from src.config import REPORTS_DIR, CHARTS_DIR, MODELS_DIR


def generate_report(
    eda_results: dict,
    business_results: dict,
    eval_results: dict | None = None,
) -> str:
    """
    Gera relatório técnico em Markdown.

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "relatorio_analise.md")

    lines = []
    lines.append("# Relatório de Análise — E-commerce Brasileiro Olist\n")
    lines.append("---\n")

    # Visão geral
    ov = eda_results.get("overview", {})
    lines.append("## 1. Visão Geral do Dataset\n")
    lines.append(f"- **Total de pedidos**: {ov.get('total_orders', 'N/A'):,}")
    lines.append(f"- **Clientes únicos**: {ov.get('total_customers', 'N/A'):,}")
    lines.append(f"- **Vendedores**: {ov.get('total_sellers', 'N/A'):,}")
    lines.append(f"- **Faturamento total**: R$ {ov.get('total_revenue', 0):,.2f}")
    lines.append(f"- **Ticket médio**: R$ {ov.get('avg_ticket', 0):,.2f}")
    lines.append(f"- **Período**: {ov.get('date_range', 'N/A')}")
    lines.append("")

    # EDA
    lines.append("## 2. Análise Exploratória de Dados\n")
    monthly = eda_results.get("monthly", {})
    lines.append("### 2.1 Evolução de Pedidos\n")
    lines.append(f"- Média mensal: **{monthly.get('avg_monthly_orders', 0):,.0f}** pedidos")
    lines.append(f"- Pico: **{monthly.get('max_orders', 0):,}** pedidos em {monthly.get('max_month', 'N/A')}")
    lines.append(f"- Crescimento médio mensal: **{monthly.get('avg_growth_rate', 0):.1f}%**")
    lines.append(f"\n![Evolução Mensal](../graficos/01_volume_pedidos_mensal.png)\n")

    delivery = eda_results.get("delivery", {})
    lines.append("### 2.2 Tempo de Entrega\n")
    lines.append(f"- Média: **{delivery.get('mean', 0):.1f}** dias")
    lines.append(f"- Mediana: **{delivery.get('median', 0):.1f}** dias")
    lines.append(f"- Desvio padrão: **{delivery.get('std', 0):.1f}** dias")
    lines.append(f"\n![Distribuição Entrega](../graficos/02_distribuicao_tempo_entrega.png)\n")
    lines.append(f"\n![Entrega por Região](../graficos/03_entrega_por_regiao.png)\n")

    freight = eda_results.get("freight", {})
    lines.append("### 2.3 Análise de Frete\n")
    lines.append(f"- Frete médio (mesmo estado): **R$ {freight.get('same_state_mean', 0):.2f}**")
    lines.append(f"- Frete médio (inter-estado): **R$ {freight.get('diff_state_mean', 0):.2f}**")
    lines.append(f"\n![Frete por Estado](../graficos/04_frete_estado.png)\n")

    lines.append("### 2.4 Categorias e Estados\n")
    lines.append(f"\n![Top Categorias](../graficos/05_top_categorias.png)\n")
    lines.append(f"\n![Top Estados](../graficos/06_top_estados.png)\n")
    lines.append(f"\n![Correlação](../graficos/07_correlacao.png)\n")

    lines.append("### 2.5 Avaliações dos Clientes\n")
    reviews = eda_results.get("reviews", {})
    lines.append(f"- Nota média: **{reviews.get('avg_score', 0):.2f}** / 5.0")
    lines.append(f"\n![Distribuição Avaliações](../graficos/08_distribuicao_avaliacoes.png)\n")

    # Negócio
    lines.append("## 3. Análise de Negócio\n")

    ret = business_results.get("retention", {})
    lines.append("### 3.1 Retenção de Clientes\n")
    lines.append(f"- Clientes totais: **{ret.get('total_customers', 0):,}**")
    lines.append(f"- Clientes recorrentes: **{ret.get('recurring_customers', 0):,}**")
    lines.append(f"- Taxa de retenção: **{ret.get('retention_rate', 0):.1%}**")
    lines.append(f"\n![Retenção](../graficos/09_retencao_clientes.png)\n")

    seg = business_results.get("segmentation", {})
    lines.append("### 3.2 Segmentação de Clientes (RFM)\n")
    lines.append(f"- Número ótimo de clusters: **{seg.get('best_k', 'N/A')}**")
    lines.append(f"- Silhouette score: **{seg.get('best_silhouette', 0):.3f}**\n")

    summary = seg.get("cluster_summary")
    if summary is not None and hasattr(summary, "iter_rows"):
        lines.append("| Cluster | Tamanho | Recência (dias) | Frequência | Valor (R$) |")
        lines.append("|---------|---------|-----------------|------------|------------|")
        for row in summary.iter_rows(named=True):
            lines.append(
                f"| {row['cluster']} | {row['size']:,} | "
                f"{row['avg_recency']:.0f} | {row['avg_frequency']:.1f} | "
                f"{row['avg_monetary']:,.2f} |"
            )
        lines.append("")

    lines.append(f"\n![Segmentação RFM](../graficos/10_segmentacao_rfm.png)\n")
    lines.append(f"\n![Cotovelo e Silhouette](../graficos/11_elbow_silhouette.png)\n")

    lines.append("### 3.3 Satisfação vs Entrega\n")
    lines.append(f"\n![Satisfação vs Entrega](../graficos/12_satisfacao_entrega.png)\n")

    # Modelos
    if eval_results:
        lines.append("## 4. Modelagem Preditiva\n")
        lines.append("### Objetivo\n")
        lines.append("Prever se um pedido será entregue com **atraso** em relação à data estimada.\n")
        lines.append("### 4.1 Comparação de Modelos\n")

        table = eval_results.get("comparison_table", [])
        if table:
            lines.append("| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV-AUC |")
            lines.append("|--------|----------|-----------|--------|----|---------|--------|--------|")
            for row in table:
                lines.append(
                    f"| {row['Modelo']} | {row['Accuracy']:.4f} | "
                    f"{row['Precision']:.4f} | {row['Recall']:.4f} | "
                    f"{row['F1-Score']:.4f} | {row['ROC-AUC']:.4f} | "
                    f"{row['PR-AUC']:.4f} | {row['CV-AUC']:.4f} |"
                )
            lines.append("")

        best = eval_results.get("best_model", {})
        lines.append(f"**Melhor modelo**: {best.get('name', 'N/A')} "
                      f"(ROC-AUC = {best.get('metrics', {}).get('roc_auc', 0):.4f})\n")

        lines.append(f"\n![Curvas ROC](../modelos/13_curvas_roc.png)\n")
        lines.append(f"\n![Curvas PR](../modelos/14_curvas_precision_recall.png)\n")
        lines.append(f"\n![Matrizes de Confusão](../modelos/15_matrizes_confusao.png)\n")
        lines.append(f"\n![Feature Importance](../modelos/16_feature_importance.png)\n")
        lines.append(f"\n![Comparação Radar](../modelos/17_comparacao_modelos.png)\n")
        lines.append(f"\n![Learning Curves](../modelos/18_learning_curves.png)\n")

        # Hiperparâmetros do melhor modelo
        best_name = best.get("name", "")
        if best_name and best_name in eval_results.get("model_results", {}):
            params = eval_results["model_results"][best_name].get("best_params", {})
            if params:
                lines.append(f"### 4.2 Hiperparâmetros Otimizados ({best_name})\n")
                lines.append("| Parâmetro | Valor |")
                lines.append("|-----------|-------|")
                for k, v in params.items():
                    if isinstance(v, float):
                        lines.append(f"| {k} | {v:.6f} |")
                    else:
                        lines.append(f"| {k} | {v} |")
                lines.append("")

    lines.append("---\n")
    lines.append("*Relatório gerado automaticamente pelo pipeline de análise.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[report] Relatório salvo em: {report_path}")
    return report_path
