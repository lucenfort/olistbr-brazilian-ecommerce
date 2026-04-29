"""
Módulo de gráficos profissionais.

Gera todas as visualizações estáticas e interativas do projeto
utilizando Plotly com tema visual consistente e profissional.
"""

import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import COLORS, COLOR_SEQUENCE, CHARTS_DIR, MODELS_DIR


def _get_layout(**kwargs) -> dict:
    """Retorna layout base profissional para gráficos Plotly."""
    base = {
        "paper_bgcolor": COLORS["bg_dark"],
        "plot_bgcolor": COLORS["bg_dark"],
        "font": {"color": COLORS["text_white"], "family": "Inter, Arial, sans-serif", "size": 13},
        "title": {"font": {"size": 18, "color": COLORS["text_white"]}, "x": 0.5, "xanchor": "center"},
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"], "showgrid": True},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"], "showgrid": True},
        "margin": {"t": 80, "b": 60, "l": 70, "r": 30},
    }
    base.update(kwargs)
    return base


def _save(fig: go.Figure, filename: str, directory: str | None = None) -> None:
    """Salva figura em PNG e HTML."""
    d = directory or CHARTS_DIR
    os.makedirs(d, exist_ok=True)
    base = os.path.splitext(filename)[0]
    try:
        fig.write_image(os.path.join(d, f"{base}.png"), width=1200, height=700, scale=2)
    except Exception as e:
        print(f"  [charts] Aviso PNG ({base}): {e}")
    fig.write_html(os.path.join(d, f"{base}.html"), include_plotlyjs="cdn")
    print(f"  [charts] Salvo: {base}")


# ---------------------------------------------------------------------------
# Gráficos de EDA
# ---------------------------------------------------------------------------

def plot_monthly_orders(eda: dict) -> None:
    """Gráfico de evolução mensal de pedidos."""
    data = eda["monthly"]["monthly_data"]
    months = data["order_month"].to_list()
    counts = data["order_count"].to_list()
    revenue = data["revenue"].to_list()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=months, y=revenue, name="Faturamento (R$)",
               marker_color=COLORS["primary"], opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=months, y=counts, name="Pedidos", mode="lines+markers",
                   line={"color": COLORS["accent"], "width": 3},
                   marker={"size": 7}),
        secondary_y=True,
    )
    fig.update_layout(
        **_get_layout(title={"text": "Evolução Mensal de Pedidos e Faturamento"}),
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01,
                "bgcolor": "rgba(0,0,0,0.5)"},
    )
    fig.update_yaxes(title_text="Faturamento (R$)", secondary_y=False,
                     gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Número de Pedidos", secondary_y=True,
                     gridcolor=COLORS["grid"])
    fig.update_xaxes(title_text="Mês", tickangle=-45)
    _save(fig, "01_volume_pedidos_mensal")


def plot_delivery_distribution(eda: dict) -> None:
    """Histograma de distribuição do tempo de entrega."""
    values = eda["delivery"]["delivery_values"]
    # Limita para visualização (remove extremos)
    values = values[values <= np.percentile(values, 99)]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=50,
        marker_color=COLORS["primary"],
        opacity=0.8,
        name="Frequência",
    ))
    # Adiciona linha de mediana
    median = np.median(values)
    fig.add_vline(x=median, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text=f"Mediana: {median:.0f} dias",
                  annotation_font_color=COLORS["text_white"])

    fig.update_layout(**_get_layout(
        title={"text": "Distribuição do Tempo de Entrega"},
        xaxis_title="Dias até a Entrega",
        yaxis_title="Frequência",
    ))
    _save(fig, "02_distribuicao_tempo_entrega")


def plot_delivery_by_region(eda: dict) -> None:
    """Tempo de entrega por região."""
    data = eda["delivery"]["by_region"]
    regions = data["region"].to_list()
    avg = data["avg_delivery"].to_list()
    median = data["median_delivery"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=regions, x=avg, orientation="h",
        name="Média", marker_color=COLORS["primary"],
    ))
    fig.add_trace(go.Bar(
        y=regions, x=median, orientation="h",
        name="Mediana", marker_color=COLORS["secondary"],
    ))
    fig.update_layout(**_get_layout(
        title={"text": "Tempo de Entrega por Região"},
        xaxis_title="Dias",
        yaxis_title="",
        barmode="group",
        legend={"bgcolor": "rgba(0,0,0,0.5)"},
    ))
    _save(fig, "03_entrega_por_regiao")


def plot_freight_analysis(eda: dict) -> None:
    """Análise de frete: mesmo estado vs inter-estado."""
    same = eda["freight"]["same_state_values"]
    diff = eda["freight"]["diff_state_values"]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=same[same <= np.percentile(same, 95)],
        name="Mesmo Estado",
        marker_color=COLORS["secondary"],
        boxmean=True,
    ))
    fig.add_trace(go.Box(
        y=diff[diff <= np.percentile(diff, 95)],
        name="Inter-Estado",
        marker_color=COLORS["accent"],
        boxmean=True,
    ))
    fig.update_layout(**_get_layout(
        title={"text": "Valor do Frete: Mesmo Estado vs Inter-Estado"},
        yaxis_title="Valor do Frete (R$)",
    ))
    _save(fig, "04_frete_estado")


def plot_top_categories(eda: dict) -> None:
    """Top 10 categorias por faturamento."""
    data = eda["categories"]["top10"]
    cats = data["product_category_name"].to_list()[::-1]
    revenue = data["revenue"].to_list()[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats, x=revenue, orientation="h",
        marker_color=COLORS["primary"],
        text=[f"R$ {v:,.0f}" for v in revenue],
        textposition="outside",
        textfont={"color": COLORS["text_white"], "size": 11},
    ))
    fig.update_layout(**_get_layout(
        title={"text": "Top 10 Categorias por Faturamento"},
        xaxis_title="Faturamento (R$)",
        margin={"l": 200},
    ))
    _save(fig, "05_top_categorias")


def plot_top_states(eda: dict) -> None:
    """Top 10 estados por valor médio de pedido."""
    data = eda["states"]["top10"]
    states = data["customer_state"].to_list()[::-1]
    avg_values = data["avg_order_value"].to_list()[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=states, x=avg_values, orientation="h",
        marker_color=COLORS["secondary"],
        text=[f"R$ {v:,.2f}" for v in avg_values],
        textposition="outside",
        textfont={"color": COLORS["text_white"], "size": 11},
    ))
    fig.update_layout(**_get_layout(
        title={"text": "Top 10 Estados por Valor Médio de Pedido"},
        xaxis_title="Valor Médio (R$)",
    ))
    _save(fig, "06_top_estados")


def plot_correlation_heatmap(eda: dict) -> None:
    """Heatmap de correlação entre variáveis numéricas."""
    cols = eda["correlation"]["columns"]
    matrix = eda["correlation"]["matrix"]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=cols, y=cols,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar={"title": "Correlação", "tickfont": {"color": COLORS["text_white"]}},
    ))
    fig.update_layout(**_get_layout(
        title={"text": "Matriz de Correlação"},
        width=800, height=700,
    ))
    _save(fig, "07_correlacao")


def plot_review_distribution(eda: dict) -> None:
    """Distribuição de avaliações."""
    data = eda["reviews"]["distribution"]
    scores = data["review_score"].to_list()
    counts = data["count"].to_list()

    colors_bars = [COLORS["error"] if s <= 2 else COLORS["warning"] if s == 3
                   else COLORS["success"] for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(s) for s in scores], y=counts,
        marker_color=colors_bars,
        text=[f"{c:,}" for c in counts],
        textposition="outside",
        textfont={"color": COLORS["text_white"]},
    ))
    fig.update_layout(**_get_layout(
        title={"text": "Distribuição das Avaliações dos Clientes"},
        xaxis_title="Nota de Avaliação",
        yaxis_title="Quantidade",
    ))
    _save(fig, "08_distribuicao_avaliacoes")


# ---------------------------------------------------------------------------
# Gráficos de Negócio
# ---------------------------------------------------------------------------

def plot_customer_retention(biz: dict) -> None:
    """Gráfico de distribuição de pedidos por cliente."""
    values = biz["retention"]["order_count_values"]
    values = values[values <= 10]  # Limita para visualização

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=10,
        marker_color=COLORS["primary"],
        opacity=0.8,
    ))

    total = biz["retention"]["total_customers"]
    recurring = biz["retention"]["recurring_customers"]
    rate = biz["retention"]["retention_rate"]

    fig.add_annotation(
        x=0.95, y=0.95, xref="paper", yref="paper",
        text=(f"Total: {total:,}<br>"
              f"Recorrentes: {recurring:,}<br>"
              f"Taxa: {rate:.1%}"),
        showarrow=False,
        font={"size": 13, "color": COLORS["text_white"]},
        bgcolor="rgba(0,0,0,0.6)", borderpad=10,
    )

    fig.update_layout(**_get_layout(
        title={"text": "Distribuição de Pedidos por Cliente"},
        xaxis_title="Número de Pedidos",
        yaxis_title="Quantidade de Clientes",
    ))
    _save(fig, "09_retencao_clientes")


def plot_rfm_segmentation(biz: dict) -> None:
    """Scatter plot 3D da segmentação RFM."""
    rfm = biz["segmentation"]["rfm_data"]
    summary = biz["segmentation"]["cluster_summary"]

    fig = go.Figure()
    for cluster in sorted(rfm["cluster"].unique().to_list()):
        subset = rfm.filter(rfm["cluster"] == cluster)
        size_info = summary.filter(summary["cluster"] == cluster)["size"][0]
        fig.add_trace(go.Scatter(
            x=subset["recency"].to_list(),
            y=subset["monetary"].to_list(),
            mode="markers",
            name=f"Cluster {cluster} (n={size_info:,})",
            marker={
                "size": 4,
                "opacity": 0.5,
                "color": COLOR_SEQUENCE[cluster % len(COLOR_SEQUENCE)],
            },
        ))

    fig.update_layout(**_get_layout(
        title={"text": "Segmentação de Clientes (RFM)"},
        xaxis_title="Recência (dias)",
        yaxis_title="Valor Monetário (R$)",
        legend={"bgcolor": "rgba(0,0,0,0.5)"},
    ))
    _save(fig, "10_segmentacao_rfm")


def plot_elbow_silhouette(biz: dict) -> None:
    """Método do cotovelo e silhouette score para clusters."""
    seg = biz["segmentation"]
    k_range = seg["k_range"]
    inertias = seg["inertias"]
    silhouettes = seg["silhouette_scores"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Método do Cotovelo", "Silhouette Score"))

    fig.add_trace(
        go.Scatter(x=k_range, y=inertias, mode="lines+markers",
                   line={"color": COLORS["primary"], "width": 3},
                   marker={"size": 8}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=k_range, y=silhouettes, mode="lines+markers",
                   line={"color": COLORS["accent"], "width": 3},
                   marker={"size": 8}),
        row=1, col=2,
    )

    # Marca K ótimo
    best_k = seg["best_k"]
    best_idx = k_range.index(best_k)
    fig.add_vline(x=best_k, line_dash="dash", line_color=COLORS["success"],
                  row=1, col=2)
    fig.add_annotation(x=best_k, y=silhouettes[best_idx],
                       text=f"K={best_k}", showarrow=True,
                       font={"color": COLORS["text_white"]},
                       row=1, col=2)

    fig.update_layout(**_get_layout(
        title={"text": "Seleção do Número Ótimo de Clusters"},
    ))
    fig.update_xaxes(title_text="K (clusters)", gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    _save(fig, "11_elbow_silhouette")


def plot_satisfaction_vs_delivery(biz: dict) -> None:
    """Satisfação vs tempo de entrega."""
    data = biz["satisfaction"]["by_bracket"]
    brackets = data["delivery_bracket"].to_list()
    scores = data["avg_score"].to_list()
    counts = data["count"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=brackets, y=scores,
        marker_color=[COLORS["success"] if s >= 4 else COLORS["warning"] if s >= 3
                      else COLORS["error"] for s in scores],
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
        textfont={"color": COLORS["text_white"]},
    ))
    fig.update_layout(**_get_layout(
        title={"text": "Nota Média de Satisfação por Faixa de Entrega"},
        xaxis_title="Tempo de Entrega",
        yaxis_title="Nota Média",
        yaxis={"range": [0, 5.5], "gridcolor": COLORS["grid"]},
    ))
    _save(fig, "12_satisfacao_entrega")


# ---------------------------------------------------------------------------
# Gráficos de Modelos
# ---------------------------------------------------------------------------

def plot_roc_curves(eval_results: dict) -> None:
    """Curvas ROC de todos os modelos sobrepostas."""
    fig = go.Figure()

    for i, (name, data) in enumerate(eval_results["model_results"].items()):
        roc = data["roc_curve"]
        auc = data["metrics"]["roc_auc"]
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            mode="lines",
            name=f"{name} (AUC={auc:.4f})",
            line={"color": COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], "width": 2.5},
        ))

    # Linha diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line={"color": "gray", "dash": "dash", "width": 1},
        name="Random",
        showlegend=False,
    ))

    fig.update_layout(**_get_layout(
        title={"text": "Curvas ROC — Comparação de Modelos"},
        xaxis_title="Taxa de Falsos Positivos (FPR)",
        yaxis_title="Taxa de Verdadeiros Positivos (TPR)",
        legend={"bgcolor": "rgba(0,0,0,0.6)", "font": {"size": 12}},
    ))
    _save(fig, "13_curvas_roc", MODELS_DIR)


def plot_pr_curves(eval_results: dict) -> None:
    """Curvas Precision-Recall de todos os modelos."""
    fig = go.Figure()

    for i, (name, data) in enumerate(eval_results["model_results"].items()):
        pr = data["pr_curve"]
        auc = data["metrics"]["pr_auc"]
        fig.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"],
            mode="lines",
            name=f"{name} (PR-AUC={auc:.4f})",
            line={"color": COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], "width": 2.5},
        ))

    fig.update_layout(**_get_layout(
        title={"text": "Curvas Precision-Recall — Comparação de Modelos"},
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend={"bgcolor": "rgba(0,0,0,0.6)", "font": {"size": 12}},
    ))
    _save(fig, "14_curvas_precision_recall", MODELS_DIR)


def plot_confusion_matrices(eval_results: dict) -> None:
    """Matrizes de confusão em grid 2x2."""
    model_names = list(eval_results["model_results"].keys())
    n = len(model_names)
    rows = (n + 1) // 2
    cols = 2

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=model_names,
        horizontal_spacing=0.15,
        vertical_spacing=0.12,
    )

    for i, name in enumerate(model_names):
        cm = eval_results["model_results"][name]["confusion_matrix"]
        r = i // 2 + 1
        c = i % 2 + 1

        # Normaliza para percentual
        cm_pct = cm / cm.sum() * 100

        fig.add_trace(go.Heatmap(
            z=cm_pct, x=["Pred: Pontual", "Pred: Atrasado"],
            y=["Real: Pontual", "Real: Atrasado"],
            colorscale=[[0, COLORS["bg_dark"]], [1, COLORS["primary"]]],
            text=[[f"{cm[j][k]:,}\n({cm_pct[j][k]:.1f}%)" for k in range(2)] for j in range(2)],
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=False,
        ), row=r, col=c)

    fig.update_layout(**_get_layout(
        title={"text": "Matrizes de Confusão"},
        height=500 * rows,
    ))
    _save(fig, "15_matrizes_confusao", MODELS_DIR)


def plot_feature_importance(eval_results: dict) -> None:
    """Feature importance comparativa dos modelos."""
    # Usa o melhor modelo
    best_name = eval_results["best_model"]["name"]
    fi = eval_results["model_results"][best_name]["feature_importance"]

    if fi is None:
        print("  [charts] Feature importance não disponível.")
        return

    # Top 15 features
    top_features = dict(list(fi.items())[:15])
    names = list(top_features.keys())[::-1]
    values = list(top_features.values())[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=COLORS["accent"],
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont={"color": COLORS["text_white"], "size": 10},
    ))
    fig.update_layout(**_get_layout(
        title={"text": f"Top 15 Features Mais Importantes ({best_name})"},
        xaxis_title="Importância",
        margin={"l": 200},
    ))
    _save(fig, "16_feature_importance", MODELS_DIR)


def plot_model_comparison(eval_results: dict) -> None:
    """Radar chart de comparação de modelos."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]

    fig = go.Figure()
    for i, row in enumerate(eval_results["comparison_table"]):
        values = [row[m] for m in metrics_names]
        values.append(values[0])  # Fecha o polígono
        names_closed = metrics_names + [metrics_names[0]]

        fig.add_trace(go.Scatterpolar(
            r=values, theta=names_closed,
            name=row["Modelo"],
            line={"color": COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], "width": 2},
            fill="toself",
            opacity=0.3,
        ))

    fig.update_layout(
        polar={
            "bgcolor": COLORS["bg_dark"],
            "radialaxis": {
                "visible": True, "range": [0, 1],
                "gridcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text_white"]},
            },
            "angularaxis": {
                "gridcolor": COLORS["grid"],
                "tickfont": {"color": COLORS["text_white"], "size": 12},
            },
        },
        paper_bgcolor=COLORS["bg_dark"],
        font={"color": COLORS["text_white"], "family": "Inter, Arial, sans-serif"},
        title={"text": "Comparação de Modelos", "font": {"size": 18}, "x": 0.5},
        legend={"bgcolor": "rgba(0,0,0,0.5)"},
        margin={"t": 80, "b": 40},
    )
    _save(fig, "17_comparacao_modelos", MODELS_DIR)


def plot_learning_curves(eval_results: dict) -> None:
    """Learning curves do melhor modelo."""
    lc = eval_results.get("learning_curves", {})
    if not lc:
        return

    fig = go.Figure()

    # Treino
    fig.add_trace(go.Scatter(
        x=lc["train_sizes"], y=lc["train_mean"],
        mode="lines", name="Treino",
        line={"color": COLORS["primary"], "width": 2.5},
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([lc["train_sizes"], lc["train_sizes"][::-1]]),
        y=np.concatenate([lc["train_mean"] + lc["train_std"],
                          (lc["train_mean"] - lc["train_std"])[::-1]]),
        fill="toself", fillcolor="rgba(13,71,161,0.2)",
        line={"color": "rgba(0,0,0,0)"}, showlegend=False,
    ))

    # Validação
    fig.add_trace(go.Scatter(
        x=lc["train_sizes"], y=lc["test_mean"],
        mode="lines", name="Validação",
        line={"color": COLORS["accent"], "width": 2.5},
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([lc["train_sizes"], lc["train_sizes"][::-1]]),
        y=np.concatenate([lc["test_mean"] + lc["test_std"],
                          (lc["test_mean"] - lc["test_std"])[::-1]]),
        fill="toself", fillcolor="rgba(255,111,0,0.2)",
        line={"color": "rgba(0,0,0,0)"}, showlegend=False,
    ))

    best_name = eval_results["best_model"]["name"]
    fig.update_layout(**_get_layout(
        title={"text": f"Learning Curves — {best_name}"},
        xaxis_title="Amostras de Treino",
        yaxis_title="ROC-AUC",
        legend={"bgcolor": "rgba(0,0,0,0.5)"},
    ))
    _save(fig, "18_learning_curves", MODELS_DIR)


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def generate_all_charts(
    eda_results: dict,
    business_results: dict,
    eval_results: dict | None = None,
) -> None:
    """
    Gera todos os gráficos do projeto.

    Args:
        eda_results: Resultados da análise exploratória.
        business_results: Resultados da análise de negócio.
        eval_results: Resultados da avaliação de modelos.
    """
    print("\n[charts] Gerando gráficos profissionais...")
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # EDA
    plot_monthly_orders(eda_results)
    plot_delivery_distribution(eda_results)
    plot_delivery_by_region(eda_results)
    plot_freight_analysis(eda_results)
    plot_top_categories(eda_results)
    plot_top_states(eda_results)
    plot_correlation_heatmap(eda_results)
    plot_review_distribution(eda_results)

    # Negócio
    plot_customer_retention(business_results)
    plot_rfm_segmentation(business_results)
    plot_elbow_silhouette(business_results)
    plot_satisfaction_vs_delivery(business_results)

    # Modelos
    if eval_results:
        plot_roc_curves(eval_results)
        plot_pr_curves(eval_results)
        plot_confusion_matrices(eval_results)
        plot_feature_importance(eval_results)
        plot_model_comparison(eval_results)
        plot_learning_curves(eval_results)

    print("[charts] Todos os gráficos gerados com sucesso.\n")
