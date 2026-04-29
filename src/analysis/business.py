"""
Módulo de análise de negócio.

Implementa análises de retenção de clientes, segmentação RFM
e análise de satisfação para insights estratégicos.
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def analyze_customer_retention(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analisa retenção de clientes e padrões de recorrência.

    Returns:
        Dicionário com métricas de retenção e dados para visualização.
    """
    print("[business] Analisando retenção de clientes...")

    customer_orders = (
        df.group_by("customer_unique_id")
        .agg(
            pl.col("order_id").n_unique().alias("order_count"),
            pl.col("order_purchase_timestamp").min().alias("first_order"),
            pl.col("order_purchase_timestamp").max().alias("last_order"),
            pl.col("price").sum().alias("total_spent"),
        )
    )

    total = customer_orders.height
    recurring = customer_orders.filter(pl.col("order_count") > 1).height
    retention_rate = recurring / total if total > 0 else 0

    # Distribuição de pedidos por cliente
    order_dist = (
        customer_orders.group_by("order_count")
        .agg(pl.col("customer_unique_id").count().alias("num_customers"))
        .sort("order_count")
    )

    # Tempo entre pedidos (apenas recorrentes)
    recurrent = customer_orders.filter(pl.col("order_count") > 1)
    if recurrent.height > 0:
        recurrent = recurrent.with_columns(
            ((pl.col("last_order") - pl.col("first_order")).dt.total_days())
            .alias("days_between")
        )
        days_between_values = recurrent["days_between"].to_numpy()
    else:
        days_between_values = np.array([])

    print(f"  -> Clientes totais: {total:,}")
    print(f"  -> Clientes recorrentes: {recurring:,} ({retention_rate:.1%})")

    return {
        "total_customers": total,
        "recurring_customers": recurring,
        "retention_rate": retention_rate,
        "order_distribution": order_dist,
        "days_between_values": days_between_values,
        "order_count_values": customer_orders["order_count"].to_numpy(),
    }


def segment_customers_rfm(df: pl.DataFrame) -> dict[str, Any]:
    """
    Segmenta clientes usando análise RFM (Recency, Frequency, Monetary).

    Determina o número ótimo de clusters via silhouette score.

    Returns:
        Dicionário com dados de segmentação e métricas.
    """
    print("[business] Segmentando clientes (RFM)...")

    max_date = df["order_purchase_timestamp"].max()

    rfm = (
        df.group_by("customer_unique_id")
        .agg(
            ((pl.lit(max_date) - pl.col("order_purchase_timestamp").max())
             .dt.total_days().alias("recency")),
            pl.col("order_id").n_unique().alias("frequency"),
            pl.col("price").sum().alias("monetary"),
        )
    )

    # Remove outliers via IQR
    for col in ["recency", "frequency", "monetary"]:
        q1 = rfm[col].quantile(0.05)
        q3 = rfm[col].quantile(0.95)
        rfm = rfm.filter(
            (pl.col(col) >= q1) & (pl.col(col) <= q3)
        )

    # Normalização
    features_np = rfm.select(["recency", "frequency", "monetary"]).to_numpy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)

    # Determina K ótimo via silhouette score (K=2..8)
    silhouette_scores = []
    inertias = []
    k_range = range(2, 9)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels, sample_size=min(5000, len(features_scaled)))
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)

    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"  -> K ótimo determinado: {best_k} (silhouette={max(silhouette_scores):.3f})")

    # Treina modelo final
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    rfm = rfm.with_columns(
        pl.Series("cluster", kmeans_final.fit_predict(features_scaled))
    )

    # Sumário dos clusters
    cluster_summary = (
        rfm.group_by("cluster")
        .agg(
            pl.col("recency").mean().alias("avg_recency"),
            pl.col("frequency").mean().alias("avg_frequency"),
            pl.col("monetary").mean().alias("avg_monetary"),
            pl.col("customer_unique_id").count().alias("size"),
        )
        .sort("cluster")
    )

    print("  -> Sumário dos clusters:")
    for row in cluster_summary.iter_rows(named=True):
        print(
            f"     Cluster {row['cluster']}: {row['size']:,} clientes | "
            f"Recência={row['avg_recency']:.0f}d | "
            f"Freq={row['avg_frequency']:.1f} | "
            f"Valor=R${row['avg_monetary']:.2f}"
        )

    return {
        "rfm_data": rfm,
        "cluster_summary": cluster_summary,
        "best_k": best_k,
        "silhouette_scores": list(silhouette_scores),
        "inertias": list(inertias),
        "k_range": list(k_range),
        "best_silhouette": max(silhouette_scores),
    }


def analyze_satisfaction(df: pl.DataFrame) -> dict[str, Any]:
    """
    Análise aprofundada da satisfação do cliente.

    Returns:
        Dicionário com dados de satisfação.
    """
    print("[business] Analisando satisfação dos clientes...")

    valid = df.filter(
        pl.col("review_score").is_not_null() & pl.col("delivery_time").is_not_null()
    )

    # Satisfação por faixa de tempo de entrega
    valid = valid.with_columns(
        pl.when(pl.col("delivery_time") <= 7).then(pl.lit("0-7 dias"))
        .when(pl.col("delivery_time") <= 14).then(pl.lit("8-14 dias"))
        .when(pl.col("delivery_time") <= 21).then(pl.lit("15-21 dias"))
        .when(pl.col("delivery_time") <= 30).then(pl.lit("22-30 dias"))
        .otherwise(pl.lit("30+ dias"))
        .alias("delivery_bracket")
    )

    satisfaction_by_bracket = (
        valid.group_by("delivery_bracket")
        .agg(
            pl.col("review_score").mean().alias("avg_score"),
            pl.col("review_score").median().alias("median_score"),
            pl.col("order_id").n_unique().alias("count"),
        )
        .sort("delivery_bracket")
    )

    # Satisfação por atraso vs pontual
    satisfaction_by_late = (
        valid.group_by("is_late")
        .agg(
            pl.col("review_score").mean().alias("avg_score"),
            pl.col("review_score").median().alias("median_score"),
            pl.col("order_id").n_unique().alias("count"),
        )
        .sort("is_late")
    )

    avg_on_time = satisfaction_by_late.filter(pl.col("is_late") == 0)["avg_score"]
    avg_late = satisfaction_by_late.filter(pl.col("is_late") == 1)["avg_score"]

    print(f"  -> Nota média (pontual): {avg_on_time[0]:.2f}" if len(avg_on_time) > 0 else "")
    print(f"  -> Nota média (atrasado): {avg_late[0]:.2f}" if len(avg_late) > 0 else "")

    return {
        "by_bracket": satisfaction_by_bracket,
        "by_late": satisfaction_by_late,
        "review_scores": valid["review_score"].to_numpy(),
        "delivery_times": valid["delivery_time"].to_numpy(),
    }


def run_business_analysis(df: pl.DataFrame) -> dict[str, Any]:
    """
    Executa todas as análises de negócio.

    Args:
        df: DataFrame Polars com dados de EDA.

    Returns:
        Dicionário consolidado com resultados.
    """
    print("[business] Iniciando análises de negócio...")

    results = {
        "retention": analyze_customer_retention(df),
        "segmentation": segment_customers_rfm(df),
        "satisfaction": analyze_satisfaction(df),
    }

    print("[business] Análises de negócio concluídas.")
    return results
