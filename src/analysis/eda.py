"""
Módulo de Análise Exploratória de Dados (EDA).

Implementa análises descritivas e estatísticas sobre o dataset do e-commerce
Olist, gerando métricas e dados para visualizações.
"""

from typing import Any

import polars as pl
import numpy as np

from src.config import STATE_REGION_MAP


def analyze_monthly_orders(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analisa a evolução do volume de pedidos por mês.

    Returns:
        Dicionário com dados mensais e métricas de tendência.
    """
    monthly = (
        df.group_by("order_month")
        .agg(
            pl.col("order_id").n_unique().alias("order_count"),
            pl.col("price").sum().alias("revenue"),
        )
        .sort("order_month")
    )

    growth_rates = []
    counts = monthly["order_count"].to_list()
    for i in range(1, len(counts)):
        if counts[i - 1] > 0:
            growth_rates.append((counts[i] - counts[i - 1]) / counts[i - 1] * 100)

    return {
        "monthly_data": monthly,
        "avg_monthly_orders": monthly["order_count"].mean(),
        "max_month": monthly.filter(
            pl.col("order_count") == pl.col("order_count").max()
        )["order_month"][0],
        "max_orders": monthly["order_count"].max(),
        "avg_growth_rate": np.mean(growth_rates) if growth_rates else 0,
        "total_revenue": monthly["revenue"].sum(),
    }


def analyze_delivery_time(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analisa a distribuição do tempo de entrega.

    Returns:
        Dicionário com estatísticas de entrega.
    """
    delivery = df.filter(pl.col("delivery_time").is_not_null())

    stats = {
        "mean": delivery["delivery_time"].mean(),
        "median": delivery["delivery_time"].median(),
        "std": delivery["delivery_time"].std(),
        "min": delivery["delivery_time"].min(),
        "max": delivery["delivery_time"].max(),
        "q25": delivery["delivery_time"].quantile(0.25),
        "q75": delivery["delivery_time"].quantile(0.75),
        "delivery_values": delivery["delivery_time"].to_numpy(),
    }

    # Entrega por região
    delivery_by_region = (
        delivery.group_by("region")
        .agg(
            pl.col("delivery_time").mean().alias("avg_delivery"),
            pl.col("delivery_time").median().alias("median_delivery"),
            pl.col("order_id").n_unique().alias("order_count"),
        )
        .sort("avg_delivery", descending=True)
    )

    stats["by_region"] = delivery_by_region
    return stats


def analyze_freight(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analisa a relação entre frete, distância e valor do pedido.

    Returns:
        Dicionário com dados de frete.
    """
    # Frete médio por estado
    freight_state = (
        df.group_by("customer_state")
        .agg(
            pl.col("freight_value").mean().alias("avg_freight"),
            pl.col("freight_value").median().alias("median_freight"),
            pl.col("price").mean().alias("avg_price"),
        )
        .sort("avg_freight", descending=True)
    )

    # Frete: mesmo estado vs inter-estado
    same_state_freight = df.filter(pl.col("customer_state") == pl.col("seller_state"))["freight_value"]
    diff_state_freight = df.filter(pl.col("customer_state") != pl.col("seller_state"))["freight_value"]

    return {
        "by_state": freight_state,
        "same_state_mean": same_state_freight.mean(),
        "diff_state_mean": diff_state_freight.mean(),
        "same_state_values": same_state_freight.to_numpy(),
        "diff_state_values": diff_state_freight.to_numpy(),
        "price_values": df["price"].to_numpy(),
        "freight_values": df["freight_value"].to_numpy(),
    }


def analyze_top_categories(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analisa as categorias de produtos com maior faturamento.

    Returns:
        Dicionário com ranking de categorias.
    """
    cat_sales = (
        df.filter(pl.col("product_category_name").is_not_null())
        .group_by("product_category_name")
        .agg(
            pl.col("price").sum().alias("revenue"),
            pl.col("order_id").n_unique().alias("order_count"),
            pl.col("price").mean().alias("avg_ticket"),
        )
        .sort("revenue", descending=True)
    )

    return {
        "top10": cat_sales.head(10),
        "total_categories": cat_sales.height,
        "top_category": cat_sales["product_category_name"][0] if cat_sales.height > 0 else "N/A",
        "top_revenue": cat_sales["revenue"][0] if cat_sales.height > 0 else 0,
    }


def analyze_state_orders(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analisa estados com maior valor médio de pedido.

    Returns:
        Dicionário com dados por estado.
    """
    # Valor total por pedido (agrupando itens)
    order_values = (
        df.group_by(["order_id", "customer_state"])
        .agg(pl.col("price").sum().alias("order_value"))
    )

    state_avg = (
        order_values.group_by("customer_state")
        .agg(
            pl.col("order_value").mean().alias("avg_order_value"),
            pl.col("order_value").median().alias("median_order_value"),
            pl.col("order_id").count().alias("order_count"),
        )
        .sort("avg_order_value", descending=True)
    )

    # Adiciona região
    state_avg = state_avg.with_columns(
        pl.col("customer_state")
        .replace_strict(STATE_REGION_MAP, default="Desconhecido")
        .alias("region")
    )

    return {
        "state_data": state_avg,
        "top10": state_avg.head(10),
    }


def analyze_correlation(df: pl.DataFrame) -> dict[str, Any]:
    """
    Calcula a matriz de correlação entre variáveis numéricas.

    Returns:
        Dicionário com matriz de correlação.
    """
    numeric_cols = [
        "price", "freight_value", "delivery_time", "review_score",
        "product_weight_g", "payment_installments",
    ]
    available = [c for c in numeric_cols if c in df.columns]

    # Converte para numpy para correlação
    data = df.select(available).drop_nulls().to_numpy()
    corr_matrix = np.corrcoef(data, rowvar=False)

    return {
        "columns": available,
        "matrix": corr_matrix,
    }


def analyze_reviews(df: pl.DataFrame) -> dict[str, Any]:
    """
    Analisa a distribuição de avaliações dos clientes.

    Returns:
        Dicionário com dados de avaliação.
    """
    reviews = df.filter(pl.col("review_score").is_not_null())

    review_dist = (
        reviews.group_by("review_score")
        .agg(pl.col("order_id").n_unique().alias("count"))
        .sort("review_score")
    )

    # Satisfação vs tempo de entrega
    satisfaction_delivery = (
        reviews.filter(pl.col("delivery_time").is_not_null())
        .group_by("review_score")
        .agg(
            pl.col("delivery_time").mean().alias("avg_delivery"),
            pl.col("delivery_time").median().alias("median_delivery"),
            pl.col("price").mean().alias("avg_price"),
            pl.col("order_id").n_unique().alias("count"),
        )
        .sort("review_score")
    )

    return {
        "distribution": review_dist,
        "satisfaction_delivery": satisfaction_delivery,
        "avg_score": reviews["review_score"].mean(),
        "score_values": reviews["review_score"].to_numpy(),
        "delivery_values": reviews.filter(
            pl.col("delivery_time").is_not_null()
        )["delivery_time"].to_numpy(),
        "review_delivery_pairs": reviews.filter(
            pl.col("delivery_time").is_not_null()
        ).select(["review_score", "delivery_time"]),
    }


def run_full_eda(df: pl.DataFrame) -> dict[str, Any]:
    """
    Executa todas as análises exploratórias.

    Args:
        df: DataFrame Polars com dados de EDA.

    Returns:
        Dicionário consolidado com todos os resultados.
    """
    print("[eda] Iniciando análise exploratória de dados...")

    results = {
        "monthly": analyze_monthly_orders(df),
        "delivery": analyze_delivery_time(df),
        "freight": analyze_freight(df),
        "categories": analyze_top_categories(df),
        "states": analyze_state_orders(df),
        "correlation": analyze_correlation(df),
        "reviews": analyze_reviews(df),
    }

    # Métricas gerais
    results["overview"] = {
        "total_orders": df["order_id"].n_unique(),
        "total_customers": df["customer_unique_id"].n_unique(),
        "total_sellers": df["seller_id"].n_unique(),
        "total_revenue": df["price"].sum(),
        "avg_ticket": df["price"].mean(),
        "date_range": f"{df['order_month'].min()} a {df['order_month'].max()}",
    }

    print("[eda] Análise exploratória concluída.")
    return results
