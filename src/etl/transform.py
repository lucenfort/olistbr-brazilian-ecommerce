"""
Módulo de transformação, limpeza e feature engineering.

Implementa o pipeline de transformação de dados usando Polars e DuckDB,
com atenção especial para evitar data leakage nas features preditivas.
"""

import duckdb
import polars as pl
import numpy as np

from src.config import STATE_REGION_MAP


def build_analytical_dataset(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """
    Constrói o dataset analítico principal a partir do DuckDB.

    Aplica limpeza, tratamento de nulos, remoção de outliers e
    criação de features derivadas.

    Args:
        conn: Conexão ativa com o DuckDB.

    Returns:
        DataFrame Polars limpo e enriquecido com features.
    """
    print("[transform] Construindo dataset analítico...")

    df = conn.execute("""
        SELECT
            o.order_id,
            o.customer_id,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_approved_at,
            o.order_delivered_carrier_date,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,
            i.order_item_id,
            i.product_id,
            i.seller_id,
            i.price,
            i.freight_value,
            p.product_category_name,
            p.product_name_length,
            p.product_description_lenght AS product_description_length,
            p.product_photos_qty,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,
            c.customer_unique_id,
            c.customer_zip_code_prefix,
            c.customer_city,
            c.customer_state,
            s.seller_zip_code_prefix,
            s.seller_city,
            s.seller_state,
            r.review_score,
            pay.payment_type,
            pay.payment_installments,
            pay.payment_value
        FROM orders o
        INNER JOIN order_items i    ON o.order_id = i.order_id
        INNER JOIN products p       ON i.product_id = p.product_id
        INNER JOIN customers c      ON o.customer_id = c.customer_id
        INNER JOIN sellers s        ON i.seller_id = s.seller_id
        LEFT  JOIN order_reviews r  ON o.order_id = r.order_id
        LEFT  JOIN (
            SELECT order_id,
                   payment_type,
                   payment_installments,
                   payment_value,
                   ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY payment_sequential) AS rn
            FROM payments
        ) pay ON o.order_id = pay.order_id AND pay.rn = 1
        WHERE o.order_status = 'delivered'
    """).pl()

    print(f"  -> Registros brutos carregados: {len(df):,}")

    df = _clean_data(df)
    df = _create_temporal_features(df)
    df = _create_product_features(df)
    df = _create_geographic_features(df)
    df = _create_target(df)

    print(f"  -> Dataset final: {len(df):,} registros, {len(df.columns)} colunas")
    return df


def build_eda_dataset(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """
    Constrói dataset para análise exploratória (inclui todas as informações).

    Args:
        conn: Conexão ativa com o DuckDB.

    Returns:
        DataFrame Polars para EDA.
    """
    print("[transform] Construindo dataset para EDA...")

    df = conn.execute("""
        SELECT
            o.order_id,
            o.customer_id,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,
            i.price,
            i.freight_value,
            i.seller_id,
            p.product_category_name,
            c.customer_unique_id,
            c.customer_state,
            s.seller_state,
            r.review_score,
            pay.payment_type,
            pay.payment_installments
        FROM orders o
        INNER JOIN order_items i    ON o.order_id = i.order_id
        INNER JOIN products p       ON i.product_id = p.product_id
        INNER JOIN customers c      ON o.customer_id = c.customer_id
        INNER JOIN sellers s        ON i.seller_id = s.seller_id
        LEFT  JOIN order_reviews r  ON o.order_id = r.order_id
        LEFT  JOIN (
            SELECT order_id, payment_type, payment_installments,
                   ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY payment_sequential) AS rn
            FROM payments
        ) pay ON o.order_id = pay.order_id AND pay.rn = 1
    """).pl()

    # Converte timestamps
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Datetime("us")).alias(col))

    # Filtra pedidos entregues com datas válidas
    df_delivered = df.filter(
        (pl.col("order_status") == "delivered")
        & pl.col("order_purchase_timestamp").is_not_null()
        & pl.col("order_delivered_customer_date").is_not_null()
        & (pl.col("price") > 0)
    )

    # Cria features temporais para EDA
    df_delivered = df_delivered.with_columns([
        ((pl.col("order_delivered_customer_date") - pl.col("order_purchase_timestamp"))
         .dt.total_days().alias("delivery_time")),
        (pl.col("order_delivered_customer_date") > pl.col("order_estimated_delivery_date"))
        .cast(pl.Int8).alias("is_late"),
        pl.col("order_purchase_timestamp").dt.strftime("%Y-%m").alias("order_month"),
        pl.col("customer_state").replace_strict(STATE_REGION_MAP, default="Desconhecido").alias("region"),
    ])

    # Preenche review_score nulo com mediana
    median_score = df_delivered.select(pl.col("review_score").median()).item()
    df_delivered = df_delivered.with_columns(
        pl.col("review_score").fill_null(median_score)
    )

    # Traduz categorias usando tabela do DuckDB
    try:
        translations = conn.execute("""
            SELECT product_category_name, product_category_name_english
            FROM category_translation
        """).pl()
        translation_map = dict(
            zip(
                translations["product_category_name"].to_list(),
                translations["product_category_name_english"].to_list(),
            )
        )
        df_delivered = df_delivered.with_columns(
            pl.col("product_category_name")
            .replace_strict(translation_map, default=None)
            .fill_null(pl.col("product_category_name"))
            .alias("product_category_name")
        )
    except Exception:
        pass

    print(f"  -> Dataset EDA: {len(df_delivered):,} registros")
    return df_delivered


def _clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Remove registros inválidos e trata valores nulos."""
    initial = len(df)

    # Converte timestamps
    ts_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Datetime("us")).alias(col))

    # Filtra registros com datas essenciais e preços válidos
    df = df.filter(
        pl.col("order_purchase_timestamp").is_not_null()
        & pl.col("order_delivered_customer_date").is_not_null()
        & pl.col("order_estimated_delivery_date").is_not_null()
        & (pl.col("price") > 0)
        & (pl.col("freight_value") >= 0)
    )

    # Preenche nulos numéricos com mediana
    numeric_fill_cols = [
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
        "product_name_length", "product_description_length",
        "product_photos_qty", "payment_installments",
    ]
    for col in numeric_fill_cols:
        if col in df.columns:
            median_val = df.select(pl.col(col).median()).item()
            if median_val is not None:
                df = df.with_columns(pl.col(col).fill_null(median_val))

    # Preenche nulos categóricos
    if "payment_type" in df.columns:
        df = df.with_columns(pl.col("payment_type").fill_null("unknown"))
    if "product_category_name" in df.columns:
        df = df.with_columns(pl.col("product_category_name").fill_null("unknown"))

    # Preenche review_score nulo com mediana
    if "review_score" in df.columns:
        median_score = df.select(pl.col("review_score").median()).item()
        df = df.with_columns(pl.col("review_score").fill_null(median_score))

    print(f"  -> Limpeza: {initial:,} -> {len(df):,} registros ({initial - len(df):,} removidos)")
    return df


def _create_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cria features temporais a partir dos timestamps."""
    df = df.with_columns([
        # Dia da semana (0=segunda)
        pl.col("order_purchase_timestamp").dt.weekday().alias("purchase_weekday"),
        # Hora do dia
        pl.col("order_purchase_timestamp").dt.hour().alias("purchase_hour"),
        # Mês
        pl.col("order_purchase_timestamp").dt.month().alias("purchase_month"),
        # Trimestre
        pl.col("order_purchase_timestamp").dt.quarter().alias("purchase_quarter"),
        # Fim de semana
        (pl.col("order_purchase_timestamp").dt.weekday() >= 5)
        .cast(pl.Int8).alias("is_weekend"),
        # Tempo de entrega em dias (para EDA, NÃO para modelo de predição)
        ((pl.col("order_delivered_customer_date") - pl.col("order_purchase_timestamp"))
         .dt.total_days().alias("delivery_time")),
        # Tempo estimado de entrega em dias (disponível antes da entrega)
        ((pl.col("order_estimated_delivery_date") - pl.col("order_purchase_timestamp"))
         .dt.total_days().alias("estimated_delivery_days")),
        # Tempo até aprovação do pedido
        ((pl.col("order_approved_at") - pl.col("order_purchase_timestamp"))
         .dt.total_hours().alias("approval_time_hours")),
        # Mês-ano para agrupamentos
        pl.col("order_purchase_timestamp").dt.strftime("%Y-%m").alias("order_month"),
    ])

    # Trata valores negativos em approval_time_hours
    df = df.with_columns(
        pl.when(pl.col("approval_time_hours") < 0)
        .then(0.0)
        .otherwise(pl.col("approval_time_hours"))
        .alias("approval_time_hours")
    )

    # Preenche nulos de approval_time_hours com mediana
    median_approval = df.select(pl.col("approval_time_hours").median()).item()
    if median_approval is not None:
        df = df.with_columns(
            pl.col("approval_time_hours").fill_null(median_approval)
        )

    print("  -> Features temporais criadas.")
    return df


def _create_product_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cria features derivadas dos atributos de produto."""
    df = df.with_columns([
        # Volume do produto (cm³)
        (pl.col("product_length_cm") * pl.col("product_height_cm") * pl.col("product_width_cm"))
        .alias("product_volume_cm3"),
        # Razão frete/preço
        (pl.col("freight_value") / pl.col("price").clip(lower_bound=0.01))
        .alias("freight_price_ratio"),
        # Valor total do item
        (pl.col("price") + pl.col("freight_value")).alias("total_item_value"),
    ])

    # Densidade (peso/volume) — evita divisão por zero
    df = df.with_columns(
        pl.when(pl.col("product_volume_cm3") > 0)
        .then(pl.col("product_weight_g") / pl.col("product_volume_cm3"))
        .otherwise(0.0)
        .alias("product_density")
    )

    print("  -> Features de produto criadas.")
    return df


def _create_geographic_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cria features geográficas baseadas em seller e customer."""
    df = df.with_columns([
        # Mesmo estado
        (pl.col("customer_state") == pl.col("seller_state"))
        .cast(pl.Int8).alias("same_state"),
        # Mesma cidade
        (pl.col("customer_city") == pl.col("seller_city"))
        .cast(pl.Int8).alias("same_city"),
        # Região do cliente
        pl.col("customer_state").replace_strict(STATE_REGION_MAP, default="Desconhecido").alias("customer_region"),
        # Região do vendedor
        pl.col("seller_state").replace_strict(STATE_REGION_MAP, default="Desconhecido").alias("seller_region"),
    ])

    # Mesma região
    df = df.with_columns(
        (pl.col("customer_region") == pl.col("seller_region"))
        .cast(pl.Int8).alias("same_region")
    )

    print("  -> Features geográficas criadas.")
    return df


def _create_target(df: pl.DataFrame) -> pl.DataFrame:
    """Cria a variável-alvo: atraso na entrega."""
    df = df.with_columns(
        (pl.col("order_delivered_customer_date") > pl.col("order_estimated_delivery_date"))
        .cast(pl.Int8).alias("is_late")
    )

    late_count = df.filter(pl.col("is_late") == 1).height
    total = df.height
    print(f"  -> Target criado: {late_count:,} atrasos de {total:,} pedidos ({100*late_count/total:.1f}%)")
    return df
