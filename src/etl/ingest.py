"""
Módulo de ingestão de dados CSV para DuckDB utilizando Polars.

Lê os arquivos CSV com Polars, carrega no DuckDB e cria índices
para otimização de consultas analíticas.
"""

import os

import duckdb
import polars as pl

from src.config import DATA_DIR, DB_PATH, DATASET_FILES


def ingest_to_duckdb() -> duckdb.DuckDBPyConnection:
    """
    Importa todos os CSVs para o DuckDB via Polars.

    Returns:
        Conexão ativa com o banco DuckDB.
    """
    # Remove banco anterior para garantir dados limpos
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = duckdb.connect(DB_PATH)
    print("[ingest] Iniciando ingestão dos CSVs no DuckDB...")

    for csv_file, table_name in DATASET_FILES.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"[ingest] AVISO: {csv_file} não encontrado. Pulando.")
            continue

        df = pl.read_csv(csv_path, infer_schema_length=10000, try_parse_dates=True)

        # Corrige nome de coluna com typo no arquivo de produtos
        if table_name == "products" and "product_name_lenght" in df.columns:
            df = df.rename({"product_name_lenght": "product_name_length"})

        # Registra o DataFrame Polars no DuckDB e cria tabela persistente
        conn.register("_tmp_df", df.to_arrow())
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
        conn.unregister("_tmp_df")

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  -> Tabela '{table_name}': {row_count:,} registros")

    _create_indexes(conn)
    _create_views(conn)

    print("[ingest] Ingestão concluída com sucesso.")
    return conn


def _create_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Cria índices para otimização de consultas."""
    # DuckDB não requer índices explícitos para a maioria dos casos
    # pois usa um otimizador baseado em vectorized execution.
    # Mas podemos criar para joins frequentes.
    print("[ingest] DuckDB utiliza otimização vetorizada automática.")


def _create_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Cria views analíticas para consultas recorrentes."""
    conn.execute("""
        CREATE OR REPLACE VIEW vw_orders_complete AS
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
            i.shipping_limit_date,
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
            s.seller_state
        FROM orders o
        INNER JOIN order_items i    ON o.order_id = i.order_id
        INNER JOIN products p       ON i.product_id = p.product_id
        INNER JOIN customers c      ON o.customer_id = c.customer_id
        INNER JOIN sellers s        ON i.seller_id = s.seller_id
    """)

    conn.execute("""
        CREATE OR REPLACE VIEW vw_orders_with_reviews AS
        SELECT
            oc.*,
            r.review_score,
            r.review_comment_title,
            r.review_comment_message
        FROM vw_orders_complete oc
        LEFT JOIN order_reviews r ON oc.order_id = r.order_id
    """)

    conn.execute("""
        CREATE OR REPLACE VIEW vw_orders_with_payments AS
        SELECT
            oc.*,
            pay.payment_type,
            pay.payment_installments,
            pay.payment_value
        FROM vw_orders_complete oc
        INNER JOIN payments pay ON oc.order_id = pay.order_id
    """)

    print("[ingest] Views analíticas criadas.")
