"""
Configurações globais do projeto de análise de e-commerce Olist.

Define constantes, caminhos, paletas de cores e mapeamentos utilizados
em todos os módulos do pipeline de dados.
"""

import os

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "olist.duckdb")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "resultado")
CHARTS_DIR = os.path.join(RESULTS_DIR, "graficos")
MODELS_DIR = os.path.join(RESULTS_DIR, "modelos")
REPORTS_DIR = os.path.join(RESULTS_DIR, "relatorios")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
)

DATASET_FILES = {
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_sellers_dataset.csv": "sellers",
    "olist_order_payments_dataset.csv": "payments",
    "olist_customers_dataset.csv": "customers",
    "product_category_name_translation.csv": "category_translation",
}

# ---------------------------------------------------------------------------
# Mapeamento de estados para regiões
# ---------------------------------------------------------------------------
STATE_REGION_MAP = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste",
    "MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# ---------------------------------------------------------------------------
# Paleta de cores profissional
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#0D47A1",
    "primary_light": "#5472D3",
    "primary_dark": "#002171",
    "secondary": "#00897B",
    "secondary_light": "#4EBAAA",
    "secondary_dark": "#005B4F",
    "accent": "#FF6F00",
    "accent_light": "#FFA040",
    "error": "#C62828",
    "warning": "#F9A825",
    "success": "#2E7D32",
    "bg_dark": "#0A1929",
    "bg_card": "#132F4C",
    "bg_light": "#F5F7FA",
    "text_dark": "#1A2027",
    "text_light": "#B2BAC2",
    "text_white": "#FFFFFF",
    "grid": "#1E3A5F",
}

# Sequência de cores para gráficos com múltiplas séries
COLOR_SEQUENCE = [
    "#0D47A1", "#00897B", "#FF6F00", "#C62828",
    "#6A1B9A", "#00695C", "#EF6C00", "#AD1457",
    "#283593", "#1565C0",
]

# ---------------------------------------------------------------------------
# Configurações de gráficos (Plotly)
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["bg_dark"],
        "plot_bgcolor": COLORS["bg_dark"],
        "font": {"color": COLORS["text_white"], "family": "Inter, sans-serif"},
        "title": {"font": {"size": 20, "color": COLORS["text_white"]}},
        "colorway": COLOR_SEQUENCE,
        "xaxis": {
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
        },
        "yaxis": {
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
        },
    }
}

# ---------------------------------------------------------------------------
# Configurações de modelagem
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OPTUNA_TRIALS = 5

# Target: prever atraso na entrega
TARGET_COLUMN = "is_late"

# Features que NÃO devem ser usadas (data leakage)
LEAKAGE_COLUMNS = [
    "delivery_time",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
]
