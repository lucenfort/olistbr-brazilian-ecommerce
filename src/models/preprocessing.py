"""
Módulo de pré-processamento para modelagem preditiva.

Prepara as features para treinamento dos modelos, garantindo
ausência de data leakage e tratamento adequado de variáveis.
"""

from typing import Any

import numpy as np
import polars as pl
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import LEAKAGE_COLUMNS, RANDOM_STATE, TEST_SIZE


def prepare_features(df: pl.DataFrame) -> dict[str, Any]:
    """
    Prepara features para o modelo de predição de atraso.

    Garante que:
    - Nenhuma feature com data leakage é incluída
    - SMOTE é aplicado APENAS no conjunto de treino
    - Split treino/teste é estratificado

    Args:
        df: DataFrame Polars com todas as features.

    Returns:
        Dicionário com X_train, X_test, y_train, y_test e metadados.
    """
    print("[preprocessing] Preparando features para modelagem...")

    # Features numéricas seguras (disponíveis no momento do pedido)
    numeric_features = [
        "price",
        "freight_value",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_name_length",
        "product_description_length",
        "product_photos_qty",
        "product_volume_cm3",
        "product_density",
        "freight_price_ratio",
        "total_item_value",
        "estimated_delivery_days",
        "approval_time_hours",
        "payment_installments",
        "purchase_hour",
        "purchase_weekday",
        "purchase_month",
        "purchase_quarter",
        "is_weekend",
        "same_state",
        "same_city",
        "same_region",
    ]

    # Features categóricas
    categorical_features = [
        "payment_type",
        "customer_region",
        "seller_region",
    ]

    # Filtra apenas features disponíveis no DataFrame
    available_numeric = [f for f in numeric_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]

    # Verifica se não há features com data leakage
    for feat in available_numeric + available_categorical:
        if feat in LEAKAGE_COLUMNS:
            raise ValueError(f"Feature com data leakage detectada: {feat}")

    print(f"  -> Features numéricas: {len(available_numeric)}")
    print(f"  -> Features categóricas: {len(available_categorical)}")

    # Seleciona colunas e remove registros com target nulo
    cols_to_use = available_numeric + available_categorical + ["is_late"]
    df_model = df.select(cols_to_use).drop_nulls(subset=["is_late"])

    # Encode de features categóricas
    label_encoders = {}
    for cat_col in available_categorical:
        le = LabelEncoder()
        values = df_model[cat_col].fill_null("unknown").to_list()
        encoded = le.fit_transform(values)
        df_model = df_model.with_columns(
            pl.Series(cat_col, encoded).cast(pl.Float64)
        )
        label_encoders[cat_col] = le

    # Converte para numpy
    all_features = available_numeric + available_categorical
    X = df_model.select(all_features).to_numpy().astype(np.float64)
    y = df_model["is_late"].to_numpy().astype(np.int32)

    # Trata NaN e Inf remanescentes
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Split ESTRATIFICADO (antes do SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"  -> Treino: {X_train.shape[0]:,} amostras")
    print(f"  -> Teste:  {X_test.shape[0]:,} amostras")
    print(f"  -> Distribuição treino: {np.mean(y_train):.1%} positivos")
    print(f"  -> Distribuição teste:  {np.mean(y_test):.1%} positivos")

    # SMOTE apenas no treino
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"  -> Treino após SMOTE: {X_train_balanced.shape[0]:,} amostras "
          f"({np.mean(y_train_balanced):.1%} positivos)")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_balanced,
        "y_test": y_test,
        "feature_names": all_features,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "n_features": len(all_features),
        "train_size": X_train_scaled.shape[0],
        "test_size": X_test_scaled.shape[0],
        "class_distribution": {
            "train_positive_rate": float(np.mean(y_train_balanced)),
            "test_positive_rate": float(np.mean(y_test)),
        },
    }
