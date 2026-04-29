"""
Módulo de avaliação e comparação de modelos.

Calcula métricas de classificação, gera relatórios comparativos
e produz dados para visualizações de performance.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import learning_curve, StratifiedKFold

from src.config import RANDOM_STATE, CV_FOLDS


def evaluate_all_models(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Avalia todos os modelos no conjunto de teste e gera relatórios comparativos.

    Args:
        models: Dicionário de modelos treinados.
        X_test: Features de teste.
        y_test: Target de teste.
        X_train: Features de treino (para learning curves).
        y_train: Target de treino.
        feature_names: Nomes das features.

    Returns:
        Dicionário com todas as métricas e dados para visualização.
    """
    print("[evaluation] Avaliando modelos no conjunto de teste...")

    results = {}
    comparison_table = []

    for name, model_data in models.items():
        model = model_data["model"]
        print(f"\n  -> {name}:")

        # Predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métricas
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
        }

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        # Curva Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)

        # Feature importance (se disponível)
        fi = _get_feature_importance(model, feature_names)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        results[name] = {
            "metrics": metrics,
            "roc_curve": {"fpr": fpr, "tpr": tpr},
            "pr_curve": {"precision": precision_curve, "recall": recall_curve},
            "confusion_matrix": cm,
            "feature_importance": fi,
            "classification_report": report,
            "best_params": model_data["best_params"],
            "best_cv_auc": model_data["best_cv_auc"],
        }

        comparison_table.append({
            "Modelo": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "CV-AUC": model_data["best_cv_auc"],
        })

        print(f"     Accuracy:  {metrics['accuracy']:.4f}")
        print(f"     Precision: {metrics['precision']:.4f}")
        print(f"     Recall:    {metrics['recall']:.4f}")
        print(f"     F1-Score:  {metrics['f1']:.4f}")
        print(f"     ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"     PR-AUC:    {metrics['pr_auc']:.4f}")

    # Learning curves (apenas para o melhor modelo)
    best_model_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_model = models[best_model_name]["model"]

    print(f"\n  -> Gerando learning curves para o melhor modelo ({best_model_name})...")
    lc = _compute_learning_curves(best_model, X_train, y_train)

    # Determina o melhor modelo
    best_info = {
        "name": best_model_name,
        "metrics": results[best_model_name]["metrics"],
    }

    print(f"\n[evaluation] Melhor modelo: {best_model_name} "
          f"(ROC-AUC={results[best_model_name]['metrics']['roc_auc']:.4f})")

    return {
        "model_results": results,
        "comparison_table": comparison_table,
        "learning_curves": lc,
        "best_model": best_info,
    }


def _get_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float] | None:
    """Extrai feature importance do modelo, se disponível."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return None

        fi = dict(zip(feature_names, importances))
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        return None


def _compute_learning_curves(
    model: Any, X: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    """Calcula learning curves para análise de bias/variância."""
    try:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            cv=cv,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="roc_auc",
            random_state=RANDOM_STATE,
        )

        return {
            "train_sizes": train_sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "test_mean": test_scores.mean(axis=1),
            "test_std": test_scores.std(axis=1),
        }
    except Exception as e:
        print(f"  -> Aviso: não foi possível calcular learning curves: {e}")
        return {}
