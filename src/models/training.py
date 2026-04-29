"""
Módulo de treinamento de modelos com otimização de hiperparâmetros.

Treina quatro modelos de classificação com Optuna para otimização
bayesiana de hiperparâmetros e validação cruzada estratificada.
"""

import warnings
from typing import Any

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

from src.config import RANDOM_STATE, CV_FOLDS, OPTUNA_TRIALS

# Suprime logs verbosos do Optuna e LightGBM
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def _get_cv() -> StratifiedKFold:
    """Retorna validação cruzada estratificada."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Funções objetivo para Optuna
# ---------------------------------------------------------------------------

def _objective_logistic(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Objetivo para Logistic Regression."""
    C = trial.suggest_float("C", 1e-3, 100, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    model = LogisticRegression(
        C=C, penalty=penalty, solver="saga",
        max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1,
    )
    scores = cross_val_score(model, X, y, cv=_get_cv(), scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_rf(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Objetivo para Random Forest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    scores = cross_val_score(model, X, y, cv=_get_cv(), scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_xgboost(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Objetivo para XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = xgb.XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    scores = cross_val_score(model, X, y, cv=_get_cv(), scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_lgbm(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Objetivo para LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }
    model = lgb.LGBMClassifier(
        **params,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    scores = cross_val_score(model, X, y, cv=_get_cv(), scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ---------------------------------------------------------------------------
# Treinamento principal
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """
    Treina os quatro modelos com otimização de hiperparâmetros via Optuna.

    Args:
        X_train: Features de treino (já balanceadas e escaladas).
        y_train: Target de treino.
        n_trials: Número de trials do Optuna (default: OPTUNA_TRIALS).

    Returns:
        Dicionário com modelos treinados e seus melhores hiperparâmetros.
    """
    if n_trials is None:
        n_trials = OPTUNA_TRIALS

    print(f"[training] Treinando 4 modelos com {n_trials} trials Optuna cada...")

    models = {}

    # 1. Logistic Regression
    print("\n  [1/4] Logistic Regression...")
    study_lr = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lr.optimize(lambda trial: _objective_logistic(trial, X_train, y_train), n_trials=n_trials)
    best_lr = study_lr.best_params
    model_lr = LogisticRegression(
        C=best_lr["C"], penalty=best_lr["penalty"],
        solver="saga", max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model_lr.fit(X_train, y_train)
    models["Logistic Regression"] = {
        "model": model_lr,
        "best_params": best_lr,
        "best_cv_auc": study_lr.best_value,
    }
    print(f"    -> CV AUC: {study_lr.best_value:.4f}")

    # 2. Random Forest
    print("  [2/4] Random Forest...")
    study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(lambda trial: _objective_rf(trial, X_train, y_train), n_trials=n_trials)
    best_rf = study_rf.best_params
    model_rf = RandomForestClassifier(**best_rf, random_state=RANDOM_STATE, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    models["Random Forest"] = {
        "model": model_rf,
        "best_params": best_rf,
        "best_cv_auc": study_rf.best_value,
    }
    print(f"    -> CV AUC: {study_rf.best_value:.4f}")

    # 3. XGBoost
    print("  [3/4] XGBoost...")
    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(lambda trial: _objective_xgboost(trial, X_train, y_train), n_trials=n_trials)
    best_xgb = study_xgb.best_params
    model_xgb = xgb.XGBClassifier(
        **best_xgb, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
    )
    model_xgb.fit(X_train, y_train)
    models["XGBoost"] = {
        "model": model_xgb,
        "best_params": best_xgb,
        "best_cv_auc": study_xgb.best_value,
    }
    print(f"    -> CV AUC: {study_xgb.best_value:.4f}")

    # 4. LightGBM
    print("  [4/4] LightGBM...")
    study_lgbm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgbm.optimize(lambda trial: _objective_lgbm(trial, X_train, y_train), n_trials=n_trials)
    best_lgbm = study_lgbm.best_params
    model_lgbm = lgb.LGBMClassifier(
        **best_lgbm, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    model_lgbm.fit(X_train, y_train)
    models["LightGBM"] = {
        "model": model_lgbm,
        "best_params": best_lgbm,
        "best_cv_auc": study_lgbm.best_value,
    }
    print(f"    -> CV AUC: {study_lgbm.best_value:.4f}")

    print("\n[training] Treinamento concluído.")
    return models
