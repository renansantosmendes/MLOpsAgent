import itertools
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # type: ignore
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

def model_training_tool(
    input_data_train: pd.DataFrame,
    target_column: str,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Train and cross-validate multiple candidate classification models.

    Builds a preprocessing pipeline (imputation, scaling, one-hot encoding),
    then evaluates each candidate model via stratified k-fold cross-validation.
    Returns raw cross-validation results for downstream model selection.
    """
    if target_column not in input_data_train.columns:
        raise ValueError(
            f"target_column '{target_column}' not found in input_data_train. "
            f"Available columns: {list(input_data_train.columns)}"
        )

    logger.info(
        "model_training_tool: starting training on %d rows, target='%s', cv_folds=%d",
        len(input_data_train),
        target_column,
        cv_folds,
    )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(input_data_train[target_column])
    feature_data = input_data_train.drop(columns=[target_column])
    feature_columns = list(feature_data.columns)

    train_val_features, test_features, train_val_target, test_target = train_test_split(
        feature_data,
        encoded_target,
        test_size=0.15,
        random_state=42,
        stratify=encoded_target,
    )

    train_val_data = train_val_features.copy()
    train_val_data[target_column] = train_val_target
    test_data = test_features.copy()
    test_data[target_column] = test_target

    numerical_columns = feature_data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )

    candidate_classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting_XGB": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        ) if XGBClassifier is not None else None,
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SupportVectorMachine": SVC(probability=True, random_state=42),
        "KNearestNeighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    # Remove any None entries if XGBClassifier wasn't available
    candidate_classifiers = {k: v for k, v in candidate_classifiers.items() if v is not None}

    cross_val_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring_metrics = {"f1_macro": "f1_macro", "roc_auc": "roc_auc_ovr"}

    candidate_results: dict[str, Any] = {}

    for model_name, classifier in candidate_classifiers.items():
        logger.info("model_training_tool: training candidate '%s'", model_name)
        try:
            full_pipeline = Pipeline(
                steps=[
                    ("preprocessor", column_transformer),
                    ("classifier", classifier),
                ]
            )
            cv_results = cross_validate(
                full_pipeline,
                train_val_features,
                train_val_target,
                cv=cross_val_splitter,
                scoring=scoring_metrics,
                return_estimator=True,
                n_jobs=-1,
            )
            mean_f1 = float(np.mean(cv_results["test_f1_macro"]))
            std_f1 = float(np.std(cv_results["test_f1_macro"]))
            mean_roc_auc = float(np.mean(cv_results["test_roc_auc"]))
            std_roc_auc = float(np.std(cv_results["test_roc_auc"]))

            best_fold_index = int(np.argmax(cv_results["test_f1_macro"]))
            best_fold_pipeline = cv_results["estimator"][best_fold_index]

            full_pipeline_fitted = Pipeline(
                steps=[
                    ("preprocessor", column_transformer),
                    ("classifier", classifier),
                ]
            )
            full_pipeline_fitted.fit(train_val_features, train_val_target)

            candidate_results[model_name] = {
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1,
                "mean_roc_auc": mean_roc_auc,
                "std_roc_auc": std_roc_auc,
                "fitted_pipeline": full_pipeline_fitted,
                "cv_raw_scores": {
                    "f1_macro": cv_results["test_f1_macro"].tolist(),
                    "roc_auc": cv_results["test_roc_auc"].tolist(),
                },
            }
            logger.info(
                "model_training_tool: '%s' — mean_f1_macro=%.4f (±%.4f), mean_roc_auc=%.4f (±%.4f)",
                model_name,
                mean_f1,
                std_f1,
                mean_roc_auc,
                std_roc_auc,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Training candidate model '{model_name}' failed: {exc}"
            ) from exc

    return {
        "candidate_results": candidate_results,
        "feature_columns": feature_columns,
        "label_encoder": label_encoder,
        "train_val_data": train_val_data,
        "test_data": test_data,
        "target_column": target_column,
    }


def model_selection_tool(
    candidate_results: dict[str, Any],
    primary_metric: str = "f1_macro",
    tiebreaker_metric: str = "roc_auc",
) -> dict[str, Any]:
    """Select the best candidate model based on cross-validation metrics.

    Ranks all trained candidates by their mean cross-validated primary metric
    and uses the tiebreaker metric to resolve ties.
    """
    if not candidate_results:
        raise ValueError("candidate_results is empty; no models to select from.")

    primary_metric_key = f"mean_{primary_metric}"
    tiebreaker_metric_key = f"mean_{tiebreaker_metric}"

    all_candidates_metrics: dict[str, dict[str, float]] = {}
    for model_name, result in candidate_results.items():
        all_candidates_metrics[model_name] = {
            "mean_f1_macro": result["mean_f1_macro"],
            "std_f1_macro": result["std_f1_macro"],
            "mean_roc_auc": result["mean_roc_auc"],
            "std_roc_auc": result["std_roc_auc"],
        }

    sorted_candidates = sorted(
        candidate_results.items(),
        key=lambda item: (
            item[1][primary_metric_key],
            item[1][tiebreaker_metric_key],
        ),
        reverse=True,
    )

    selected_model_name, best_candidate_data = sorted_candidates[0]
    best_estimator = best_candidate_data["fitted_pipeline"]

    logger.info(
        "model_selection_tool: selected '%s' — %s=%.4f, %s=%.4f",
        selected_model_name,
        primary_metric_key,
        best_candidate_data[primary_metric_key],
        tiebreaker_metric_key,
        best_candidate_data[tiebreaker_metric_key],
    )

    return {
        "best_estimator": best_estimator,
        "selected_model_name": selected_model_name,
        "all_candidates_metrics": all_candidates_metrics,
        "best_f1_macro": best_candidate_data[primary_metric_key],
        "best_roc_auc": best_candidate_data[tiebreaker_metric_key],
    }


def model_evaluation_tool(
    trained_pipeline: Pipeline,
    input_data_test: pd.DataFrame,
    target_test: pd.Series,
    current_model_metrics_path: str | None,
    improvement_threshold: float = 0.005,
) -> dict[str, Any]:
    """Evaluate a newly trained model and compare it to the production model.

    Computes a full suite of classification metrics on the held-out test set and
    loads the current production model's metrics from disk for comparison. The
    new model is considered better if its F1-macro exceeds the production model's
    F1-macro by at least ``improvement_threshold``.
    """
    logger.info(
        "model_evaluation_tool: evaluating new model on %d test samples",
        len(input_data_test),
    )

    predicted_labels = trained_pipeline.predict(input_data_test)
    predicted_probabilities = trained_pipeline.predict_proba(input_data_test)

    number_of_classes = len(np.unique(target_test))
    roc_auc_averaging = "ovr" if number_of_classes > 2 else "raise"

    if number_of_classes == 2:
        roc_auc_value = float(
            roc_auc_score(target_test, predicted_probabilities[:, 1])
        )
    else:
        roc_auc_value = float(
            roc_auc_score(
                target_test,
                predicted_probabilities,
                multi_class="ovr",
                average="macro",
            )
        )

    new_model_metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(target_test, predicted_labels)),
        "f1_macro": float(
            f1_score(target_test, predicted_labels, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(
                target_test, predicted_labels, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                target_test, predicted_labels, average="macro", zero_division=0
            )
        ),
        "roc_auc": roc_auc_value,
    }

    current_model_metrics: dict[str, float] | None = None
    if current_model_metrics_path is not None:
        import json
        from pathlib import Path

        metrics_path = Path(current_model_metrics_path)
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Current model metrics file not found: {current_model_metrics_path}"
            )
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            current_model_metrics = json.load(metrics_file)

    if current_model_metrics is None:
        is_better = True
        logger.info("model_evaluation_tool: no production model found; new model is accepted by default")
    else:
        production_f1 = current_model_metrics.get("f1_macro", 0.0)
        new_f1 = new_model_metrics["f1_macro"]
        is_better = (new_f1 - production_f1) >= improvement_threshold
        logger.info(
            "model_evaluation_tool: new_f1=%.4f, production_f1=%.4f, improvement=%.4f, threshold=%.4f, is_better=%s",
            new_f1,
            production_f1,
            new_f1 - production_f1,
            improvement_threshold,
            is_better,
        )

    return {
        "new_model_metrics": new_model_metrics,
        "current_model_metrics": current_model_metrics,
        "is_better": is_better,
    }
