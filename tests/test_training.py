import pandas as pd
import numpy as np
from mlops_agent.training import (
    model_training_tool,
    model_selection_tool,
    model_evaluation_tool,
)


def test_model_training_tool():
    # Generate simple synthetic dataset
    data = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.choice(["A", "B"], size=100),
        }
    )

    result = model_training_tool(
        input_data_train=data, target_column="target", cv_folds=3
    )

    assert "candidate_results" in result
    assert len(result["candidate_results"]) > 0
    assert "label_encoder" in result
    assert "train_val_data" in result
    assert "test_data" in result


def test_model_selection_tool():
    candidate_results = {
        "ModelA": {
            "mean_f1_macro": 0.8,
            "std_f1_macro": 0.1,
            "mean_roc_auc": 0.85,
            "std_roc_auc": 0.05,
            "fitted_pipeline": None,
        },
        "ModelB": {
            "mean_f1_macro": 0.75,
            "std_f1_macro": 0.08,
            "mean_roc_auc": 0.82,
            "std_roc_auc": 0.06,
            "fitted_pipeline": None,
        },
    }

    result = model_selection_tool(candidate_results=candidate_results)

    assert result["selected_model_name"] == "ModelA"
    assert result["best_f1_macro"] == 0.8


def test_model_evaluation_tool_no_production_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Create a trivial trained pipeline
    pipeline = Pipeline([("classifier", LogisticRegression())])
    X_train = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [2, 3, 4, 5]})
    y_train = pd.Series([0, 1, 0, 1])
    pipeline.fit(X_train, y_train)

    X_test = pd.DataFrame({"f1": [5, 6], "f2": [6, 7]})
    y_test = pd.Series([1, 0])

    result = model_evaluation_tool(
        trained_pipeline=pipeline,
        input_data_test=X_test,
        target_test=y_test,
        current_model_metrics_path=None,
    )

    assert "new_model_metrics" in result
    assert "is_better" in result
    assert result["is_better"] is True
