import os
import json
import tempfile
from mlops_agent.deployment import deploy_application_tool
from unittest.mock import MagicMock


def test_deploy_application_tool_saves_artifacts(tmp_path, monkeypatch):
    """Test that deploy_application_tool saves model and metrics artifacts correctly."""
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    # Mock the requests.post to avoid real GitHub API call
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_response.text = ""

    def mock_post(*args, **kwargs):
        return mock_response

    import mlops_agent.deployment

    monkeypatch.setattr(mlops_agent.deployment.requests, "post", mock_post)

    # Use a minimal trained pipeline placeholder
    trained_pipeline = {"dummy": "pipeline"}
    new_model_metrics = {"f1_macro": 0.92, "accuracy": 0.88}

    result = deploy_application_tool(
        trained_pipeline=trained_pipeline,
        new_model_metrics=new_model_metrics,
        model_artifact_path=str(model_path),
        metrics_artifact_path=str(metrics_path),
        github_owner="test_owner",
        github_repo="test_repo",
        workflow_id="ci.yml",
        target_branch="main",
        github_token="fake_token",
    )

    assert result["deployment_triggered"] is True
    assert result["http_status_code"] == 204
    assert model_path.exists()
    assert metrics_path.exists()

    with open(metrics_path, "r") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["f1_macro"] == 0.92
