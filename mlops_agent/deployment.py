import json
import logging
from pathlib import Path
from typing import Any

import joblib
import requests

logger = logging.getLogger(__name__)


def deploy_application_tool(
    trained_pipeline,
    new_model_metrics: dict[str, float],
    model_artifact_path: str,
    metrics_artifact_path: str,
    github_owner: str,
    github_repo: str,
    workflow_id: str,
    target_branch: str,
    github_token: str,
) -> dict[str, bool | int | str]:
    """Persist the trained model and trigger a GitHub Actions deployment."""
    logger.info(
        "deploy_application_tool: saving model artifact to '%s'", model_artifact_path
    )
    try:
        Path(model_artifact_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_pipeline, model_artifact_path)
        logger.info("deploy_application_tool: model artifact saved successfully")
    except OSError as exc:
        raise OSError(
            f"Failed to save model artifact to '{model_artifact_path}': {exc}"
        ) from exc

    logger.info(
        "deploy_application_tool: saving metrics to '%s'", metrics_artifact_path
    )
    try:
        Path(metrics_artifact_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_artifact_path, "w", encoding="utf-8") as metrics_file:
            json.dump(new_model_metrics, metrics_file, indent=2)
        logger.info("deploy_application_tool: metrics file saved successfully")
    except OSError as exc:
        raise OSError(
            f"Failed to save metrics to '{metrics_artifact_path}': {exc}"
        ) from exc

    model_version = new_model_metrics.get("f1_macro", 0.0)
    dispatch_url = (
        f"https://api.github.com/repos/{github_owner}/{github_repo}"
        f"/actions/workflows/{workflow_id}/dispatches"
    )
    dispatch_payload = {
        "ref": target_branch,
        "inputs": {
            "model_version": str(model_version),
            "artifact_path": model_artifact_path,
            "f1_score": str(new_model_metrics.get("f1_macro", "")),
        },
    }
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    logger.info(
        "deploy_application_tool: dispatching workflow '%s' on branch '%s' for repo '%s/%s'",
        workflow_id,
        target_branch,
        github_owner,
        github_repo,
    )

    try:
        api_response = requests.post(
            dispatch_url,
            headers=headers,
            json=dispatch_payload,
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        raise requests.exceptions.RequestException(
            f"GitHub API request failed (network error): {exc}"
        ) from exc

    http_status_code = api_response.status_code
    response_body = api_response.text
    deployment_triggered = 200 <= http_status_code < 300

    if deployment_triggered:
        logger.info(
            "deploy_application_tool: GitHub Actions dispatch succeeded (status=%d)",
            http_status_code,
        )
    else:
        raise RuntimeError(
            f"GitHub Actions dispatch failed with HTTP {http_status_code}: {response_body}"
        )

    return {
        "deployment_triggered": deployment_triggered,
        "http_status_code": http_status_code,
        "response_body": response_body,
    }
