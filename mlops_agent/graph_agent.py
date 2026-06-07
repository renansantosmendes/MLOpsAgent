from typing import Any

import pandas as pd

from mlops_agent.ingestion import data_process_tool
from mlops_agent.drift import data_drift_tool
from mlops_agent.training import (
    model_training_tool,
    model_selection_tool,
    model_evaluation_tool,
)
from mlops_agent.deployment import deploy_application_tool
from langgraph.graph import StateGraph, END


def node_data_process(state: dict) -> dict:
    result = data_process_tool(
        input_source=state["input_source"],
        reference_dataset_path=state["reference_dataset_path"],
    )
    return {**state, **result}


def node_data_drift(state: dict) -> dict:
    result = data_drift_tool(
        reference_data=state["reference_data"],
        new_data=state["new_data"],
    )
    return {**state, **result}


def node_model_training(state: dict) -> dict:
    result = model_training_tool(
        input_data_train=state["merged_data"],
        target_column=state["target_column"],
    )
    return {**state, **result}


def node_model_selection(state: dict) -> dict:
    result = model_selection_tool(candidate_results=state["candidate_results"])
    return {**state, **result}


def node_model_evaluation(state: dict) -> dict:
    test_features = state["test_data"].drop(columns=[state["target_column"]])
    test_target = state["test_data"][state["target_column"]]
    result = model_evaluation_tool(
        trained_pipeline=state["best_estimator"],
        input_data_test=test_features,
        target_test=test_target,
        current_model_metrics_path=state.get("current_model_metrics_path"),
    )
    return {**state, **result}


def node_deploy(state: dict) -> dict:
    result = deploy_application_tool(
        trained_pipeline=state["best_estimator"],
        new_model_metrics=state["new_model_metrics"],
        model_artifact_path=state["model_artifact_path"],
        metrics_artifact_path=state["metrics_artifact_path"],
        github_owner=state["github_owner"],
        github_repo=state["github_repo"],
        workflow_id=state["workflow_id"],
        target_branch=state["target_branch"],
        github_token=state["github_token"],
    )
    return {**state, **result}


def node_halt_no_change(state: dict) -> dict:
    reason = "No drift detected, no new data. Skipping retraining."
    return {**state, "halt_reason": reason}


def node_halt_underperforms(state: dict) -> dict:
    new_f1 = state.get("new_model_metrics", {}).get("f1_macro", float("nan"))
    prod_f1 = (state.get("current_model_metrics") or {}).get("f1_macro", float("nan"))
    reason = (
        f"New model underperforms. Skipping deployment. "
        f"new_f1_macro={new_f1:.4f}, production_f1_macro={prod_f1:.4f}"
    )
    return {**state, "halt_reason": reason}


def route_after_data_process(state: dict) -> str:
    if state.get("has_new_records"):
        return "data_drift"
    return "halt_no_change"


def route_after_data_drift(state: dict) -> str:
    if state.get("drift_detected") or state.get("has_new_records"):
        return "model_training"
    return "halt_no_change"


def route_after_evaluation(state: dict) -> str:
    if state.get("is_better"):
        return "deploy"
    return "halt_underperforms"


class MLOpsTrainingAgent:
    """Refactored agent with modular graph and pluggable nodes."""

    def __init__(
        self,
        reference_dataset_path: str,
        target_column: str,
        current_model_metrics_path: str | None,
        model_artifact_path: str,
        metrics_artifact_path: str,
        github_owner: str,
        github_repo: str,
        workflow_id: str,
        target_branch: str,
        github_token: str,
    ) -> None:
        self._config: dict[str, Any] = {
            "reference_dataset_path": reference_dataset_path,
            "target_column": target_column,
            "current_model_metrics_path": current_model_metrics_path,
            "model_artifact_path": model_artifact_path,
            "metrics_artifact_path": metrics_artifact_path,
            "github_owner": github_owner,
            "github_repo": github_repo,
            "workflow_id": workflow_id,
            "target_branch": target_branch,
            "github_token": github_token,
        }
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(dict)
        workflow.add_node("data_process", node_data_process)
        workflow.add_node("data_drift", node_data_drift)
        workflow.add_node("model_training", node_model_training)
        workflow.add_node("model_selection", node_model_selection)
        workflow.add_node("model_evaluation", node_model_evaluation)
        workflow.add_node("deploy", node_deploy)
        workflow.add_node("halt_no_change", node_halt_no_change)
        workflow.add_node("halt_underperforms", node_halt_underperforms)

        workflow.set_entry_point("data_process")

        workflow.add_conditional_edges(
            "data_process",
            route_after_data_process,
            {
                "data_drift": "data_drift",
                "halt_no_change": "halt_no_change",
            },
        )
        workflow.add_conditional_edges(
            "data_drift",
            route_after_data_drift,
            {
                "model_training": "model_training",
                "halt_no_change": "halt_no_change",
            },
        )

        workflow.add_edge("model_training", "model_selection")
        workflow.add_edge("model_selection", "model_evaluation")

        workflow.add_conditional_edges(
            "model_evaluation",
            route_after_evaluation,
            {
                "deploy": "deploy",
                "halt_underperforms": "halt_underperforms",
            },
        )

        workflow.add_edge("deploy", END)
        workflow.add_edge("halt_no_change", END)
        workflow.add_edge("halt_underperforms", END)

        return workflow.compile()

    def run(self, input_source: str | pd.DataFrame) -> dict:
        initial_state: dict[str, Any] = {**self._config, "input_source": input_source}
        final_state: dict[str, Any] = self._graph.invoke(initial_state)  # type: ignore
        return final_state
