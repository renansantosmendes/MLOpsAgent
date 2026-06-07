# MLOps Training Agent

[![CI Pipeline](https://github.com/renansantosmendes/MLOpsAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/renansantosmendes/MLOpsAgent/actions/workflows/ci.yml)

Production-grade MLOps agent for automated model retraining and deployment.

## Features

- **Data Ingestion**: Compare incoming data with reference dataset using row-level hashing
- **Drift Detection**: Detect data drift using Evidently AI
- **Model Training**: Train and cross-validate multiple classification models
- **Model Selection**: Automatically select the best performing model
- **Model Evaluation**: Compare new model against production baseline
- **Automated Deployment**: Trigger GitHub Actions workflows for deployment

## Project Structure

```
MLOpsAgent/
├── mlops_agent/
│   ├── __init__.py
│   ├── ingestion.py       # Data ingestion and comparison
│   ├── drift.py           # Drift detection with Evidently
│   ├── training.py        # Model training, selection, evaluation
│   ├── deployment.py      # GitHub Actions deployment
│   └── graph_agent.py     # LangGraph-based orchestration
├── tests/
│   ├── test_ingestion.py
│   ├── test_drift.py
│   ├── test_training.py
│   └── test_deployment.py
├── .github/
│   └── workflows/
│       └── ci.yml         # CI/CD pipeline
├── script.py              # Legacy monolithic script (deprecated)
├── requirements.txt
├── pytest.ini
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from mlops_agent.graph_agent import MLOpsTrainingAgent

agent = MLOpsTrainingAgent(
    reference_dataset_path="data/reference.csv",
    target_column="label",
    current_model_metrics_path="artifacts/metrics.json",
    model_artifact_path="artifacts/model.joblib",
    metrics_artifact_path="artifacts/metrics.json",
    github_owner="renansantosmendes",
    github_repo="MLOpsAgent",
    workflow_id="deploy.yml",
    target_branch="main",
    github_token="ghp_xxx",
)

final_state = agent.run("https://example.com/new_data.csv")
```

## Architecture

This agent uses **StateGraph with Nodes** instead of **LLM Agent with Tools**. Why?

- ⚡ **Performance**: No LLM calls → fast execution
- 💰 **Cost**: Zero LLM costs → predictable budget
- 🎯 **Reliability**: Deterministic flow → consistent behavior
- 🔍 **Debuggability**: Clear execution path → easy troubleshooting

**📖 Want to understand the difference?**
- [Architecture Decision Document](ARCHITECTURE.md)
- [Comparison Examples](examples/README.md)

```python
# This is a NODE (no @tool decorator needed)
def node_data_drift(state: dict) -> dict:
    """Processes state through drift detection - part of fixed graph flow"""
    result = data_drift_tool(state["reference_data"], state["new_data"])
    return {**state, **result}

# NOT a TOOL (which would be used by LLM reasoning)
# @tool  ← We don't use this!
# def detect_drift(...): ...
```

## Testing

Run unit tests with pytest:

```bash
pytest tests/ -v --cov=mlops_agent
```

## CI/CD

GitHub Actions workflow automatically:
- Lints code with flake8
- Checks formatting with black
- Type checks with mypy
- Runs unit tests with pytest
- Uploads coverage to Codecov

## License

MIT
