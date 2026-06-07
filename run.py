from mlops_agent.graph_agent import MLOpsTrainingAgent

agent = MLOpsTrainingAgent(
        reference_dataset_path="https://raw.githubusercontent.com/renansantosmendes/mlops-datasets/refs/heads/main/data_20230720_04-51-43.csv", # seu CSV de referência
        target_column="fetal_health", # nome da coluna alvo
        current_model_metrics_path=None, # None se não tem modelo em produção ainda
        model_artifact_path="artifacts/model.joblib",
        metrics_artifact_path="artifacts/metrics.json",
        github_owner="seu-usuario",
        github_repo="seu-repositorio",
        workflow_id="retrain.yml",
        target_branch="main",
        github_token="ghp_...", # seu token do GitHub
    )

final_state = agent.run("https://raw.githubusercontent.com/renansantosmendes/mlops-datasets/refs/heads/main/data_20230725_09-15-46.csv")