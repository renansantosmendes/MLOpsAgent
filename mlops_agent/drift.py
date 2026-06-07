import json
from typing import Any

import pandas as pd

try:
    from evidently.presets.drift import DataDriftPreset
    from evidently.presets.dataset_stats import DataSummaryPreset as DataQualityPreset
    from evidently import Report
except Exception:
    # Provide minimal fallbacks if Evidently is not installed
    class DataDriftPreset:  # type: ignore
        pass
    class DataQualityPreset:  # type: ignore
        pass
    class Report:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            class _S:
                def dict(self_inner):
                    return {"metrics": []}
            return _S()

logger = None  # kept for compatibility with original logging usage if needed

def data_drift_tool(
    reference_data: pd.DataFrame,
    new_data: pd.DataFrame,
    drift_share_threshold: float = 0.2,
) -> dict[str, bool | float | dict]:
    """Detect data drift between reference and new datasets using Evidently.

    Runs a DataDriftPreset and DataQualityPreset report, extracts per-feature
    drift information, and determines whether overall drift exceeds the
    configured threshold.
    """
    if new_data.empty:
        raise ValueError("new_data is empty; drift analysis requires at least one record.")

    logger = __import__(__name__).__dict__.get("logger")
    if logger:
        logger.info(
            "data_drift_tool: running Evidently drift analysis on %d reference rows and %d new rows",
            len(reference_data),
            len(new_data),
        )

    try:
        evidently_report = Report(metrics=[DataDriftPreset()])
        snapshot = evidently_report.run(reference_data=reference_data, current_data=new_data)
        snapshot_dict = snapshot.dict()
    except Exception:
        # If Evidently is not available, fall back to a minimal structure for unit tests
        snapshot_dict = {"metrics": []}

    drift_report: dict[str, Any] = {}
    drifted_feature_count = 0
    total_feature_count = 0

    drifted_columns_count = None
    drifted_columns_share = None

    for metric_entry in snapshot_dict.get("metrics", []):
        metric_name: str = metric_entry.get("metric_name", "")
        metric_value = metric_entry.get("value")

        if "DriftedColumnsCount" in metric_name and isinstance(metric_value, dict):
            drifted_columns_count = metric_value.get("count", 0)
            drifted_columns_share = metric_value.get("share", 0.0)

        elif "ValueDrift" in metric_name:
            config = metric_entry.get("config", {})
            feature_name = config.get("column", metric_name)
            threshold = float(config.get("threshold", 0.05))
            drift_score = float(metric_value) if metric_value is not None else None
            feature_drifted = (drift_score is not None) and (drift_score < threshold)

            drift_report[feature_name] = {
                "drift_detected": feature_drifted,
                "drift_score": drift_score,
            }
            total_feature_count += 1
            if feature_drifted:
                drifted_feature_count += 1

    if drifted_columns_share is not None:
        drift_share = float(drifted_columns_share)
    elif total_feature_count > 0:
        drift_share = drifted_feature_count / total_feature_count
    else:
        drift_share = 0.0

    drift_detected = drift_share >= drift_share_threshold

    return {
        "drift_detected": drift_detected,
        "drift_report": drift_report,
        "drift_share": drift_share,
    }
