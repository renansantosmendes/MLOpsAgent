import pandas as pd
import mlops_agent.drift as drift


def test_data_drift_tool_monkeypatched(monkeypatch):
    reference_data = pd.DataFrame({"col1": [1, 2, 3]})
    new_data = pd.DataFrame({"col1": [1, 2]})

    class DummySnapshot:
        def __init__(self, data):
            self._data = data

        def dict(self):
            return self._data

    class DummyReport:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return DummySnapshot(
                {
                    "metrics": [
                        {
                            "metric_name": "ValueDrift",
                            "value": 0.1,
                            "config": {"column": "col1", "threshold": 0.2},
                        }
                    ]
                }
            )

    monkeypatch.setattr(drift, "Report", DummyReport)

    result = drift.data_drift_tool(
        reference_data=reference_data, new_data=new_data, drift_share_threshold=0.2
    )
    assert "drift_detected" in result
    assert isinstance(result["drift_report"], dict)
    assert result["drift_share"] >= 0.0
