import os
import pandas as pd
import tempfile
from mlops_agent.ingestion import data_process_tool


def test_data_process_tool_basic(tmp_path):
    # Create a small reference dataset
    ref_path = tmp_path / "reference.csv"
    ref_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    ref_df.to_csv(ref_path, index=False)

    # Incoming data with one new row
    input_df = pd.DataFrame({"A": [2, 5], "B": [4, 6]})

    result = data_process_tool(
        input_source=input_df, reference_dataset_path=str(ref_path)
    )

    assert "reference_data" in result
    assert "merged_data" in result
    assert isinstance(result["new_data"], pd.DataFrame)
    assert not result["new_data"].empty
