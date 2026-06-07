import hashlib
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

def data_process_tool(
    input_source: str | pd.DataFrame,
    reference_dataset_path: str,
) -> dict[str, pd.DataFrame | bool]:
    """Ingest new data and compare it against the reference dataset.

    Loads the reference dataset from disk, then identifies new records in the
    incoming source by row-level hashing. New records are appended to the
    reference to form a merged dataset returned alongside diagnostic flags.
    """
    logger.info("data_process_tool: loading reference dataset from '%s'", reference_dataset_path)

    reference_data = pd.read_csv(reference_dataset_path)
    logger.info("data_process_tool: reference dataset loaded with %d rows", len(reference_data))

    if isinstance(input_source, str):
        logger.info("data_process_tool: fetching new data from URL '%s'", input_source)
        response = requests.get(input_source, timeout=60)
        response.raise_for_status()
        incoming_data = pd.read_csv(StringIO(response.text))
    elif isinstance(input_source, pd.DataFrame):
        incoming_data = input_source.copy()
    else:
        raise ValueError(
            f"input_source must be a URL string or pd.DataFrame, got {type(input_source)}"
        )
    logger.info("data_process_tool: incoming dataset has %d rows", len(incoming_data))

    def _row_hash(dataframe: pd.DataFrame) -> pd.Series:
        return dataframe.apply(
            lambda row: hashlib.md5(row.to_json().encode()).hexdigest(), axis=1
        )

    reference_hashes = set(_row_hash(reference_data))
    incoming_hashes = _row_hash(incoming_data)
    new_records_mask = ~incoming_hashes.isin(reference_hashes)
    new_data = incoming_data[new_records_mask].reset_index(drop=True)
    has_new_records = len(new_data) > 0

    logger.info(
        "data_process_tool: identified %d new record(s); has_new_records=%s",
        len(new_data),
        has_new_records,
    )

    merged_data = (
        pd.concat([reference_data, new_data], ignore_index=True)
        if has_new_records
        else reference_data.copy()
    )

    return {
        "reference_data": reference_data,
        "new_data": new_data,
        "merged_data": merged_data,
        "has_new_records": has_new_records,
    }
