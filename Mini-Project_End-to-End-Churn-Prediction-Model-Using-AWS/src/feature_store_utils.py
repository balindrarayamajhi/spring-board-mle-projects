
import time
from typing import Iterable, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError


def wait_for_feature_group(feature_group_name: str, region_name: Optional[str] = None, timeout_seconds: int = 1800):
    sm = boto3.client("sagemaker", region_name=region_name)
    start = time.time()
    while True:
        response = sm.describe_feature_group(FeatureGroupName=feature_group_name)
        status = response["FeatureGroupStatus"]
        if status == "Created":
            return response
        if status == "CreateFailed":
            raise RuntimeError(f"Feature group creation failed: {response}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for feature group {feature_group_name} to reach Created state")
        time.sleep(15)


def get_offline_store_table(feature_group_name: str, region_name: Optional[str] = None):
    sm = boto3.client("sagemaker", region_name=region_name)
    desc = sm.describe_feature_group(FeatureGroupName=feature_group_name)
    offline_cfg = desc.get("OfflineStoreConfig", {})
    data_catalog = offline_cfg.get("DataCatalogConfig", {})
    return {
        "database": data_catalog.get("Database"),
        "table_name": data_catalog.get("TableName"),
        "s3_uri": offline_cfg.get("S3StorageConfig", {}).get("S3Uri"),
        "description": desc,
    }


def run_athena_query(
    sql: str,
    database: str,
    output_s3_uri: str,
    region_name: Optional[str] = None,
    poll_seconds: int = 5,
):
    athena = boto3.client("athena", region_name=region_name)
    start = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_s3_uri},
    )
    qid = start["QueryExecutionId"]

    while True:
        result = athena.get_query_execution(QueryExecutionId=qid)
        state = result["QueryExecution"]["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break
        time.sleep(poll_seconds)

    if state != "SUCCEEDED":
        raise RuntimeError(f"Athena query ended in state {state}: {result}")

    return qid


def get_athena_results_as_dataframe(query_execution_id: str, region_name: Optional[str] = None) -> pd.DataFrame:
    athena = boto3.client("athena", region_name=region_name)
    paginator = athena.get_paginator("get_query_results")

    rows = []
    headers = None
    for page in paginator.paginate(QueryExecutionId=query_execution_id):
        result_set = page["ResultSet"]
        page_rows = result_set["Rows"]
        if headers is None:
            headers = [c.get("VarCharValue", "") for c in page_rows[0]["Data"]]
            page_rows = page_rows[1:]
        for row in page_rows:
            values = [col.get("VarCharValue", None) for col in row["Data"]]
            rows.append(values)

    return pd.DataFrame(rows, columns=headers)


def query_latest_records_from_offline_store(
    feature_group_name: str,
    athena_output_s3_uri: str,
    limit: int = 20,
    region_name: Optional[str] = None,
) -> pd.DataFrame:
    info = get_offline_store_table(feature_group_name, region_name=region_name)
    database = info["database"]
    table_name = info["table_name"]
    if not database or not table_name:
        raise ValueError("Feature group offline store is not configured with AWS Glue Data Catalog information.")

    sql = f"""
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY record_id
                   ORDER BY event_time DESC, api_invocation_time DESC, write_time DESC
               ) AS rn
        FROM "{database}"."{table_name}"
        WHERE is_deleted = false
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
    LIMIT {int(limit)}
    """
    qid = run_athena_query(sql, database, athena_output_s3_uri, region_name=region_name)
    return get_athena_results_as_dataframe(qid, region_name=region_name)


def get_online_features(
    feature_group_name: str,
    record_ids: Iterable[str],
    feature_names: Optional[List[str]] = None,
    region_name: Optional[str] = None,
):
    runtime = boto3.client("sagemaker-featurestore-runtime", region_name=region_name)
    results = []
    for rid in record_ids:
        response = runtime.get_record(
            FeatureGroupName=feature_group_name,
            RecordIdentifierValueAsString=str(rid),
            FeatureNames=feature_names or [],
        )
        flattened = {"record_id": str(rid)}
        for f in response.get("Record", []):
            name = f["FeatureName"]
            value = f.get("ValueAsString")
            flattened[name] = value
        results.append(flattened)
    return pd.DataFrame(results)
