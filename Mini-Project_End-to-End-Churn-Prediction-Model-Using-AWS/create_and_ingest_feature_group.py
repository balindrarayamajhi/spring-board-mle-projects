
import argparse
import json
import time
from pathlib import Path

import boto3
import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

from src.feature_store_utils import wait_for_feature_group

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "project_config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-group-name", required=True)
    parser.add_argument("--input-csv", required=True, help="S3 URI or local path to feature_store_ingest.csv")
    parser.add_argument("--role-arn", default=None)
    parser.add_argument("--offline-store-s3-uri", default=None)
    parser.add_argument("--target-feature-name", default="retained")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    boto_sess = boto3.Session(region_name=cfg["region"])
    session = sagemaker.Session(boto_session=boto_sess)
    role = args.role_arn or sagemaker.get_execution_role(session)
    offline_uri = args.offline_store_s3_uri or cfg["feature_store_offline_s3_uri"]

    df = pd.read_csv(args.input_csv)

    fg = FeatureGroup(name=args.feature_group_name, sagemaker_session=session)
    fg.load_feature_definitions(data_frame=df)

    fg.create(
        s3_uri=offline_uri,
        record_identifier_name="record_id",
        event_time_feature_name="event_time",
        role_arn=role,
        enable_online_store=True,
    )

    wait_for_feature_group(args.feature_group_name, region_name=cfg["region"])
    fg.ingest(data_frame=df, max_workers=3, wait=True)

    print(f"Feature group created and ingested: {args.feature_group_name}")
    print(f"Rows ingested: {len(df)}")
    print(f"Offline store: {offline_uri}")


if __name__ == "__main__":
    main()
