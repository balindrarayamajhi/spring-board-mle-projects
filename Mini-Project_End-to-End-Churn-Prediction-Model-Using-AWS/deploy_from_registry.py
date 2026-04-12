
import argparse
import json
from pathlib import Path

import boto3
import sagemaker
from sagemaker.model import ModelPackage
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "project_config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-status", default="Approved", choices=["Approved", "PendingManualApproval", "Rejected"])
    parser.add_argument("--endpoint-name", default="tea-store-churn-xgb-endpoint")
    parser.add_argument("--instance-type", default=None)
    return parser.parse_args()


def get_latest_model_package(model_package_group_name: str, approval_status: str, region_name: str):
    sm = boto3.client("sagemaker", region_name=region_name)
    resp = sm.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus=approval_status,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    summaries = resp.get("ModelPackageSummaryList", [])
    if not summaries:
        raise RuntimeError(
            f"No model package found in group '{model_package_group_name}' with approval status '{approval_status}'."
        )
    return summaries[0]["ModelPackageArn"]


def main():
    args = parse_args()
    cfg = load_config()

    session = sagemaker.Session(boto_session=boto3.Session(region_name=cfg["region"]))
    role = sagemaker.get_execution_role(session)
    model_package_arn = get_latest_model_package(
        cfg["model_package_group_name"], args.approval_status, cfg["region"]
    )

    model = ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=session,
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type or cfg["endpoint_instance_type"],
        endpoint_name=args.endpoint_name,
    )
    predictor.serializer = CSVSerializer()
    predictor.deserializer = JSONDeserializer()

    print("Endpoint deployed.")
    print(f"Model package ARN: {model_package_arn}")
    print(f"Endpoint name: {args.endpoint_name}")


if __name__ == "__main__":
    main()
