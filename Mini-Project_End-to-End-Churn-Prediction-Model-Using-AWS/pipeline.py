
import json
import os
from pathlib import Path

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
CONFIG_PATH = BASE_DIR / "config" / "project_config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_model_package_group(model_package_group_name: str, region_name: str):
    sm = boto3.client("sagemaker", region_name=region_name)
    try:
        sm.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription="Model package group for tea store churn XGBoost pipeline",
        )


def build_pipeline():
    cfg = load_config()
    region = cfg["region"]
    boto_sess = boto3.Session(region_name=region)
    sm_session = PipelineSession(boto_session=boto_sess)
    role = sagemaker.get_execution_role(sm_session)

    cache_config = CacheConfig(enable_caching=False, expire_after="30d")

    input_data = ParameterString(name="InputDataS3Uri", default_value=cfg["input_data_s3_uri"])
    pipeline_root = ParameterString(name="PipelineRoot", default_value=cfg["pipeline_root"])
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value=cfg["processing_instance_type"]
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value=cfg["training_instance_type"]
    )
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount", default_value=cfg["training_instance_count"]
    )
    auc_threshold = ParameterFloat(name="AucThreshold", default_value=float(cfg["auc_threshold"]))
    model_package_group_name = ParameterString(
        name="ModelPackageGroupName", default_value=cfg["model_package_group_name"]
    )
    feature_group_name = ParameterString(
        name="FeatureGroupName", default_value=f"{cfg['feature_group_name_prefix']}"
    )
    feature_store_offline_s3_uri = ParameterString(
        name="FeatureStoreOfflineS3Uri", default_value=cfg["feature_store_offline_s3_uri"]
    )

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=sm_session,
        base_job_name=f"{cfg['base_job_prefix']}-prep",
    )

    step_process = ProcessingStep(
        name="PreprocessChurnData",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input",
                input_name="rawinput",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="featurestore", source="/opt/ml/processing/feature-store"),
            ProcessingOutput(output_name="metadata", source="/opt/ml/processing/metadata"),
        ],
        code=str(SRC_DIR / "preprocess.py"),
        job_arguments=[
            "--input-data",
            "/opt/ml/processing/input/storedata.csv",
        ],
        cache_config=cache_config,
    )

    xgb_image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version=cfg["xgboost_version"],
        py_version="py3",
        image_scope="training",
        instance_type=cfg["training_instance_type"],
    )

    xgb_estimator = Estimator(
        image_uri=xgb_image_uri,
        role=role,
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        output_path=pipeline_root,
        sagemaker_session=sm_session,
        base_job_name=f"{cfg['base_job_prefix']}-train",
    )
    xgb_estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        num_round=200,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.8,
        silent=0,
    )

    step_train = TrainingStep(
        name="TrainXGBoostModel",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    eval_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            image_scope="processing",
            instance_type=cfg["processing_instance_type"],
        ),
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=sm_session,
        base_job_name=f"{cfg['base_job_prefix']}-eval",
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateXGBoostModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ],
        code=str(SRC_DIR / "evaluate.py"),
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{step_eval.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri}/evaluation.json",
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterTeaStoreChurnModel",
        estimator=xgb_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[cfg["endpoint_instance_type"]],
        transform_instances=[cfg["endpoint_instance_type"]],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    cond = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.auc.value",
        ),
        right=auc_threshold,
    )

    step_cond = ConditionStep(
        name="CheckAucThreshold",
        conditions=[cond],
        if_steps=[register_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=cfg["pipeline_name"],
        parameters=[
            input_data,
            pipeline_root,
            processing_instance_type,
            training_instance_type,
            training_instance_count,
            auc_threshold,
            model_package_group_name,
            feature_group_name,
            feature_store_offline_s3_uri,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sm_session,
    )
    return pipeline


if __name__ == "__main__":
    cfg = load_config()
    ensure_model_package_group(cfg["model_package_group_name"], cfg["region"])
    pipeline = build_pipeline()
    role = sagemaker.get_execution_role()
    upsert_response = pipeline.upsert(role_arn=role)
    print("Pipeline upserted.")
    print(upsert_response)
    execution = pipeline.start()
    print("Pipeline execution started:")
    print(execution.arn)
