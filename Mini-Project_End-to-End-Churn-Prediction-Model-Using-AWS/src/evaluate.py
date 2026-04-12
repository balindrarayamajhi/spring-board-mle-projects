
import argparse
import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--test-path", type=str, default="/opt/ml/processing/test/test.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    return parser.parse_args()


def load_booster(model_tar_gz: str) -> xgb.Booster:
    extract_dir = "/tmp/xgb_model"
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(model_tar_gz) as tar:
        tar.extractall(path=extract_dir)

    # Built-in XGBoost usually writes xgboost-model under /opt/ml/model
    candidate_paths = [
        os.path.join(extract_dir, "xgboost-model"),
        os.path.join(extract_dir, "model.xgb"),
        os.path.join(extract_dir, "model.bin"),
    ]
    found = None
    for path in candidate_paths:
        if os.path.exists(path):
            found = path
            break

    if found is None:
        # fallback: first file in tar
        for root, _, files in os.walk(extract_dir):
            if files:
                found = os.path.join(root, files[0])
                break

    if found is None:
        raise FileNotFoundError("Could not locate extracted XGBoost model file inside model.tar.gz")

    booster = xgb.Booster()
    booster.load_model(found)
    return booster


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    test_df = pd.read_csv(args.test_path, header=None)
    y_true = test_df.iloc[:, 0].astype(int).to_numpy()
    X_test = test_df.iloc[:, 1:].to_numpy()

    dtest = xgb.DMatrix(X_test)
    booster = load_booster(args.model_path)
    y_pred_prob = booster.predict(dtest)
    auc = roc_auc_score(y_true, y_pred_prob)

    report = {
        "binary_classification_metrics": {
            "auc": {
                "value": float(auc),
                "standard_deviation": 0.0
            }
        }
    }

    output_path = os.path.join(args.output_dir, "evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
