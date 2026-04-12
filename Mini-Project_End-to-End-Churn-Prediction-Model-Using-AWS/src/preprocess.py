
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input/storedata.csv")
    parser.add_argument("--train-output", type=str, default="/opt/ml/processing/train")
    parser.add_argument("--validation-output", type=str, default="/opt/ml/processing/validation")
    parser.add_argument("--test-output", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--feature-store-output", type=str, default="/opt/ml/processing/feature-store")
    parser.add_argument("--metadata-output", type=str, default="/opt/ml/processing/metadata")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def preprocess_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"retained"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    if "custid" not in df.columns:
        df["custid"] = np.arange(1, len(df) + 1)

    # Standard cleanup
    df = df.drop_duplicates().reset_index(drop=True)

    # Parse likely date columns if present
    for col in ["created", "firstorder", "lastorder"]:
        if col in df.columns:
            df[col] = safe_datetime(df[col])

    # Numeric cleanup
    for col in df.columns:
        if col != "retained":
            if pd.api.types.is_object_dtype(df[col]):
                # Try numeric conversion first; fallback to strings later
                converted = pd.to_numeric(df[col], errors="ignore")
                df[col] = converted

    # Domain-oriented feature engineering inspired by the AWS churn example
    if {"firstorder", "lastorder"}.issubset(df.columns):
        df["first_last_days_diff"] = (df["lastorder"] - df["firstorder"]).dt.days

    if {"created", "firstorder"}.issubset(df.columns):
        df["created_first_days_diff"] = (df["created"] - df["firstorder"]).dt.days

    if {"created", "lastorder"}.issubset(df.columns):
        df["created_last_days_diff"] = (df["created"] - df["lastorder"]).dt.days

    # Fill missing values: numeric with median, datetime-derived leftovers with median, categorical with mode
    for col in df.columns:
        if col == "retained":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(med) else med)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Convert datetimes to ordinal seconds then fill
            epoch = pd.Timestamp("1970-01-01")
            seconds = (df[col] - epoch).dt.total_seconds()
            med = seconds.median()
            df[col] = seconds.fillna(0 if pd.isna(med) else med)
        else:
            mode = df[col].mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill_value).astype(str).str.strip().replace("", fill_value)

    # Ensure binary integer target
    df["retained"] = pd.to_numeric(df["retained"], errors="coerce").fillna(0).astype(int)
    if not set(df["retained"].unique()).issubset({0, 1}):
        # Coerce non-binary to 0/1 based on non-zero
        df["retained"] = (df["retained"] != 0).astype(int)

    record_id = df["custid"].astype(str)

    # Remove columns not suitable for training
    drop_for_training = [c for c in ["custid"] if c in df.columns]
    train_df = df.drop(columns=drop_for_training)

    # One-hot encode categoricals
    categorical_cols = [c for c in train_df.columns if train_df[c].dtype == "object" and c != "retained"]
    if categorical_cols:
        train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)

    # Clean inf/nan after transforms
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    for col in train_df.columns:
        if col == "retained":
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(0 if pd.isna(med) else med)

    feature_columns = [c for c in train_df.columns if c != "retained"]

    X = train_df[feature_columns]
    y = train_df["retained"]

    X_train, X_temp, y_train, y_temp, rid_train, rid_temp = train_test_split(
        X,
        y,
        record_id,
        test_size=0.30,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    relative_val_size = 0.15 / 0.30
    X_validation, X_test, y_validation, y_test, rid_validation, rid_test = train_test_split(
        X_temp,
        y_temp,
        rid_temp,
        test_size=(1 - relative_val_size),
        random_state=42,
        stratify=y_temp if y_temp.nunique() > 1 else None,
    )

    train_xgb = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    validation_xgb = pd.concat([y_validation.reset_index(drop=True), X_validation.reset_index(drop=True)], axis=1)
    test_xgb = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

    event_time_value = datetime.now(timezone.utc).isoformat()
    feature_store_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    feature_store_df.insert(0, "record_id", rid_train.reset_index(drop=True))
    feature_store_df["event_time"] = event_time_value

    test_labels = pd.DataFrame({"label": y_test.reset_index(drop=True)})

    metadata = {
        "feature_columns": feature_columns,
        "train_rows": int(len(train_xgb)),
        "validation_rows": int(len(validation_xgb)),
        "test_rows": int(len(test_xgb)),
        "feature_store_rows": int(len(feature_store_df)),
        "event_time_example": event_time_value,
    }
    return train_xgb, validation_xgb, test_xgb, feature_store_df, test_labels, metadata


def main():
    args = parse_args()

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.validation_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)
    os.makedirs(args.feature_store_output, exist_ok=True)
    os.makedirs(args.metadata_output, exist_ok=True)

    df = pd.read_csv(args.input_data)

    train_xgb, validation_xgb, test_xgb, feature_store_df, test_labels, metadata = preprocess_dataframe(df)

    train_xgb.to_csv(os.path.join(args.train_output, "train.csv"), header=False, index=False)
    validation_xgb.to_csv(os.path.join(args.validation_output, "validation.csv"), header=False, index=False)
    test_xgb.to_csv(os.path.join(args.test_output, "test.csv"), header=False, index=False)
    test_labels.to_csv(os.path.join(args.test_output, "test_labels.csv"), header=False, index=False)
    feature_store_df.to_csv(os.path.join(args.feature_store_output, "feature_store_ingest.csv"), index=False)

    with open(os.path.join(args.metadata_output, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Preprocessing complete.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
