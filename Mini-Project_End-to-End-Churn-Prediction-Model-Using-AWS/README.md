# End-to-End Churn Prediction Model Using AWS SageMaker

This project is a submission-ready SageMaker solution for the mini-project instructions. It uses:

- **Amazon SageMaker Processing** for data preparation
- **Amazon SageMaker Feature Store** for feature ingestion and querying
- **Amazon SageMaker built-in XGBoost** for model training
- **Amazon SageMaker Pipelines** for orchestration
- **AUC-ROC** for evaluation
- **Model Registry** for conditional registration


## What this covers

1. Load the Kaggle tea-store dataset after you convert it from XLSX to CSV.
2. Preprocess, clean, and engineer features.
3. Export training, validation, and test datasets for the built-in XGBoost algorithm.
4. Create and ingest a SageMaker Feature Store feature group.
5. Train an XGBoost model with `objective=binary:logistic` and `eval_metric=auc`.
6. Evaluate the model with AUC-ROC on the held-out test set.
7. Register the model only if it clears the configured AUC threshold.
8. Deploy the approved model from the Model Registry.
9. Query both offline and online Feature Store data.

## Files to submit

For grading, the most important file is:

- `notebooks/churn_assignment_submission.ipynb`

You can also include the Python scripts if your instructor wants the full project.

## Before you run

1. Convert the Kaggle XLSX file to CSV using `notebooks/customer_churn_file_convertor.ipynb` and upload it to S3.
2. Open SageMaker Studio.
3. Upload this whole project folder.
4. Open the notebook in `notebooks/customer_churn_prediction.ipynb`.

## Notes

- The pipeline expects the raw CSV to be in S3.
- The preprocessing script automatically creates:
  - train CSV
  - validation CSV
  - test CSV
  - test labels CSV
  - a Feature Store ingestion CSV with `record_id` and `event_time`
  - a JSON file containing the final feature column order
- The built-in XGBoost container expects the label in the **first column** and no header row.



