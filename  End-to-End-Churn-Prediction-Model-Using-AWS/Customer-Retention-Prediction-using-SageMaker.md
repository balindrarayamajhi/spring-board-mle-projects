This project builds an end-to-end machine learning pipeline using AWS SageMaker to predict customer retention.

The workflow includes:
1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/uttamp/store-data  
   - The dataset is provided in `.xlsx` format  
   - Convert it to `.csv` for easier processing in SageMaker  
2. Data preprocessing using SageMaker Data Wrangler
3. Feature engineering and pipeline creation
4. Model training and evaluation using AUC-ROC
5.  Model deployment via SageMaker endpoint



The notebook responsible to download the kaggle dataset and convert to csv [here](kaggle_data.ipynb) and 
generated csv file [here](storedata.csv)

Report of data preprocessing using SageMaker Data Wrangler [here](data-wrangler-insights-report.png)

Data Wrangler: Data Workflow screenshot [here](data-wrangler_data-workflow-screenshot.png)

- Handled missing values
- Created new features:
  - Customer tenure
  - Recency
  - First purchase delay
  - Engagement score
  - Lifetime value
- Encoded categorical variables (city, favorite day) 

