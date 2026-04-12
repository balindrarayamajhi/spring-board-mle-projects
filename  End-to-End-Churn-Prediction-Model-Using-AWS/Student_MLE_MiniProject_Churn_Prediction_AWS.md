# Mini-Project: An End-to-End Churn Prediction Model Using AWS

Churn prediction is a crucial aspect for businesses, especially those operating in subscription-based models or industries with high customer turnover. Churn refers to the phenomenon where customers discontinue using a product or service. Predicting churn is important for several reasons:

1. **Revenue Impact:**
   - Retaining existing customers is often more cost-effective than acquiring new ones. Losing customers means losing the associated revenue. Churn prediction helps businesses identify customers at risk of leaving so that proactive measures can be taken to retain them.

2. **Resource Allocation:**
   - By predicting which customers are likely to churn, businesses can allocate resources more efficiently. They can concentrate efforts and resources on retaining high-value customers who are at a higher risk of leaving, rather than applying blanket retention strategies.

3. **Customer Experience Improvement:**
   - Understanding why customers churn provides valuable insights into areas that may need improvement. It could be issues related to product quality, customer service, or competitive factors. Identifying and addressing these issues can enhance overall customer experience.

In this mini-project, you'll be building an end-to-end churn prediction model using [AWS Sagemaker AI](https://aws.amazon.com/sagemaker-ai). The Kaggle dataset used in this project is for an online tea retail store, which contains customer data, including whether or not a given customer has been retained.

Your tasks are as follows:

1. Download the [pertinent Kaggle dataset](https://www.kaggle.com/datasets/uttamp/store-data). Note, the dataset is stored as an xlsx file. Convert this to a csv file for easier processing in SageMaker AI. 
2. Initiate a SageMaker Canvas instance using the AWS console. 
3. Open Data Wrangler and do a local upload of the tabular dataset.  
4. Use Data Wrangler to explore and clean the data. Be creative in engineering new features for use in building your machine learning model. Once you are satisfied with all data preprocessing steps, export your workflow as a Jupyter Notebook.  
5. In Sagemaker Studio, upload the Jupyter Notebook and corresponding flow file and proceed to execute all cells in the notebook. This will export the pipeline to Sagemaker Feature Store.
6. Open a Jupyter Notebook from Sagemaker Studio and import the pipeline you saved to Sagemaker Feature Store. Train a machine learning model to predict the `retained` field.
7. Evaluate your model by calculating the AUC ROC metric on a test dataset. 
8. Register your model and deploy it using Sagemaker Studio. Test the resulting API endpoint to ensure it works properly. 
9. Make sure to turn off all running services in AWS when you’re done. Submit the Jupyter Notebook used in Sagemaker.
