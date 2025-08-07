# Credit Card Fraud Detection

The goal of this project is to detect the Fraud cases in Credit card transactions

### Introduction about the Data:
Input Features:
* 'Time' - Time taken for each transaction (seconds elapsed between each transaction and the first transaction in the dataset)
* 'Amount' - Transaction amount
* 'V1-V28' - Features of integer data type

Target Feature:
* 'Class' - Output Feature | Which tells whether the transaction is normal or fraud type.

Initially the Exploratory Data Analysis (EDA) and model training is performed in Jupyter notebook files. 

## Approach for the Project:

1. Data Ingestion : 
    * In Data Ingestion phase the dataset is first read as csv. 
    * Then the data is split into training and testing and saved as csv files.

2. Data Transformation : 
    * In this phase a Pipeline is created.
    * For the Train data, Standard Scaling is performed and then SMOTE technique is performed to handle the imbalance data.
    * For the Test dataset, data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested on different machine learning algorithms such as RandomForest Classifier, XGBoost Classifier, AdaBoost Classifier, Catboost Classifier, LGBM Classifier. The best model found was Random Forest Classifier.
    * After this hyperparameter tuning is performed.
    * A final model is created with the RandomForest Classifier.
    * This model is saved as pickle file.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict the gemstone prices inside a Web Application.


