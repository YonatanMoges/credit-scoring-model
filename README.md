
# Credit Scoring Model

## Project Overview
This project focuses on developing a credit scoring system using machine learning techniques. It includes data preprocessing, feature engineering, model training, evaluation, and deployment as a REST API.

---

## Table of Contents
1. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
2. [Feature Engineering](#feature-engineering)
3. [Modelling](#modelling)
4. [Model Serving API Call](#model-serving-api-call)
5. [Requirements](#requirements)
6. [Directory Structure](#directory-structure)

---

## Exploratory Data Analysis (EDA)
**Objective**: Understand the structure of the data, identify patterns, and detect any anomalies that need to be addressed before model training.

### Key Steps:
- **Overview of the Data**: 
  - Assessed the dataset structure including rows, columns, and data types.
- **Summary Statistics**: 
  - Provided insights into the central tendency, dispersion, and shape of the dataset.
- **Distribution of Numerical Features**: 
  - Visualized distributions to identify skewness and outliers.
- **Distribution of Categorical Features**: 
  - Analyzed the distribution of categorical variables to understand class frequencies.
- **Correlation Analysis**: 
  - Evaluated relationships between numerical features using correlation matrices.
- **Missing Values**: 
  - Identified missing values and strategized imputation or removal.
- **Outlier Detection**: 
  - Utilized box plots to detect outliers that could affect model performance.

Scripts: `scripts/eda.py`, Notebook: `notebooks/eda.ipynb`

---

## Feature Engineering
**Objective**: Transform and create features that will improve the model’s predictive performance.

### Key Steps:
- **Create Aggregate Features**:
  - Examples: Total Transaction Amount, Average Transaction Amount, Transaction Count, Standard Deviation of Transaction Amounts.
- **Extract Features**:
  - Time-based features such as Transaction Hour, Day, Month, and Year were extracted.
- **Encode Categorical Variables**:
  - Used One-Hot Encoding and Label Encoding for categorical variables.
- **Handle Missing Values**:
  - Missing values were handled by imputation using techniques like mean or median imputation.
- **Normalize/Standardize Numerical Features**:
  - Numerical features were standardized or normalized to ensure all features are on the same scale.
- **WoE Binning**:
  - Applied Weight of Evidence (WoE) binning for credit scoring.

Tools used: `xverse`, `woe`

Scripts: `scripts/feature_engineering.py`, Notebook: `notebooks/feature_engineering.ipynb`

---

## Modelling
**Objective**: Train and evaluate machine learning models for predicting credit defaults.

### Key Steps:
1. **Model Selection and Training**:
   - **Model Choices**:
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - Gradient Boosting Machines (GBM)
   - **Train the Models**: Trained models using a split of training and testing data.
  
2. **Hyperparameter Tuning**:
   - Used Grid Search and Random Search to fine-tune hyperparameters for improving model performance.
  
3. **Model Evaluation**:
   - Evaluated model performance based on:
     - **Accuracy**: Ratio of correct predictions to total observations.
     - **Precision**: Correctly predicted positive observations over total predicted positives.
     - **Recall**: Correctly predicted positives over all actual positive observations.
     - **F1 Score**: Weighted average of Precision and Recall.
     - **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

Scripts: `scripts/modeling.py`, Notebook: `notebooks/modeling.ipynb`

---

## Model Serving API Call
**Objective**: Serve the trained models through a REST API for real-time predictions.

### Key Steps:
1. **Framework**: Implemented the API using Flask.
2. **Load the Model**: The trained models from Task 4 are loaded using `joblib`.
3. **Define API Endpoints**:
   - `/predict`: This endpoint accepts input data in JSON format and returns predictions.
4. **Request Handling**:
   - Input data is preprocessed and fed into the model to generate predictions.
5. **Deployment**: The API is deployed on a local server for testing, but can also be deployed to a cloud service.

Sample API Request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "Recency": 2148,
    "Frequency": 1,
    "Monetary": 0.09099264705882354,
    "Seasonality": 0.0,
    "WoE_Recency": -2.336457635528553,
    "WoE_Frequency": -2.8600629859409428,
    "WoE_Monetary": -3.4128528088059653,
    "WoE_Seasonality": -3.9582216570009114,
    "RFMS_Score": 997.5
}'
```

## Requirements
To install the required packages, use:
```bash
pip install -r requirements.txt
```

## Directory Structure
```bash
├── notebook
│   ├── __init__.py
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
├── scripts
│   ├── __pycache__
│   ├── __init__.py
│   ├── api.py
│   ├── eda.py
│   ├── feature_engineering.py
│   └── modeling.py
├── src
├── tests
├── models
│   └── # Saved machine learning models (generated after training)
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE

```