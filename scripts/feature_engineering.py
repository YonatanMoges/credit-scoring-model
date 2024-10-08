# scripts/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from xverse.transformer import WOE
from datetime import datetime

class FeatureEngineering:
    def __init__(self, df):
        self.df = df

    # Aggregate Features
    def create_aggregate_features(self):
        agg_features = self.df.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Dev_Transaction_Amount=('Amount', 'std')
        ).reset_index()
        self.df = pd.merge(self.df, agg_features, on='CustomerId', how='left')
        return self.df

    # Date-Time Features Extraction
    def extract_datetime_features(self):
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year
        return self.df

    # Encode Categorical Variables
    def encode_categorical_features(self, method="onehot"):
        if method == "onehot":
            ohe = OneHotEncoder(sparse=False, drop='first')
            cat_df = pd.DataFrame(ohe.fit_transform(self.df.select_dtypes(include='object')),
                                  columns=ohe.get_feature_names_out(self.df.select_dtypes(include='object').columns))
            self.df = pd.concat([self.df.drop(columns=self.df.select_dtypes(include='object').columns), cat_df], axis=1)
        elif method == "label":
            le = LabelEncoder()
            for col in self.df.select_dtypes(include='object').columns:
                self.df[col] = le.fit_transform(self.df[col])
        return self.df

    # Handle Missing Values
    def handle_missing_values(self, strategy="mean"):
        if strategy == "mean":
            self.df = self.df.fillna(self.df.mean())
        elif strategy == "median":
            self.df = self.df.fillna(self.df.median())
        elif strategy == "mode":
            self.df = self.df.fillna(self.df.mode().iloc[0])
        return self.df

    # Normalize/Standardize Numerical Features
    def scale_features(self, method="normalize"):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if method == "normalize":
            scaler = MinMaxScaler()
        elif method == "standardize":
            scaler = StandardScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return self.df

    # Weight of Evidence (WOE) and Information Value (IV)
    def woe_transformation(self, target_col):
        woe_transformer = WOE()
        self.df = woe_transformer.fit_transform(self.df, self.df[target_col])
        return self.df
