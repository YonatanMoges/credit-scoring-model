# scripts/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from xverse.transformer import WOE
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


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


class DefaultEstimator:
    def __init__(self, df):
        self.df = df

    # Step 1: Compute RFMS Features
    def compute_rfms(self):
        # Convert TransactionStartTime to datetime and ensure it is in UTC
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        if self.df['TransactionStartTime'].dt.tz is None:
            self.df['TransactionStartTime'] = self.df['TransactionStartTime'].dt.tz_localize('UTC')
        else:
            self.df['TransactionStartTime'] = self.df['TransactionStartTime'].dt.tz_convert('UTC')
        
        # Recency: Days since the most recent transaction
        recency_df = self.df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
        # Make datetime.now() timezone-aware and convert to UTC
        now_utc = datetime.now().astimezone().astimezone(recency_df['TransactionStartTime'].dt.tz)
        recency_df['Recency'] = (now_utc - recency_df['TransactionStartTime']).dt.days

        # Frequency: Count of transactions per customer
        frequency_df = self.df.groupby('CustomerId').size().reset_index(name='Frequency')

        # Monetary: Average transaction amount per customer
        monetary_df = self.df.groupby('CustomerId')['Amount'].mean().reset_index(name='Monetary')

        # Seasonality: Standard deviation of transaction amount
        seasonality_df = self.df.groupby('CustomerId')['Amount'].std().fillna(0).reset_index(name='Seasonality')

        # Merge RFMS features
        rfms_df = recency_df[['CustomerId', 'Recency']].merge(frequency_df, on='CustomerId')
        rfms_df = rfms_df.merge(monetary_df, on='CustomerId')
        rfms_df = rfms_df.merge(seasonality_df, on='CustomerId')
        self.rfms_df = rfms_df
        return rfms_df


    

    # Step 2: Calculate RFMS Score and Assign Good/Bad Labels
    def calculate_rfms_score(self):
        # Example RFMS scoring, this may need tuning
        self.rfms_df['RFMS_Score'] = (
            self.rfms_df['Recency'].rank(ascending=False) +
            self.rfms_df['Frequency'].rank() +
            self.rfms_df['Monetary'].rank() +
            self.rfms_df['Seasonality'].rank()
        )

        # Establish threshold visually or with a statistical method
        threshold = self.rfms_df['RFMS_Score'].median()  # Example threshold
        self.rfms_df['Default_Label'] = np.where(self.rfms_df['RFMS_Score'] > threshold, 'Good', 'Bad')
        return self.rfms_df

    # Step 3: Visualize RFMS to Define Boundary
    # Modify the plot_rfms_space function in DefaultEstimator class
    def plot_rfms_space(self, save_path="../data/rfms_space.png"):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.rfms_df, x='Frequency', y='Monetary', hue='Default_Label', palette=['red', 'green'])
        plt.title("RFMS Space - Frequency vs. Monetary")
        plt.xlabel("Frequency of Transactions")
        plt.ylabel("Average Transaction Amount (Monetary)")
        plt.legend(title="Default Label")
        
        # Save plot as an image file
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")

    # Step 4: Apply WoE Binning
    def woe_binning(self, target_col='Default_Label'):
        woe_transformer = WOE()
        # Map 'Good' and 'Bad' to binary values for WOE processing
        self.rfms_df[target_col] = self.rfms_df[target_col].map({'Good': 0, 'Bad': 1})
        self.rfms_df = woe_transformer.fit_transform(self.rfms_df, self.rfms_df[target_col])

        # Display IV values for each feature after WOE binning (helpful for feature selection)
        iv_values = woe_transformer.iv
        print("Information Values (IV) for features:", iv_values)
        return self.rfms_df, iv_values

    # Step 5: Build Scorecard
    def build_scorecard(self):
        # Calculate the scorecard points for each binned feature
        self.rfms_df['Score'] = (self.rfms_df['WOE_Recency'] +
                                 self.rfms_df['WOE_Frequency'] +
                                 self.rfms_df['WOE_Monetary'] +
                                 self.rfms_df['WOE_Seasonality']) * 20  # Scale for interpretability
        return self.rfms_df[['CustomerId', 'Score']]
