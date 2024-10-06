# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, data):
        self.data = data

    def overview(self):
        """Prints an overview of the dataset: shape, info, and first few rows."""
        print("Dataset Overview:")
        print("Shape:", self.data.shape)
        print("\nData Types and Missing Values:\n")
        print(self.data.info())
        print("\nFirst 5 Rows:\n", self.data.head())
        
    def summary_statistics(self):
        """Displays summary statistics for numerical columns."""
        print("Summary Statistics:\n", self.data.describe())
        
    def numerical_distribution(self):
        """Plots distribution of numerical features."""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            plt.figure(figsize=(10, 4))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

    def categorical_distribution(self, unique_threshold=20):
        """Plots count plots for categorical features with low cardinality."""
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Skip columns with high cardinality
            if self.data[col].nunique() > unique_threshold:
                print(f"Skipping column {col} due to high cardinality ({self.data[col].nunique()} unique values).")
                continue

            plt.figure(figsize=(10, 4))
            sns.countplot(y=self.data[col], order=self.data[col].value_counts().index)
            plt.title(f'Count Plot of {col}')
            plt.show()

            
    def correlation_analysis(self):
        """Generates a correlation heatmap for numerical features."""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64'])
        plt.figure(figsize=(12, 8))
        sns.heatmap(numerical_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.show()
        
    def missing_values(self):
        """Prints missing value counts for each column or indicates if no missing values are found."""
        missing = self.data.isnull().sum()
        missing = missing[missing > 0]
        
        if missing.empty:
            print("No missing values found in the dataset.")
        else:
            print("Missing Values:\n", missing)

        
    def outlier_detection(self):
        """Plots box plots for each numerical feature to identify outliers."""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=self.data[col])
            plt.title(f'Outlier Detection for {col}')
            plt.show()
