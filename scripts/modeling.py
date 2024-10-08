# scripts/modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.models = {}
        self.results = {}

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def choose_models(self):
        self.models['Logistic Regression'] = LogisticRegression(max_iter=1000)
        self.models['Decision Tree'] = DecisionTreeClassifier()
        self.models['Random Forest'] = RandomForestClassifier()
        self.models['Gradient Boosting'] = GradientBoostingClassifier()

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.results[name] = model.predict(self.X_test)

    def hyperparameter_tuning(self):
        param_grid = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            },
            'Decision Tree': {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }

        for name, model in self.models.items():
            if name in param_grid:
                grid_search = GridSearchCV(model, param_grid[name], scoring='accuracy', cv=5)
                grid_search.fit(self.X_train, self.y_train)
                best_model = grid_search.best_estimator_
                self.models[name] = best_model
                self.results[name] = best_model.predict(self.X_test)

    def evaluate_models(self):
        evaluation_metrics = {}
        for name, preds in self.results.items():
            accuracy = accuracy_score(self.y_test, preds)
            precision = precision_score(self.y_test, preds)
            recall = recall_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds)
            roc_auc = roc_auc_score(self.y_test, preds)

            evaluation_metrics[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc
            }
        return evaluation_metrics
