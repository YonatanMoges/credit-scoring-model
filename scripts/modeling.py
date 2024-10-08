# scripts/modeling.py
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        self.models['Linear Regression'] = LinearRegression()
        self.models['Decision Tree'] = DecisionTreeRegressor()
        self.models['Random Forest'] = RandomForestRegressor()
        self.models['Gradient Boosting'] = GradientBoostingRegressor()

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.results[name] = model.predict(self.X_test)

    def hyperparameter_tuning(self):
        param_grid = {
            'Linear Regression': {},  # No hyperparameters to tune for Linear Regression
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
                grid_search = GridSearchCV(model, param_grid[name], scoring='r2', cv=5)
                grid_search.fit(self.X_train, self.y_train)
                best_model = grid_search.best_estimator_
                self.models[name] = best_model
                self.results[name] = best_model.predict(self.X_test)

    def evaluate_models(self):
        evaluation_metrics = {}
        for name, preds in self.results.items():
            mae = mean_absolute_error(self.y_test, preds)
            mse = mean_squared_error(self.y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, preds)

            evaluation_metrics[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2
            }
        return evaluation_metrics
