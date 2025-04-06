import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AutomatedModelSelector:
    def __init__(self):
        self.classification_models = {
            'Logistic Regression': LogisticRegression(max_iter=500),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'Support Vector Regressor': SVR(),
            'Decision Tree': DecisionTreeRegressor(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso()
        }
        
        self.scaler = StandardScaler()

    def analyze_dataset(self, X, y):
        """Analyze dataset characteristics to determine the best models."""
        n_samples, n_features = X.shape
        unique_classes = len(np.unique(y))
        
        # Calculate dataset characteristics
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_unique_classes': unique_classes,
            'feature_density': n_features / n_samples,
            'class_balance': min(np.bincount(y)) / max(np.bincount(y)) if len(np.unique(y)) > 1 else 1.0
        }
        
        return characteristics

    def select_models(self, X, y, target_type, max_models=3):
        """Select the most appropriate models based on dataset characteristics."""
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Analyze dataset
        characteristics = self.analyze_dataset(X_scaled, y)
        
        # Select models based on target type
        models = self.classification_models if target_type == "classification" else self.regression_models
        model_scores = {}
        
        # Evaluate each model using cross-validation
        for name, model in models.items():
            try:
                if target_type == "classification":
                    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                model_scores[name] = np.mean(scores)
            except:
                continue
        
        # Sort models by performance
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top models
        selected_models = []
        for name, score in sorted_models[:max_models]:
            model = models[name]
            selected_models.append((name, model))
        
        return selected_models, characteristics

    def get_model_recommendations(self, characteristics):
        """Generate recommendations based on dataset characteristics."""
        recommendations = []
        
        # Sample size recommendations
        if characteristics['n_samples'] < 1000:
            recommendations.append("Small dataset detected. Consider using simpler models to avoid overfitting.")
        elif characteristics['n_samples'] > 10000:
            recommendations.append("Large dataset detected. Complex models like Random Forest and Gradient Boosting are recommended.")
        
        # Feature density recommendations
        if characteristics['feature_density'] > 0.1:
            recommendations.append("High feature density detected. Consider using feature selection or regularization.")
        
        # Class balance recommendations
        if characteristics['class_balance'] < 0.5:
            recommendations.append("Imbalanced classes detected. Consider using class balancing techniques.")
        
        # Feature count recommendations
        if characteristics['n_features'] > 100:
            recommendations.append("High number of features detected. Consider using dimensionality reduction.")
        
        return recommendations 