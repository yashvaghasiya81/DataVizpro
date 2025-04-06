import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.model_selector import AutomatedModelSelector


def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_type):
    # Initialize the automated model selector
    selector = AutomatedModelSelector()
    
    # Select the best models for the dataset
    selected_models, characteristics = selector.select_models(X_train, y_train, target_type)
    
    results = {}
    confusion_matrices = {}
    model_recommendations = selector.get_model_recommendations(characteristics)
    
    # Train and evaluate selected models
    for name, model in selected_models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        if target_type == "classification":
            score = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            cm_path = os.path.join("static", f"{name.replace(' ', '_')}_confusion_matrix.png")
            plot_confusion_matrix(cm, name, cm_path)
            confusion_matrices[name] = cm_path
        else:  # regression
            score = r2_score(y_test, predictions)
        
        results[name] = {
            "accuracy" if target_type == "classification" else "r2_score": round(score, 4),
            "recommendations": model_recommendations
        }
    
    comparison_chart = plot_model_comparison(results, target_type)
    return results, confusion_matrices, comparison_chart, characteristics


def plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(results, target_type):
    metric = "accuracy" if target_type == "classification" else "r2_score"
    model_names = list(results.keys())
    scores = [results[model][metric] for model in model_names]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=scores, palette='viridis')
    plt.title(f'Model Comparison ({metric.capitalize()})')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45, ha='right')
    chart_path = os.path.join("static", "model_comparison.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path
