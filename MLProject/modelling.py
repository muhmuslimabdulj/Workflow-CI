"""
Modelling - Heart Disease Classification
Versi untuk MLflow Project (CI/CD)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os

def main():
    # Load data
    train_data = pd.read_csv('heart_disease_preprocessing/heart_disease_train.csv')
    test_data = pd.read_csv('heart_disease_preprocessing/heart_disease_test.csv')

    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    run_id = os.environ.get("MLFLOW_RUN_ID")

    if not run_id:
        mlflow.set_experiment("heart-disease-classification")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_id=run_id):
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

        print("\nModel training selesai!")

if __name__ == "__main__":
    main()
