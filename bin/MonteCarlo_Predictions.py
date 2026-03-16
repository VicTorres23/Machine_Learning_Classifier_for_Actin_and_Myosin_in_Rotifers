#!/usr/bin/env python3

import argparse
from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Run Monte Carlo Actin/Myosin classification.")
    parser.add_argument("--training_dataset", required=True, help="Training dataset CSV")
    parser.add_argument("--test_dataset", required=True, help="Unlabeled dataset CSV")
    parser.add_argument("--n_runs", type=int, default=100, help="Number of Monte Carlo runs")
    parser.add_argument("--output_csv", default="MonteCarlo_Prediction_Percentages.csv", help="Output CSV with prediction percentages")
    return parser.parse_args()

def PredictionCore(training_dataset_path, test_dataset_path):
    training_dataset = pd.read_csv(training_dataset_path)
    le = LabelEncoder()
    training_dataset['Classification'] = le.fit_transform(training_dataset['Classification'])
    X = training_dataset.drop(columns=["Title", "Classification"])
    Y = training_dataset["Classification"]

    X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_pca, y_train)

    mlp = MLPClassifier(max_iter=1000)
    mlp.fit(X_train_pca, y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_pca, y_train)

    test_data = pd.read_csv(test_dataset_path)
    titles = test_data["Title"]

    X_unlabeled = test_data.drop(columns=["Title"])
    X_unlabeled = X_unlabeled[X.columns]

    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    X_unlabeled_pca = pca.transform(X_unlabeled_scaled)

    lr_preds = le.inverse_transform(lr.predict(X_unlabeled_pca))
    mlp_preds = le.inverse_transform(mlp.predict(X_unlabeled_pca))
    xgb_preds = le.inverse_transform(xgb.predict(X_unlabeled_pca))

    results_df = pd.DataFrame({
        "Title": titles,
        "Logistic_Regression": lr_preds,
        "MLP": mlp_preds,
        "XGBoost": xgb_preds
    })
    return results_df

def main():
    args = parse_args()

    training_dataset = args.training_dataset
    test_dataset = args.test_dataset
    n_runs = args.n_runs

    prediction_counts = defaultdict(lambda:defaultdict(int))
    for i in range(n_runs):
        results = PredictionCore(training_dataset, test_dataset)
        for _, row in results.iterrows():
            title = row["Title"]
            for model in ["Logistic_Regression", "MLP", "XGBoost"]:
                prediction = row[model]
                prediction_counts[(title, model)][prediction] += 1
    records = []

    for (title, model), counts in prediction_counts.items():
        total = sum(counts.values())
        for class_label, count in counts.items():
            percentage = (count / total) * 100

            records.append({
                "Title": title,
                "Model": model,
                "Class": class_label,
                "Percentage": percentage
            })

    df = pd.DataFrame(records)

    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
