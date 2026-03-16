#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold cross-validation for Actin-Myosin models.")
    parser.add_argument("--training_dataset", required=True, help="Input training CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    training_dataset = pd.read_csv(args.training_dataset)
    le = LabelEncoder()
    training_dataset['Classification'] = le.fit_transform(training_dataset['Classification'])

    X = training_dataset.drop(columns=["Title", "Classification"])
    y = training_dataset["Classification"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "MLP": MLPClassifier(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    }

    results = []
    fold_scores_dict = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("model", model)
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro')
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"{name} - Mean F1-Score: {scores.mean():.4f} ± {scores.std():.4f}")

        fold_scores_dict[name] = scores

        results.append({
            "Model": name,
            "Fold_1": scores[0],
            "Fold_2": scores[1],
            "Fold_3": scores[2],
            "Fold_4": scores[3],
            "Fold_5": scores[4],
            "Mean_F1": mean_score,
            "Std_F1": std_score
        })
    fold_results_df = pd.DataFrame(results)
    save_path = os.path.join(args.output_dir, "CrossValidation_F1_PerFold.csv")
    fold_results_df.to_csv(save_path, index=False)
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Mean_F1",
        palette="Greys"
    )

    plt.errorbar(
        x=np.arange(len(results_df)),
        y=results_df["Mean_F1"],
        yerr=results_df["Std_F1"],
        fmt='none',
        capsize=5,
        color='black'
    )
    plt.ylabel("Macro F1-Score")
    plt.title("5-Fold Cross-Validation Macro F1-Score Comparison")
    plt.ylim(0.97, 1.01)
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "Model_Comparison_F1_Macro_5Fold.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
